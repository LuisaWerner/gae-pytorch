import utils
from utils import *
import math
from copy import copy
from torch_geometric.utils import one_hot, negative_sampling
# from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data


def shuffle_edges(data: Data) -> Data:
    """
    randomly shuffles the edge_index and edge_type of a data object
    should be done before splitting into train/val/test or before splitting batches
    """
    idx = torch.randperm(data.edge_index.shape[1])
    data['edge_index'] = data.edge_index[:, idx]
    data['edge_type'] = data.edge_type[idx]
    if hasattr(data, 'edge_label_index'):
        data['edge_label_index'] = data.edge_label_index[:, idx]
    if hasattr(data, 'edge_label'):
        data['edge_label'] = data.edge_label[idx]
    return data


class RandomLinkSplit(BaseTransform):
    """
    splits links into train, valid, test links
    given num_val, num_test as float numbers
    there is no overlap
    """

    def __init__(self, num_val: float = 0.1, num_test: float = 0.1):
        assert num_val + num_test < 1.0
        self.num_val = num_val
        self.num_test = num_test

    def __call__(self, data):
        train_data, val_data, test_data = copy(data), copy(data), copy(data)

        perm = torch.randperm(data.edge_index.size(1))

        num_val = int(self.num_val * perm.numel())
        num_test = int(self.num_test * perm.numel())

        num_train = perm.numel() - num_val - num_test
        if num_train <= 0:
            raise ValueError("Insufficient number of edges for training")

        train_edges = perm[:num_train]
        val_edges = perm[num_train:num_train + num_val]
        test_edges = perm[num_train + num_val:]

        train_data['edge_index'] = data.edge_index[:, train_edges]
        val_data['edge_index'] = data.edge_index[:, val_edges]
        test_data['edge_index'] = data.edge_index[:, test_edges]

        if hasattr(data, 'edge_type'):
            train_data['edge_type'] = data.edge_type[train_edges]
            val_data['edge_type'] = data.edge_type[val_edges]
            test_data['edge_type'] = data.edge_type[test_edges]

        return train_data, val_data, test_data


class SubgraphSampler(object):
    """
    see https://pytorch-geometric.readthedocs.io/en/1.3.1/_modules/torch_geometric/data/sampler.html
    https: // pytorch - geometric.readthedocs.io / en / latest / _modules / torch_geometric / loader / dynamic_batch_sampler.html
    Samples batches based on edge index. Equal length of edge index per batch
    Some nodes may appear in multiple batches but each link only appears once
    """

    def __init__(self, data: Data, batch_size: int = 1000, shuffle: bool = True, neg_sampling_ratio: float = 1.0,
                 neg_sampling_per_type: bool = False, drop_last: bool = True):
        """
        @param data: the data object from which batches will be sampled
        @param batch_size: the number of links per batch
        @param shuffle: shuffle edges before splitting in batches (should be done in training only)
        @param neg_sampling_ratio: fraction of edge_index number of negative edges sampled
        @param neg_sampling_per_type: if True, a link (a,r,b) is considered as false even though another link (a,r2,b)
        between two nodes exists
        @param drop_last: drops the last batch with an uneven number
        """
        self.batch_size = batch_size
        self.num_neg_samples = math.floor(neg_sampling_ratio * data.edge_index.shape[1])
        self.data = data
        self.current_index = 0
        self.e_id_start = 0
        self.num_batches = math.floor(data.num_edges / self.batch_size) if drop_last \
            else math.ceil(data.num_edges / self.batch_size)
        self.neg_batch_size = math.floor(self.num_neg_samples / self.num_batches)

        if shuffle:
            data = shuffle_edges(data)

        if neg_sampling_per_type:
            # sample per positive edge type link in edge index a negative edge
            neg_edge_index = torch.zeros_like(data.edge_index)
            for rel in torch.unique(data.edge_type):
                pos = torch.where(torch.Tensor(data.edge_type == rel))[0]
                edge_index_filtered = data.edge_index[:, pos]
                neg_edge_index_type = negative_sampling(
                    edge_index_filtered)  # , data.num_nodes, num_neg_samples=len(data.train_edge_index[1]))
                neg_edge_index[:, pos] = neg_edge_index_type
            self.data['neg_edge_index'] = neg_edge_index[:, :self.num_neg_samples]
        else:
            self.data['neg_edge_index'] = negative_sampling(data.edge_index)[:, :self.num_neg_samples]

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        """
        iteratively samples the next batch
        """
        if self.current_index <= self.num_batches:
            batch = deepcopy(self.data)
            # take the first batch_size links
            edge_index = self.data.edge_index[:, self.e_id_start:self.e_id_start + self.batch_size]
            edge_type = self.data.edge_type[self.e_id_start:self.e_id_start + self.batch_size]

            # keep the node ids of nodes in negative edge index
            if hasattr(batch, 'neg_edge_index'):
                neg_edge_index = self.data.neg_edge_index[:, self.e_id_start:self.e_id_start + self.neg_batch_size]
                neg_edge_type = torch.ones(neg_edge_index.shape[1], dtype=torch.int64) * (batch.num_relations)
                edge_index = torch.cat([neg_edge_index, edge_index], dim=1)
                edge_type = torch.cat([edge_type, neg_edge_type])

            edge_index, edge_type, mask = remove_isolated_nodes(edge_index, edge_type, num_nodes=batch.num_nodes)
            batch['edge_index'] = edge_index
            batch['edge_type'] = edge_type

            batch['pos_edge_index'] = edge_index[:, :self.batch_size]
            batch['neg_edge_index'] = edge_index[:, self.batch_size:]

            batch['edge_label'] = one_hot(batch.edge_type, num_classes=batch.num_relations + 1)
            batch['x'] = batch.x[mask, :]
            batch['y'] = batch.y[mask]
            batch['num_nodes'] = sum(mask)
            batch['num_classes'] = self.data.num_classes

            self.current_index += 1
            self.e_id_start += self.batch_size
            return batch
        raise StopIteration


def get_data(args):
    """ returns data object """
    if args.data == 'WikiAlumni':
        data = WikiAlumniData(args).preprocess()
    elif args.data == 'Family':
        data = FamilyData(args).preprocess()
    return data


class WikiAlumniData:
    def __init__(self, args):
        self.path_wiki = "./WikiAlumni"
        self.same_edge = args.same_edge
        self.num_val = args.num_val
        self.num_test = args.num_test
        self.to_hetero = False

    def preprocess(self):
        """
        loads the data object
        x are the node features, y the node labels, edge_index the message passing edges,
        edge_type the type of the relation
        after the random link split edge_label_index are the supervision edges and edge_label are the types
        train.edge_label_index + valid.edge_label_index + test.edge_label_index sum up to total edges in data.
        clarification on random link split: https: // github.com / pyg - team / pytorch_geometric / issues / 3668
        """
        try:
            file = open(self.path_wiki + '/raw.pkl', 'rb')
            data = pickle.load(file)[0]
        except (FileNotFoundError, IOError):
            print("Put the pickle file in directory")
            data = None

        data['x'] = data.x.type(torch.float32)
        data['num_classes'] = int(len(torch.unique(data.y)))
        data['num_relations'] = int(len(torch.unique(data.edge_type)))

        del data['tr_ent_idx']
        del data['val_ent_idx']
        del data['test_ent_idx']

        # create initial dictionary for edges
        data = add_edge_type_dict_wiki(data)

        # create heterodata object
        if self.to_hetero:
            hetero_data = data.to_heterogeneous(edge_type=data.edge_type)

        # add new edges
        if len(self.same_edge) != 0:
            data = add_edge_common(data, self.same_edge)

        # data = subgraph_by_edge_type(data, ["children", "parent"])

        transform = RandomLinkSplit(num_val=self.num_val, num_test=self.num_test)
        data.train_data, data.val_data, data.test_data = transform(data)

        return data


class FamilyData:
    def __init__(self, args):
        self.path = "./family"

    def triples_to_data(self, key='all') -> Data:
        """
        loads triples files in txt and creates a PyG Data object with attributes
        edge_type and edge_index
        """
        try:
            triples = open(self.path + '/' + key + '.txt').readlines()

        except (FileNotFoundError, IOError):
            print(f"File {key}.txt in {self.path} not found. Put the pickle file in directory")
            triples = None

        edge_index, edge_type = [], []
        for line in triples:
            head, relation, tail = line.split('\t')
            if tail.endswith('\n'):
                tail = tail[:-1]
            edge_index.append([int(head), int(tail)])
            edge_type.append(relation)

        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_type_dict = dict(zip(set(edge_type), range(len(set(edge_type)))))
        edge_type = torch.tensor([int(edge_type_dict.get(x)) for x in edge_type])

        data = Data()
        data['edge_index'] = edge_index
        data['edge_type'] = edge_type
        data._edge_type_dict = edge_type_dict
        return data

    def preprocess(self):
        """
        loads the data object
        x are the node features, y the node labels, edge_index the message passing edges,
        edge_type the type of the relation
        after the random link split edge_label_index are the supervision edges and edge_label are the types
        train.edge_label_index + valid.edge_label_index + test.edge_label_index sum up to total edges in data.
        clarification on random link split: https: // github.com / pyg - team / pytorch_geometric / issues / 3668
        """

        data = self.triples_to_data()

        # todo we might do our own split
        # todo how is this split done?
        data['train'] = self.triples_to_data('train')
        data['valid'] = self.triples_to_data('valid')
        data['test'] = self.triples_to_data('test')

        return data


