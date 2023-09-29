from utils import *
import math
from copy import copy
from torch_geometric.utils import one_hot, negative_sampling
# from torch_geometric.transforms import RandomLinkSplit


def shuffle_edges(data):
    idx = torch.randperm(data.edge_index.shape[1])
    data['edge_index'] = data.edge_index[:, idx]
    data['edge_type'] = data.edge_type[idx]
    if hasattr(data, 'edge_label_index'):
        data['edge_label_index'] = data.edge_label_index[:, idx]
    if hasattr(data, 'edge_label'):
        data['edge_label'] = data.edge_label[idx]
    return data


class RandomLinkSplit(object):
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
    Makes edges based on edge index. Equal length of edge index per batch
    Problem: overlaps in nodes. Some nodes appear in multiple batches.
    """

    def __init__(self, data, batch_size=1000, shuffle=True, neg_sampling_ratio=1.0, neg_sampling_per_type=False, drop_last=True):
        self.batch_size = batch_size
        self.num_neg_samples = math.floor(neg_sampling_ratio * data.edge_index.shape[1])
        self.data = data
        self.current_index = 0
        self.e_id_start = 0
        self.num_batches = math.floor(data.num_edges / self.batch_size) if drop_last \
            else math.ceil(data.num_edges / self.batch_size)
        self.neg_batch_size = math.floor(self.num_neg_samples/self.num_batches)

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
    data = WikiAlumniData(args).preprocess()
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
        data = add_edge_type_dict(data)

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
