from utils import *
import math
from torch_geometric.utils import one_hot, negative_sampling
from torch_geometric.transforms import RandomLinkSplit


class SubgraphSampler(object):
    """
    see https://pytorch-geometric.readthedocs.io/en/1.3.1/_modules/torch_geometric/data/sampler.html
    https: // pytorch - geometric.readthedocs.io / en / latest / _modules / torch_geometric / loader / dynamic_batch_sampler.html
    Makes edges based on edge index. Equal length of edge index per batch
    Problem: overlaps in nodes. Some nodes appear in multiple batches.
    todo: put batch size in args and pass
    """
    def __init__(self, data, batch_size=1000, shuffle=True, neg_sampling_per_type=False, drop_last=True):
        self.batch_size = batch_size
        self.data = data
        self.current_index = 0
        self.e_id_start = 0
        self.num_batches = math.floor(data.num_edges / self.batch_size) if drop_last \
            else math.ceil(data.num_edges / self.batch_size)

        if shuffle:
            # shuffle the edge_index before splitting into batches
            idx = torch.randperm(data.edge_index.shape[1])
            self.data['edge_index'] = data.edge_index[:, idx]
            self.data['edge_type'] = data.edge_type[idx]
            self.data['edge_label_index'] = data.edge_label_index[:, idx]
            self.data['edge_label'] = data.edge_label[idx]

        if neg_sampling_per_type:
            # sample per positive edge type link in edge index a negative edge
            neg_edge_index = torch.zeros_like(data.edge_index)
            for rel in torch.unique(data.edge_type):
                pos = torch.where(torch.Tensor(data.edge_type == rel))[0]
                edge_index_filtered = data.edge_index[:, pos]
                neg_edge_index_type = negative_sampling(edge_index_filtered) # , data.num_nodes, num_neg_samples=len(data.train_edge_index[1]))
                neg_edge_index[:, pos] = neg_edge_index_type
            self.data['neg_edge_index'] = neg_edge_index
        else:
            self.data['neg_edge_index'] = negative_sampling(data.edge_index)


    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        if self.current_index <= self.num_batches:
            batch = deepcopy(self.data)
            # take the first batch_size links
            edge_index = self.data.edge_index[:, self.e_id_start:self.e_id_start+self.batch_size]
            edge_type = self.data.edge_type[self.e_id_start:self.e_id_start+self.batch_size]

            # keep the node ids of nodes in negative edge index
            if hasattr(batch, 'neg_edge_index'):
                neg_edge_index = self.data.neg_edge_index[:,
                                                self.e_id_start:self.e_id_start + self.batch_size]
                edge_index = torch.cat([neg_edge_index, edge_index], dim=1)
                # edge_type = torch.cat([edge_type, edge_type])
                neg_type = torch.ones_like(edge_type) * (batch.num_relations) # added additional type for "we don't know type"
                edge_type = torch.cat([edge_type, neg_type])

            edge_index, edge_type, mask = remove_isolated_nodes(edge_index, edge_type, num_nodes=batch.num_nodes)
            # batch = torch_geometric.data.Data()
            batch['edge_index'] = edge_index
            batch['edge_type'] = edge_type

            batch['pos_edge_index'] = edge_index[:, :self.batch_size]
            if hasattr(batch, 'neg_edge_index'):
                batch['neg_edge_index'] = edge_index[:, self.batch_size:]

            # put here label creation
            # neg_edge_label = torch.zeros(batch.neg_edge_index.shape[1], batch.num_relations)
            # pos_edge_label = one_hot(batch.edge_type[:self.batch_size], num_classes=batch.num_relations)
            # batch['edge_label'] = torch.cat([pos_edge_label, neg_edge_label]) # not needed anymore in this setting

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

        # this gives links without negative samples and keeps the edge type in edge_label
        transform = RandomLinkSplit(is_undirected=False,
                                    num_val=self.num_val,
                                    num_test=self.num_test,
                                    add_negative_train_samples=False,
                                    neg_sampling_ratio=0.0)
        train_data, val_data, test_data = transform(data)

        data.train_data = train_data
        data.val_data = val_data
        data.test_data = test_data
        return data
