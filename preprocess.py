import pickle
import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_isolated_nodes
from torch.nn.functional import one_hot
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import LinkNeighborLoader
from copy import deepcopy
from torch_scatter import scatter
import torch_geometric
from pathlib import Path
from torch_geometric.data import HeteroData
from typing import List, Optional, Union
import math

class SubgraphSampler(object):
    """
    see https://pytorch-geometric.readthedocs.io/en/1.3.1/_modules/torch_geometric/data/sampler.html
    https: // pytorch - geometric.readthedocs.io / en / latest / _modules / torch_geometric / loader / dynamic_batch_sampler.html
    Makes edges based on edge index. Equal length of edge index per batch
    Problem: overlaps in nodes. Some nodes appear in multiple batches.
    todo: put batch size in args and pass
    """
    def __init__(self, data, batch_size=1000, shuffle=True, neg_sampling_per_type=True):
        self.batch_size = batch_size
        self.data = data
        self.current_index = 0
        self.e_id_start = 0
        self.num_batches = round(data.num_edges / self.batch_size)

        if shuffle:
            # shuffle the edge_index before splitting into batches
            idx = torch.randperm(data.edge_index.shape[1])
            self.data.edge_index = data.edge_index[:, idx]
            self.data.edge_label_index = data.edge_label_index[:, idx]
            self.data.edge_label = data.edge_label[idx]

        if neg_sampling_per_type:
            # sample per positive edge type link in edge index a negative edge
            neg_edge_index = torch.zeros_like(data.edge_index)
            for rel in torch.unique(data.edge_label):
                pos = torch.where(data.edge_label == rel)[0]
                edge_index_filtered = data.edge_index[:, pos]
                neg_edge_index_type = negative_sampling(edge_index_filtered) # , data.num_nodes, num_neg_samples=len(data.train_edge_index[1]))
                neg_edge_index[:, pos] = neg_edge_index_type
            self.data.neg_edge_index = neg_edge_index

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        if self.current_index <= self.num_batches:
            batch = deepcopy(self.data)
            # take the first batch_size links
            edge_index = self.data.edge_label_index[:, self.e_id_start:self.e_id_start+self.batch_size]
            edge_label = self.data.edge_label[self.e_id_start:self.e_id_start+self.batch_size]

            # keep the node ids of nodes in negative edge index
            if hasattr(batch, 'neg_edge_index'):
                neg_edge_index = self.data.neg_edge_index[:,
                                                self.e_id_start:self.e_id_start + self.batch_size]
                edge_index = torch.cat([neg_edge_index, edge_index], dim=1)
                edge_label = torch.cat([edge_label, edge_label])

            edge_index, edge_label, mask = remove_isolated_nodes(edge_index, edge_label, num_nodes=batch.num_nodes)

            batch.edge_index = edge_index
            batch.edge_label = edge_label
            batch.pos_edge_index = edge_index[:, :self.batch_size]
            batch.neg_edge_index = edge_index[:, self.batch_size:]
            batch.x = batch.x[mask, :]
            batch.y = batch.y[mask]
            batch.num_nodes = sum(mask)
            batch.num_classes = self.data.num_classes
            self.current_index += 1
            self.e_id_start += self.batch_size
            return batch
        raise StopIteration


class MultiRelationalSampler(object):
    """
    The idea is to filter the graph by the edge_type and use these edge types as batches.
    However, if the number of links per type is still too large (> maxbatchsize), several batches per types are made
    # todo the number of edges might still be too large and return OOM, set a maximum batch size to solve this
    """
    def __init__(self, data):
        self.data = data
        self.current_type = 0
        self.num_batches = self.data.num_relations

    def __iter__(self):
        return self

    def __len__(self):
        return self.data.num_relations

    def __next__(self):
        if self.current_type <= self.data.num_relations:
            batch = deepcopy(self.data)
            ids = torch.where(self.data.edge_label == self.current_type)[0]
            filtered_edge_index = self.data.edge_label_index[:, ids]
            filtered_edge_label = self.data.edge_label[ids]
            batch.edge_label_index, batch.edge_label, mask = remove_isolated_nodes(filtered_edge_index, filtered_edge_label, num_nodes=self.data.num_nodes)
            batch.x = batch.x[mask, :]
            batch.y = batch.y[mask]
            batch.num_nodes = sum(mask)
            self.current_type += 1
            return batch
        raise StopIteration


class SplitRandomLinks(BaseTransform):
    def __init__(self, num_val=0.2, num_test=0.2):
        self.num_val = num_val
        self.num_test = num_test

    def __call__(self, data):
        """ create subgraphs of data with disjunct sets of edge index """
        # train_data, val_data, test_data = deepcopy(data), deepcopy(data), deepcopy(data)
        num_edges = len(data.edge_index[1])
        num_val_edges = round(self.num_val * num_edges)
        num_test_edges = round(self.num_test * num_edges)
        num_train_edges = num_edges - num_val_edges - num_test_edges
        rand_perm = torch.randperm(num_edges)
        train_mask = torch.cat([torch.ones(num_train_edges, dtype=torch.bool), torch.zeros(num_val_edges + num_test_edges, dtype=torch.bool)])
        val_mask = torch.cat([torch.zeros(num_train_edges, dtype=torch.bool), torch.ones(num_val_edges, dtype=torch.bool), torch.zeros(num_test_edges, dtype=torch.bool)])
        test_mask = torch.cat([torch.zeros(num_train_edges + num_val_edges, dtype=torch.bool), torch.ones(num_test_edges, dtype=torch.bool)])

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        train_data = data.edge_subgraph(rand_perm[train_mask])
        val_data = data.edge_subgraph(rand_perm[val_mask])
        test_data = data.edge_subgraph(rand_perm[test_mask])

        return data, train_data, val_data, test_data


def get_data(args):
    """ returns data object """
    data = WikiAlumniData(args).preprocess()
    return data


def add_edge_type_dict(data):
    """ adds attribute edge type dict """
    edge_types_str = ['about', 'actor', 'affiliation', 'author', 'award', 'birthplace',
                      'character', 'children', 'competitor', 'composer', 'contributor',
                      'creator', 'deathplace', 'director', 'editor', 'founder', 'gender',
                      'hasOccupation', 'homeLocation', 'knowsLanguage', 'lyricist', 'memberOf',
                      'musicBy', 'nationality', 'parent', 'producer', 'publisher', 'spouse',
                      'worksFor']
    edge_type_dict = {}
    for k, item in enumerate(edge_types_str):
        edge_type_dict[k] = item
    data._edge_type_dict = edge_type_dict
    return data


def subgraph_by_edge_type(data, edge_list: [str], keep_all_nodes=False):
    """
    filter edges by certain types defined in edge list
    keeps
    """
    filter_mask = sum(data.edge_type == list(data.edge_type_dict.values()).index(i) for i in edge_list).bool()
    data.edge_type = data.edge_type[filter_mask]
    data.edge_index = data.edge_index[:, filter_mask]
    node_mask = remove_isolated_nodes(data.edge_index, data.edge_type, num_nodes=data.num_nodes)[2]
    if not keep_all_nodes:
        data.x = data.x[node_mask]
        data.y = data.y[node_mask]
    return data


def add_edge_common(data, edge_list, path=Path.cwd() / 'Wikialumni' / 'augmented.pkl'):
    """
     Add new type of edge between two "people" nodes for knowing if they have a common third "other" node inn common.
     Each element of the pair of "people" node is linked with the same edge type on this third "other" node in common.
     A list variable is expected as list_edge input.
     Remark: some edge types introduce MANY edges and the graph gets very large (see overview in documentation)
     """
    if path.exists():
        file = open(path, 'rb')
        data = pickle.load(file)
        print('loaded file augmented ')
        return data

    else:
        print('Create wikialumni_augmented. This can take a while .... ')
        for i, edge_type_str in enumerate(edge_list):

            if edge_type_str in ["deathplace", "homeLocation", "hasOccupation", "nationality", "birthplace", "affiliation",
                                 "gender", "knowsLanguage", "memberOf", "award", "worksFor"]:
                people = 0
                other = 1
            elif (edge_type_str in ["about", "actor", "author", "director", "character", "competitor", "composer",
                                    "contributor", "creator", "editor", "founder", "lyricist", "musicBy", "producer",
                                    "publisher"]) or (type(edge_type_str) == list):
                people = 1
                other = 0
            else:
                raise ValueError('Edge type %s not found' % edge_type_str)

            id_type_edge = len(data.edge_type_dict)

            if isinstance(edge_type_str, str):
                data.edge_type_dict[id_type_edge] = "same" + edge_type_str.capitalize()
            elif isinstance(edge_type_str, list):
                capitalized_strings = [s.capitalize() for s in edge_type_str]
                data.edge_type_dict[id_type_edge] = "same" + "Or".join(capitalized_strings)

            edge_type = [i for i, et in enumerate(data.edge_type_dict.values()) if et in edge_type_str]
            edge_type_index = torch.cat([(data.edge_type == t).nonzero(as_tuple=False).squeeze() for t in edge_type])
            people_nodes = data.edge_index[people, edge_type_index]
            other_nodes = data.edge_index[other, edge_type_index]
            pairs = torch.empty(0, 2, dtype=torch.int)
            for value in torch.unique(other_nodes):
                indices = torch.where(other_nodes == value)[0]
                corresponding_values = people_nodes[indices]

                if corresponding_values.size(0) <= 1:
                    continue
                else:
                    nodes_pairs = torch.combinations(corresponding_values, 2)
                    pairs = torch.cat((pairs, nodes_pairs), dim=0)
                    pairs = torch.unique(pairs, dim=0)

            print(f'{pairs.size(0)} links added for relation type {edge_type_str}')
            data.edge_type = torch.cat((data.edge_type, torch.full((pairs.size(0),), id_type_edge)))
            data.edge_index = torch.cat([data.edge_index, torch.transpose(pairs, 0, 1)], dim=1)
            data.num_classes += 1
            if data.edge_weight is not None:
                data.edge_weight = torch.cat([data.edge_weight, torch.ones_like(pairs[:, 0], dtype=torch.float32)])


        with open(path, 'wb') as handle:
            print('created file augmented')
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return data


class WikiAlumniData:
    def __init__(self, args):
        self.path_wiki = "./WikiAlumni"
        self.same_edge = args.same_edge
        self.num_val = args.num_val
        self.num_test = args.num_test
        self.batch_size = args.batch_size
        self.to_hetero = False

    def preprocess(self):
        try:
            file = open(self.path_wiki + '/raw.pkl', 'rb')
            data = pickle.load(file)[0]
        except (FileNotFoundError, IOError):
            print("Put the pickle file in directory")
            data = None

        data.x = data.x.type(torch.float32)
        data.num_classes = int(len(torch.unique(data.y)))
        data.edge_label = data.edge_type
        data.num_relations = int(len(torch.unique(data.edge_type)))

        del data['tr_ent_idx']
        del data['val_ent_idx']
        del data['test_ent_idx']
        del data['edge_type']

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
                                    num_val=0.1,
                                    num_test=0.3,
                                    add_negative_train_samples=False,
                                    neg_sampling_ratio=0.0)
        train_data, val_data, test_data = transform(data)

        # don't know if this is correct
        # todo is this even needed
        # data.train_edge_index = train_data.edge_label_index
        # data.train_edge_label = train_data.edge_label
        # data.val_edge_index = val_data.edge_index
        # data.val_edge_label = val_data.edge_label
        # data.test_edge_index = test_data.edge_label_index
        # data.test_edge_label = test_data.edge_label

        data.train_data = train_data
        data.val_data = val_data
        data.test_data = test_data
        return data
