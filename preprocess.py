import pickle
import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.transforms import BaseTransform
from torch_geometric.loader import LinkNeighborLoader
from copy import copy
import torch_geometric

from typing import List, Optional, Union


class SubgraphSampler(object):
    """
    see https://pytorch-geometric.readthedocs.io/en/1.3.1/_modules/torch_geometric/data/sampler.html
    https: // pytorch - geometric.readthedocs.io / en / latest / _modules / torch_geometric / loader / dynamic_batch_sampler.html
    """
    def __init__(self):
        raise NotImplementedError


class SplitRandomLinks(BaseTransform):
    def __init__(self, num_val=0.2, num_test=0.2):
        self.num_val = num_val
        self.num_test = num_test

    def __call__(self, data):
        """ create subgraphs of data with disjunct sets of edge index """
        train_data, val_data, test_data = copy(data), copy(data), copy(data)
        num_edges = len(data.edge_index[1])
        num_val_edges = round(self.num_val * num_edges)
        num_test_edges = round(self.num_test * num_edges)
        num_train_edges = num_edges - num_val_edges - num_test_edges
        rand_perm = torch.randperm(num_edges)
        train_mask = torch.cat([torch.ones(num_train_edges, dtype=torch.bool), torch.zeros(num_val_edges + num_test_edges, dtype=torch.bool)])
        val_mask = torch.cat([torch.zeros(num_train_edges, dtype=torch.bool), torch.ones(num_val_edges, dtype=torch.bool), torch.zeros(num_test_edges, dtype=torch.bool)])
        test_mask = torch.cat([torch.zeros(num_train_edges + num_val_edges, dtype=torch.bool), torch.ones(num_test_edges, dtype=torch.bool)])

        train_data = data.edge_subgraph(rand_perm[train_mask])
        val_data = data.edge_subgraph(rand_perm[val_mask])
        test_data = data.edge_subgraph(rand_perm[test_mask])

        return train_data, val_data, test_data



def get_data(args):
    """ returns data object """
    data = WikiAlumniData(args).preprocess()
    return data


def add_edge_common(data, edge_list):
    """
     Add new type of edge between two "people" nodes for knowing if they have a common third "other" node inn common.
     Each element of the pair of "people" node is linked with the same edge type on this third "other" node in common.
     A list variable is expected as list_edge input.
     """
    for i, edge_type_str in enumerate(edge_list):

        if edge_type_str in ["deathplace", "homeLocation", "hasOccupation", "nationality", "birthplace", "affiliation",
                             "gender", "knowsLanguage", "memberOf"]:
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

        data.edge_type = torch.cat((data.edge_type, torch.full((pairs.size(0),), id_type_edge)))
        data.edge_index = torch.cat([data.edge_index, torch.transpose(pairs, 0, 1)], dim=1)
        data.edge_weight = torch.cat([data.edge_weight, torch.ones_like(pairs[:, 0], dtype=torch.float32)])
    return data


class WikiAlumniData:
    def __init__(self, args):
        self.path_wiki = "./WikiAlumni"
        self.same_edge = args.same_edge
        self.num_val = args.num_val
        self.num_test = args.num_test

    def preprocess(self):
        try:
            file = open(self.path_wiki + '/raw.pkl', 'rb')
            data = pickle.load(file)[0]
        except (FileNotFoundError, IOError):
            print("Put the pickle file in directory")
            data = None

        data.x = data.x.type(torch.float32)

        del data['tr_ent_idx']
        del data['val_ent_idx']
        del data['test_ent_idx']

        # add edge types as dictionary
        if len(self.same_edge) != 0:
            edge_types_str = ['about', 'actor', 'affiliation', 'author', 'award', 'birthplace',
                              'character', 'children', 'competitor', 'composer', 'contributor',
                              'creator', 'deathplace', 'director', 'editor', 'founder', 'gender',
                              'hasOccupation', 'homeLocation', 'knowsLanguage', 'lyricist', 'memberOf',
                              'musicBy', 'nationality', 'parent', 'producer', 'publisher', 'spouse',
                              'worksFor']
            edge_type_dict = {}
            for k, item in enumerate(edge_types_str):
                edge_type_dict[k] = item
            data.edge_type_dict = edge_type_dict
            data = add_edge_common(data, self.same_edge)

        transform = SplitRandomLinks()
        train_data, val_data, test_data = transform(data)

        # samplers for batch learning
        # train_loader = SubgraphSampler(data, batch_size=128)
        # todo split into batches ? write subgraph into list
        # split edge index and then create iterable of subgraphs? 

        return data, train_data, val_data, test_data
