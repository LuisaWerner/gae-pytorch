from tqdm import tqdm
import pickle
import torch
from torch_geometric.transforms import BaseTransform, RandomLinkSplit
from torch_geometric.utils import remove_isolated_nodes, negative_sampling
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
import random
from torch_geometric.data import Data
import torch.nn.functional as F



class NegativeSampler:
    def __init__(self, data, n_corrupted):
        self.edge_index = data.edge_index
        self.edge_type = data.edge_type
        self.triples = torch.vstack([self.edge_index[0, :], self.edge_type, self.edge_index[1,:]])
        self.n_corrupted = n_corrupted

    def get_corrupted_triples(self, triple):
        head, rel, tail = triple

        nodes = torch.unique(self.edge_index.reshape([-1]))
        rels = torch.unique(self.edge_type)

        # randomly sample nodes
        sampled_heads = torch.randperm(len(nodes))[:self.n_corrupted]
        sampled_tails = torch.randperm(len(nodes))[:self.n_corrupted]
        sampled_rels = torch.Tensor([random.choice(list(rels)) for i in range(self.n_corrupted)])

        cor_head = torch.vstack([sampled_heads, torch.ones_like(sampled_heads) * rel, torch.ones_like(sampled_heads) * tail])
        cor_tail = torch.vstack([torch.ones_like(sampled_heads) * head, torch.ones_like(sampled_heads) * rel, sampled_tails])
        cor_rel = torch.vstack([torch.ones_like(sampled_heads) * head, sampled_rels.int(), torch.ones_like(sampled_heads) * tail])

        neg_samples = torch.hstack([cor_head, cor_tail, cor_rel])
        # todo filter false negatives
        mask = torch.ones_like(neg_samples[0], dtype=torch.bool)
        return neg_samples


class Evaluator:
    def __init__(self, ground_truth: torch.Tensor, out: torch.Tensor, model, data: Data, num_neg_samples: int = 100):
        self.ground_truth = ground_truth
        self.out = out
        self.model = model
        self.data = data
        self.num_neg_samples = num_neg_samples

    def roc_auc_score(self):
        return roc_auc_score(self.ground_truth.cpu().numpy(), self.out.cpu().numpy())


    @torch.no_grad()
    def compute_rank(self):
        """ for each triple, flip head and tail and do n negative samples and see how it ranks """
        edge_index, edge_type = self.data.edge_index, self.data.edge_type

        triples = torch.vstack([edge_index[0, :], edge_type, edge_index[1,:]])
        sampler = NegativeSampler(self.data, 10)

        for i in range(edge_index.numel()):
            ranks = []
            (head, tail), rel = edge_index[:, i], edge_type[i]
            neg_triples = sampler.get_corrupted_triples((head, rel, tail))
            true_triple = torch.vstack([head, rel, tail])
            all_triples = torch.hstack([true_triple, neg_triples]) # the first triple is the true triple

             # todo node ids don't correspond
            eval_batch = Data()
            edge_index = torch.vstack([all_triples[0, :], all_triples[2, :]])
            edge_type = all_triples[1, :]

            edge_index, edge_type, mask = remove_isolated_nodes(edge_index, edge_type, num_nodes=self.data.num_nodes)
            eval_batch['edge_index'] = eval_batch['pos_edge_index'] = edge_index
            eval_batch['edge_type'] = edge_type
            # node ids have to correspond 
            eval_batch['node_ids'] = torch.unique(eval_batch.edge_index).reshape([-1])

            z = F.sigmoid(self.model.decode(self.model.encode(eval_batch), eval_batch))



            # now rank the scores
























@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


@torch.no_grad()
def compute_mrr(z, edge_index, edge_type, data, model):
    ranks = []
    for i in tqdm(range(edge_type.numel())):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [(data.edge_index, data.edge_type)
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        tail = torch.arange(data.num_nodes)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(tail, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.val_edge_index, data.val_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        head = torch.arange(data.num_nodes)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(head, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

    return (1. / torch.tensor(ranks, dtype=torch.float)).mean()


def add_edge_type_dict_wiki(data):
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


def count_frequencies(data: list) -> None:
    """
    counts the absolute frequency and makes a histogram
    eg. for the frequency of classes in data
    """
    counted_data = {}
    for v in sorted(set(data)):
        counted_data[v] = data.count(v)
        print(f'{v}, {data.count(v)}')

    val, weight = zip(*[(k, v) for k, v in counted_data.items()])
    plt.hist(val, weights=weight, rwidth=0.5)
    plt.title('Relation Frequency, all.txt')
    plt.savefig('./plots/freq_family_classes.pdf', format='pdf')


def add_edge_common(data, edge_list, path=Path.cwd() / 'Wikialumni' / 'augmented.pkl'):
    """
    Author: Pauline
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

            if edge_type_str in ["deathplace", "homeLocation", "hasOccupation", "nationality", "birthplace",
                                 "affiliation",
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
        train_mask = torch.cat([torch.ones(num_train_edges, dtype=torch.bool),
                                torch.zeros(num_val_edges + num_test_edges, dtype=torch.bool)])
        val_mask = torch.cat(
            [torch.zeros(num_train_edges, dtype=torch.bool), torch.ones(num_val_edges, dtype=torch.bool),
             torch.zeros(num_test_edges, dtype=torch.bool)])
        test_mask = torch.cat([torch.zeros(num_train_edges + num_val_edges, dtype=torch.bool),
                               torch.ones(num_test_edges, dtype=torch.bool)])

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        train_data = data.edge_subgraph(rand_perm[train_mask])
        val_data = data.edge_subgraph(rand_perm[val_mask])
        test_data = data.edge_subgraph(rand_perm[test_mask])

        return data, train_data, val_data, test_data


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
            ids = torch.where(self.data.edge_type == self.current_type)[0]
            filtered_edge_index = self.data.edge_label_index[:, ids]
            filtered_edge_type = self.data.edge_type[ids]
            batch.edge_label_index, batch.edge_type, mask = remove_isolated_nodes(filtered_edge_index,
                                                                                  filtered_edge_type,
                                                                                  num_nodes=self.data.num_nodes)
            batch.x = batch.x[mask, :]
            batch.y = batch.y[mask]
            batch.num_nodes = sum(mask)
            self.current_type += 1
            return batch
        raise StopIteration
