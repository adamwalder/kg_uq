import json
import os
import re

import numpy as np
import pandas as pd
import torch as tn
import torch.optim
import torch_geometric.transforms as T
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData

log_sig = torch.nn.LogSigmoid()

from typing import List, Tuple, Dict


def get_recipe_subkgs(data_dir: str = './recipe_data') -> List[Tuple[str, List[Dict[str, str]]]]:
    """Reads in recipe subgraphs from 'recipe_data' directory.

    :param data_dir: (str) path to jsons containing recipe triples.
    :return: List of tuples containing title of recipe, and list of tuples with type and feature.
    """

    tmp_file = 'tmp_list_' + bin(np.random.randint(2 ** 31)) + '.txt'
    os.system('ls ' + data_dir + '/kg_files/*.json > ' + data_dir + '/' + tmp_file)

    with open(data_dir + '/' + tmp_file, 'r') as f:
        json_locs = [x.split('\n')[0] for x in f.readlines()]
        recipes = [x.split('/')[-1] for x in json_locs]
        recipes = [re.sub('_', ' ', x.split('.json')[0]) for x in recipes]

    os.system('rm ' + data_dir + '/' + tmp_file)

    kgs = []
    for x in zip(recipes, json_locs):
        with open(x[1], 'r') as f:
            kgs.append((x[0], json.load(f)))

    return kgs

def get_recipe_kg(data_dir: str = './recipe_data') -> Tuple[pd.DataFrame, np.ndarray]:
    kgs = get_recipe_subkgs(data_dir)
    y = []
    y_idx = []
    for i, x in enumerate(kgs):
        y_idx.extend([i]*len(x[1]))
        y.extend(x[1])

    y = pd.DataFrame(y)
    y_idx = np.asarray(y_idx)

    return y, y_idx


def kg_to_hetero(data_dir: str,
                 embeddings_model,
                 undirected: bool = False) -> Tuple[
    List[HeteroData], Tuple[pd.DataFrame, np.ndarray], Tuple[Dict[str, tn.Tensor], Dict[str, Dict[str, int]]]]:
    """

    :param embeddings_model:
    :param undirected:
    :return:
    """
    kg, kg_idx = get_recipe_kg(data_dir=data_dir)
    nkgs = np.max(kg_idx)+1

    # count all features by node type, we'll use this for mapping embedded node features to HeteroData
    node_map, cnt = dict(), 0
    for node_type in np.union1d(kg['head_type'].unique(), kg['tail_type'].unique()):
        if node_type not in node_map:
            node_map[node_type] = {"__cnt__": 0}

        head_type_idx = kg['head_type'] == node_type

        if np.any(head_type_idx):
            for head in kg['head'][head_type_idx].unique():
                node_map[node_type][head] = node_map[node_type]["__cnt__"]
                node_map[node_type]["__cnt__"] += 1

        tail_type_idx = kg['tail_type'] == node_type
        if np.any(tail_type_idx):
            for tail in kg['tail'][tail_type_idx].unique():
                if tail not in node_map[node_type]:
                    node_map[node_type][tail] = node_map[node_type]["__cnt__"]
                    node_map[node_type]["__cnt__"] += 1

    for k in node_map.keys():
        del node_map[k]['__cnt__']

    # maps the node type to an enumerated feature type for the given node. Allows selection for
    # head_type --> index of feature for given node_type
    head_idx = np.zeros(len(kg), dtype=np.int32)
    tail_idx = np.zeros(len(kg), dtype=np.int32)

    for node_type in node_map.keys():
        for feat in node_map[node_type].keys():
            tmp = np.bitwise_and(kg['head_type'] == node_type, kg['head'] == feat)
            if np.any(tmp):
                head_idx = np.where(tmp, node_map[node_type][feat], head_idx)

            tmp = np.bitwise_and(kg['tail_type'] == node_type, kg['tail'] == feat)
            if np.any(tmp):
                tail_idx = np.where(tmp, node_map[node_type][feat], tail_idx)

    # dictionary map of all features for a gvien node type and the corresponding embeddings of text.
    embeddings_model = SentenceTransformer('all-MiniLM-L6-v2') if embeddings_model is None else embeddings_model
    emb_map = dict()
    for node_type in node_map.keys():
        emb_map[node_type] = embeddings_model.encode([x for x in node_map[node_type].keys()])

    # set training data with mappings to correct embeddings for each node.
    train = []
    for i in range(nkgs):

        data = HeteroData()
        for node_type in node_map.keys():
            data[node_type].x = tn.tensor(emb_map[node_type], dtype=tn.float32)

        kg_i = kg_idx == i
        urels = kg[kg_i]['relation'].unique()

        for rel in urels:
            rel_idx = np.bitwise_and(kg_i, kg['relation'] == rel)
            rel_heads = np.unique(kg[rel_idx]['head_type'])
            rel_tails = np.unique(kg[rel_idx]['tail_type'])

            for rhead in rel_heads:
                rhead_idx = np.bitwise_and(rel_idx, kg['head_type'] == rhead)

                for rtail in rel_tails:
                    rtail_idx = np.bitwise_and(rhead_idx, kg['tail_type'] == rtail)

                    tmp0 = tn.tensor(head_idx[rtail_idx], dtype=tn.int64)
                    tmp1 = tn.tensor(tail_idx[rtail_idx], dtype=tn.int64)

                    if (rhead, rel, rtail) not in data and np.any(rtail_idx):
                        data[rhead, rel, rtail].edge_index = tn.zeros((2, np.count_nonzero(rtail_idx)), dtype=tn.int64)
                        data[rhead, rel, rtail].edge_index[0] = tmp0
                        data[rhead, rel, rtail].edge_index[1] = tmp1
                    else:
                        tmp = tn.row_stack((tmp0, tmp1))
                        data[rhead, rel, rtail].edge_index = tn.column_stack((data[rhead, rel, rtail].edge_index, tmp))

        # remove isolated nodes and add reverse edges
        data = T.RemoveIsolatedNodes()(data)
        if undirected:
            data = T.ToUndirected()(data)

        train.append(data)

    return train, (kg, kg_idx), (node_map, emb_map)

