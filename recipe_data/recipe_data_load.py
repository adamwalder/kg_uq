"""Functions to load in extracted recipe kg triples as DataFrame and HeteroData objects."""
import json
import os
import re
from typing import List, Tuple, Dict, Union

import numpy as np
import pandas as pd
import torch as tn
import torch.optim
import torch_geometric.transforms as T
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData

log_sig = torch.nn.LogSigmoid()


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

def get_recipe_kg(data_dir: str = './recipe_data') -> pd.DataFrame:
    """Returns DataFrame containing all triples or recipe subgraphs.

    Returns pd.DataFrame with head, head_type, relation, tail, tail_type, and an indicator `kg_idx` denoting
    the id of subgraph loaded in.

    :param data_dir: Path to 'recipe_data' folder.
    :return: Tuple of DataFrame

    """
    kgs = get_recipe_subkgs(data_dir)
    y = []
    y_idx = []
    y_lab = []

    for i, x in enumerate(kgs):
        y_idx.extend([i]*len(x[1]))
        y_lab.extend([x[0]]*len(x[1]))
        y.extend(x[1])

    y = pd.DataFrame(y)
    y = y.assign(kg_idx=np.asarray(y_idx))
    y = y.assign(kg_name=np.asarray(y_lab))

    return y

def recipe_kg_to_hetero(data_dir: str,
                        sentence_transformer_model: str = 'all-MiniLM-L6-v2',
                        undirected: bool = False) -> Dict[str, Union[List[HeteroData], pd.DataFrame,
                                                              Tuple[Dict[str, Dict[str, int]], Dict[str, tn.Tensor]]]]:
    """Returns a dictionary containing sequence of HeteroData with embedded features for each subgraph in kg.

    Sentences are treated as features, which are embedded according to SentenceTransformer(sentence_transformer_model).

    The dictionary returns contains:
        - 'train': List of HeteroData() objects for each subgraph kg with embedded features.
        - 'kg': DataFrame containing head, head_type, relation, tail, tail_type, and id of subgraphs.
        - 'embedding_maps': Tuple of dictionaries :
            (0): Map from node_type to features as sentences and their respective enumerations over the node type
            (1): Map from node_type to embedded sentences corresponding to the enumeration of features above.

    :param data_dir: Path to 'recipe_data' folder in repo.
    :param sentence_transformer_model: Model used for SentenceTransformer embeddings. Default 'all-MiniLM-L6-v2'.
    :param undirected: If True, each subgraph has reversed edges.
    :return: See above

    """
    kg = get_recipe_kg(data_dir=data_dir)

    # clean data of any empty entries
    bad_rows = np.bitwise_or(kg['tail'] == '', kg['tail_type'] == '')
    bad_rows = np.bitwise_or(bad_rows, kg['head'] == '')
    bad_rows = np.bitwise_or(bad_rows, kg['head_type'] == '')
    bad_rows = np.bitwise_or(bad_rows, kg['relation'] == '')
    kg = kg[~bad_rows].reset_index(drop=True)

    # total number of subgraphs
    kg_ids = kg['kg_idx'].unique()

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
    embeddings_model = SentenceTransformer(sentence_transformer_model)
    emb_map = dict()
    for node_type in node_map.keys():
        emb_map[node_type] = embeddings_model.encode([x for x in node_map[node_type].keys()])

    # set training data with mappings to correct embeddings for each node.
    train = []
    for i in kg_ids:

        data = HeteroData()
        for node_type in node_map.keys():
            data[node_type].x = tn.tensor(emb_map[node_type], dtype=tn.float32)

        kg_i = kg['kg_idx'] == i
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

    rv = {'train': train, 'kg': kg, 'embedding_maps': (node_map, emb_map)}

    return rv

