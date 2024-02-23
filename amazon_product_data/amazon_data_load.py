import json
import os
import re
from typing import List, Tuple, Dict, Union, Optional
import pickle

import numpy as np
from numpy.random import RandomState
import pandas as pd
import torch as tn
import torch_geometric.transforms as T
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData


def get_top_product_types(data: Optional[pd.DataFrame] = None,
                          data_path: Optional[str] = None,
                          n_types: int = 10,
                          n: int = 10,
                          rng: Optional[RandomState] = None) -> pd.DataFrame:
    """Select top `n_types` product types from amazon_product dataset.

    :param data_path: Path to `./amazon_product_data`. Will load in DataFrame is data is None.
    :param data: DataFrame containing amazon product information.
    :param n_types: Number of product types to extract (max is 50).
    :param n: Number of samples to draw from top n_types. (train + test).
    :param rng: Optional RandomState for subsampling from product types.
    :return: DataFrame with top 'n_type' triples. 'content' contains text to have triples extracted from.

    """

    if data is None:
        if data_path is None:
            data_path = '/'.join([os.getcwd(), 'amazon_product_data'])

        with open('/'.join([data_path, 'amazon_product.pkl']), 'rb') as f:
            data = pickle.load(f)

    if 1 >= n_types or n_types > 50:
        raise Exception('n_types must be between [2,50].')

    rng = np.random.RandomState() if rng is None else rng

    val_counts = data['PRODUCT_TYPE_ID'].value_counts()
    type_ids = val_counts.keys().values[:n_types]

    # make sure the max number of training samples is possible.
    max_train = val_counts.values[:n_types].min()
    n = max_train if n > max_train else n

    df = data[np.any(data['PRODUCT_TYPE_ID'].values == type_ids[:, None], axis=0)]
    df = df.reset_index(drop=True)

    subset = np.zeros(len(df), dtype=bool)

    for i, x in enumerate(df['PRODUCT_TYPE_ID'].unique()):
        subset[rng.choice(df[df['PRODUCT_TYPE_ID'] == x].index.values, n, replace=False)] = True

    df = df[subset].reset_index()
    df = df.assign(content=df['TITLE'] + df['DESCRIPTION'] + df['BULLET_POINTS'])

    return df

def get_amazon_kg(data_path: str = './amazon_product_data') -> pd.DataFrame:
    """Create a DataFrame from triples extracted in `./amazon_product/kg_files`

    :param data_path: Path to './amazon_product_data/kg_files'
    :return: DataFrame of amazon product triples. product_id is the unique product key and group_id is
        the product_type_id.

    """
    path_int = np.random.randint(2 ** 31)
    path_to_data = f'{data_path}/kg_files'

    # get all file_paths in temp .txt file, read into file_paths then delete tmp_*.txt
    tmp_file = f'{data_path}/tmp_{path_int}.txt'
    os.system(f'ls {path_to_data}/*.json > {tmp_file}')

    with open(f'{data_path}/tmp_{path_int}.txt', 'r') as f:
        file_paths = [x.split('\n')[0] for x in f.readlines()]

    os.system(f'rm {tmp_file}')

    # create pandas DataFrame for each document
    product_id = []
    group_idx = []
    kg = []

    for cnt, file_str in enumerate(file_paths):
        u0, u1 = re.findall(r'\d+', file_str.split('/')[-1])

        with open(file_str, 'r') as f:
            tmp = json.load(f)

        product_id.extend([int(u1)] * len(tmp))
        group_idx.extend([int(u0)] * len(tmp))
        kg.extend(tmp)

    kg = pd.DataFrame(kg)
    kg = kg.assign(product_id=product_id)
    kg = kg.assign(group_id=group_idx)

    return kg

def amazon_kg_to_hetero(data_path: Optional[str] = None,
                        kg: Optional[pd.DataFrame] = None,
                        sentence_transformer_model: str = 'all-MiniLM-L6-v2',
                        undirected: bool = False,
                        return_kg: bool = False) -> \
        Tuple[Dict[str, List[HeteroData]], Tuple[Dict[str, Dict[str, int]], Dict[str, tn.Tensor]], Optional[pd.DataFrame]]:
    """Returns a dictionary containing sequence of HeteroData with embedded features for each subgraph in kg.

    Sentences are treated as features, which are embedded according to SentenceTransformer(sentence_transformer_model).

    The dictionary returns contains:
        - 'train': List of HeteroData() objects for each subgraph kg with embedded features.
        - 'kg': DataFrame containing head, head_type, relation, tail, tail_type, and id of subgraphs.
        - 'embedding_maps': Tuple of dictionaries :
            (0): Map from node_type to features as sentences and their respective enumerations over the node type
            (1): Map from node_type to embedded sentences corresponding to the enumeration of features above.

    :param data_path: Path to 'amazon_product_data/kg_files' folder in repo.
    :param kg: Pandas DataFrame, if passed this treated at triple set to build HeteroData objects from.
    :param sentence_transformer_model: Model used for SentenceTransformer embeddings. Default 'all-MiniLM-L6-v2'.
    :param undirected: If True, each subgraph has reversed edges.
    :param return_kg: If True the knowledge graphs are returned as a DataFrame
    :return: See above

    """
    if data_path and kg is None:
        kg = get_amazon_kg(data_path=data_path)
    elif data_path is None and kg is None:
        raise Exception('Must give path to ./amazon_product_data or DataFrame with triples...')

    # clean data of any empty entries
    bad_rows = np.bitwise_or(kg['tail'] == '', kg['tail_type'] == '')
    bad_rows = np.bitwise_or(bad_rows, kg['head'] == '')
    bad_rows = np.bitwise_or(bad_rows, kg['head_type'] == '')
    bad_rows = np.bitwise_or(bad_rows, kg['relation'] == '')
    kg = kg[~bad_rows].reset_index(drop=True)

    # total number of subgraphs
    kg_ids = kg['product_id'].unique()

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
    train = dict()
    for i in kg_ids:

        data = HeteroData()
        for node_type in node_map.keys():
            data[node_type].x = tn.tensor(emb_map[node_type], dtype=tn.float32)

        kg_i = kg['product_id'] == i
        kg_group = kg[kg['product_id'] == i]['group_id'].values[0]

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
        #data = T.RemoveIsolatedNodes()(data)

        if undirected:
            data = T.ToUndirected()(data)

        if kg_group not in train:
            train[kg_group] = [data]

        else:
            train[kg_group].append(data)

    if return_kg:
        return train, (node_map, emb_map), kg
    else:
        return train, (node_map, emb_map), None

