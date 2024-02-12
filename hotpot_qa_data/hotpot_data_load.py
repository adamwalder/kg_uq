import os

import pandas as pd
from datasets import load_dataset
import string
import re
import json
from typing import Dict, List, Tuple, Optional
import numpy as np

def write_hotpot_to_txt(path_to_data: str, ndocs: int = 10) -> Dict[str, Dict[str, List[str]]]:
    """Write out hotpot qa entries to a txt file.

    :param path_to_data: Path to './hotpot_qa_data/txt_files'
    :param ndocs: Number of question/amswer pairs to write to txt.
    :return: Dict mapping question id to the set of context names and file paths for each context txt file.

    """
    dataset = load_dataset("hotpot_qa", "distractor")
    hotpot_json = '/'.join([path_to_data, 'hotpot_qa_data.json'])
    if not os.path.exists(hotpot_json):

        pattern = r"[{}]".format(string.punctuation)
        hotpot_files = dict()
        query_answer = []

        for i in range(ndocs):

            x = dataset['train'][i]
            topics = [re.sub(pattern, '', y).lower() for y in x['context']['title']]
            hotpot_files[x['id']] = {'topics': [], 'file_paths': []}
            query_answer.append({'query': x['query'], 'answer': x['answer']})

            for j in range(len(topics)):
                file_name = '/'.join([path_to_data, 'txt_files', re.sub(' ', '_', topics[j]) + '.txt'])
                hotpot_files[x['id']]['topics'].append(topics[j])
                hotpot_files[x['id']]['file_paths'].append(file_name)

                if not os.path.exists(file_name):
                    with open(file_name, 'w') as f:
                        f.writelines(x['context']['sentences'][j])

        with open(hotpot_json, 'w') as f:
            json.dump(hotpot_files, f)

        with open('/'.join([path_to_data, 'query_answer.json']), 'w') as f:
            json.dump(query_answer, f)

    else:
        with open(hotpot_json, 'r') as f:
            hotpot_files = json.load(f)

    return hotpot_files

def load_hotpot_kgs(path_to_data: str, query_answer: bool = False) -> \
        Tuple[pd.DataFrame,  Optional[List[Dict[str, str]]]]:
    """Read the hotpot_qa json kg triples from ./kg_files to DataFrame.

    `doc_id` is the id for the group (question group id), `sub_idx` is the id of the subgraph extracted for
        context entry `j` for a particular question.


    :param path_to_data: Path to './hotpot_qa_data/txt_files'
    :param query_answer: If True, returns list of dictionary of containing query/answer pairs for subset of data.
    :return: DataFrame with extracted subgraphs.
    """
    hotpot_json = '/'.join([path_to_data, 'hotpot_qa_data.json'])
    if os.path.exists(hotpot_json):
        with open(hotpot_json, 'r') as f:
            hotpot_files = json.load(f)

        kgs = []
        kg_idx = []
        qa_idx = []
        file_paths = []
        for i, hid in enumerate(hotpot_files.keys()):

            for j, file in enumerate(hotpot_files[hid]['file_paths']):
                file_name = re.sub('.txt', '.json', re.sub('txt_files', 'kg_files', file))

                if os.path.exists(file_name):
                    with open(file_name, 'r') as f:
                        tmp = json.load(f)
                        kg_idx.extend([j]*len(tmp))
                        qa_idx.extend([i]*len(tmp))
                        kgs.extend(tmp)
                        file_paths.extend([hotpot_files[hid]['file_paths'][j]]*len(tmp))

        y = pd.DataFrame(kgs)
        y = y.assign(file_path=file_paths)
        y = y.assign(doc_id=qa_idx)
        y = y.assign(sub_idx=kg_idx)

        if query_answer:
            with open('/'.join([path_to_data, 'query_answer.json']), 'r') as f:
                qa = json.load(f)
            return y, qa
        else:
            return y, None

    else:
        raise Exception('You need to scrape the txt_files and write to kg_files.')
