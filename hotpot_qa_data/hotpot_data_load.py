import os

import pandas as pd
from datasets import load_dataset
import string
import re
import json
from typing import Dict, List
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

        for i in range(ndocs):

            x = dataset['train'][i]
            topics = [re.sub(pattern, '', y).lower() for y in x['context']['title']]
            hotpot_files[x['id']] = {'topics': [], 'file_paths': []}
            # print(topics, [re.sub(' ', '_', topics[j]) for j in range(len(topics))])

            for j in range(len(topics)):
                file_name = '/'.join([path_to_data, 'txt_files', re.sub(' ', '_', topics[j]) + '.txt'])
                hotpot_files[x['id']]['topics'].append(topics[j])
                hotpot_files[x['id']]['file_paths'].append(file_name)

                if not os.path.exists(file_name):
                    with open(file_name, 'w') as f:
                        f.writelines(x['context']['sentences'][j])

        with open(hotpot_json, 'w') as f:
            json.dump(hotpot_files, f)
    else:
        with open(hotpot_json, 'r') as f:
            hotpot_files = json.load(f)

    return hotpot_files

def load_hotpot_kgs(path_to_data: str) -> pd.DataFrame:
    """Read the hotpot_qa json kg triples from ./kg_files to DataFrame.

    `doc_id` is the id for the group (question group id), `sub_idx` is the id of the subgraph extracted for
        context entry `j` for a particular question.


    :param path_to_data: Path to './hotpot_qa_data/txt_files'
    :return: DataFrame with extracted subgraphs.
    """
    hotpot_json = '/'.join([path_to_data, 'hotpot_qa_data.json'])
    if os.path.exists(hotpot_json):
        with open(hotpot_json, 'r') as f:
            hotpot_files = json.load(f)

        kgs = []
        kg_idx = []
        qa_idx = []
        titles = []
        for i, hid in enumerate(hotpot_files.keys()):

            for j, file in enumerate(hotpot_files[hid]['file_paths']):
                file_name = re.sub('.txt', '.json', re.sub('txt_files', 'kg_files', file))

                if os.path.exists(file_name):
                    with open(file_name, 'r') as f:
                        tmp = json.load(f)
                        kg_idx.extend([j]*len(tmp))
                        qa_idx.extend([i]*len(tmp))
                        kgs.extend(tmp)
                        titles.extend([hotpot_files[hid]['topics'][j]]*len(tmp))

        y = pd.DataFrame(kgs)
        y = y.assign(titles=titles)
        y = y.assign(doc_id=qa_idx)
        y = y.assign(sub_idx=kg_idx)

        return y

    else:
        raise Exception('You need to scrape the txt_files and write to kg_files.')
