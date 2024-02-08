import os
from datasets import load_dataset
import string
import re
import json
from typing import Dict, Tuple, List

def load_hotpot_qa(path_to_data: str, ndocs: int = 2) -> Dict[str, Dict[str, List[str]]]:
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
