"""This script extracts triples from hotpot_qa and reads them into hotpot_qa_data/kg_files"""
import os
import sys
import nest_asyncio
nest_asyncio.apply()

open_ai_key = '...'
os.environ['OPENAI_API_KEY'] = open_ai_key
sys.path = ['/Users/walder2/kg_uq/'] + sys.path
path_to_data = '/Users/walder2/kg_uq/hotpot_qa_data'

from datasets import load_dataset
dataset = load_dataset("hotpot_qa", "distractor")

from kg_extraction import *

entity_types = {
    "person": 'https://schema.org/Person',
    'place': 'https://schema.org/Place',
    'thing': 'https://schema.org/Thing',
    'creativeWork': 'https://schema.org/CreativeWork',
    'event': 'https://schema.org/Event',
    'product': 'https://schema.org/Product'

}

relation_types = {
    "hasCharacteristic": "https://schema.org/additionalProperty",
    "hasColor": "https://schema.org/color",
    "hasMeasurement": "https://schema.org/hasMeasurement",
    "person": 'https://schema.org/Person',
    'place': 'https://schema.org/Place',
    'thing': 'https://schema.org/Thing',
    'creativeWork': 'https://schema.org/CreativeWork',
    'event': 'https://schema.org/Event',
    'product': 'https://schema.org/Product'

}

from hotpot_qa_data.hotpot_data_load import write_hotpot_to_txt

if __name__ == "__main__":

    hotpot_files = write_hotpot_to_txt(path_to_data=path_to_data, ndocs=20)

    cnt = 0
    tot = sum([len(v['topics']) for v in hotpot_files.values()])
    print('Starting extraction...')
    for hid in hotpot_files.keys():
        n = len(hotpot_files[hid]['topics'])

        extract_kg(entity_types=entity_types,
                   relation_types=relation_types,
                   data_dir=path_to_data,
                   txt_files=hotpot_files[hid]['file_paths'], verbose=False)
        cnt += n
        print(f'Completed {cnt} out of {tot} kg extractions.')
