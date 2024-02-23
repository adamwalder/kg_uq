"""Functions for scraping recipes as kg triples from https://www.allrecipes.com/."""

import os
open_ai_key = os.environ.get('OPENAI_API_KEY', None)
import asyncio
import re
import json
from typing import Dict, List, Optional

from llama_index import SimpleDirectoryReader

from kg_extraction.extraction_prompts import TripleExtractor

dash_str = '-'.join([' ' for _ in range(30)])


def extract_kg(data_dir: str,
               entity_types: Dict[str, str],
               relation_types: Dict[str, str],
               txt_files: Optional[List[str]] = None,
               verbose: bool = False,
               model: Optional[str] = "gpt-3.5-turbo",
               system_prompt: Optional[str] = None,
               user_prompt: Optional[str] = None,
               batch_size: int = 5) -> None:
    """Extract and save kg triples into .json file in './data_dir/kg_files/' directory.

    Note user must provide entity_types and relation_types E.g.)
    Using schema.org
        entity_types = {
            "recipe": 'https://schema.org/Recipe',
            "ingredient": "https://schema.org/recipeIngredient",
            "measurement": "https://schema.org/QuantitativeValue",
        }

        relation_types = {
            "hasCharacteristic": "https://schema.org/additionalProperty",
            "hasColor": "https://schema.org/color",
            "hasMeasurement": "https://schema.org/hasMeasurement",
            "cookTime": "https://schema.org/cookTime",
            "recipeInstruction": "https://schema.org/recipeInstructions"

         }

    If you want to change the system_prompt or user_prompt, please first see extraction_prompts.py. The user_prompt
    must specify to return the KG as detailed!

    :param data_dir: Default directory for storing kg triples --> ./kg_files.
    :param entity_types: Dictionary detailing entities and their format
    :param relation_types: Dictionary detailing entities and their format
    :param txt_files: List of strings to consider for triple extraction. Should be cleaned text.
    :param verbose: If true, the triples will be printed out as they are extracted.
    :param model: model used for extracting relationships. Default to gpt-3.5-turbo
    :param system_prompt: Prompt for extraction task. See extraction_prompts.py for example.
    :param user_prompt: Prompt for how triples should be extracted. See extraction_prompts.py for example
    :param batch_size: Batch size for asynchronous calls to LLM for triple extraction.
    :return: None
    """
    batch_size = int(batch_size)
    batch_size = batch_size if batch_size > 0 else 1

    if not txt_files:
        txt_docs = SimpleDirectoryReader('/'.join([data_dir, 'txt_files'])).load_data()
    else:
        txt_docs = SimpleDirectoryReader(input_files=txt_files).load_data()

    triple_extractor = TripleExtractor(entity_types=entity_types, relation_types=relation_types,
                                       system_prompt=system_prompt, user_prompt=user_prompt, model=model)
    for i in range(0, len(txt_docs), batch_size):
        content = txt_docs[i:i+batch_size]
        content_name = [x.metadata['file_name'].split('.txt')[0] for x in content]
        file_name = ['/'.join([data_dir, 'kg_files', x + '.json']) for x in content_name]
        results = triple_extractor.extract_triples(text_prompts=content, file_names=file_name, kg_names=content_name)

        for res in results:
            if res[0]:
                with open(res[1], 'w') as f:
                    json.dump(res[0], f)

                tmp = '/'.join(['./kg_files', re.sub(' ', '_', res[2]) + '.json'])
                if verbose:
                    print('\n'.join([dash_str + f'\n{repr(tmp)}', dash_str, f'{res[0]}\n']))
                else:
                    print('\n'.join([dash_str + f'\n{repr(tmp)}', dash_str, f'Extracted {res[2]}.\n']))

            else:
                tmp = '/'.join(['./kg_files', re.sub(' ', '_', res[2]) + '.json'])
                print('\n'.join([dash_str + f'\n{repr(tmp)}', dash_str, f'Already extracted {res[2]}.\n']))

