import json
import os
import re

import requests
from bs4 import *
from openai import OpenAI as clientOpenAI

from kg_extraction.recipe_prompts import get_recipe_prompts

open_ai_key = os.environ.get('OPENAI_API_KEY', None)

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    VectorStoreIndex,
)

from llama_index.llms import OpenAI as OpenAI
from llama_index.prompts import PromptTemplate

from typing import Dict, List, Optional


def scrape_recipe_websites(websites: List[str], recipes: List[str], data_dir: str) -> None:
    """Scrape websites for recipes. Stores all info html info in txt files.

    :param websites: List of websites.
    :param recipes: List of recipe titles corresponding to the websites.
    :param data_dir: Location for 'recipe_data' path.
    :return: None
    """

    for url, rec in zip(websites, recipes):
        response = requests.get(url)
        if response.status_code == 200:
            file_name = data_dir + "/html_files/" + re.sub(' ', '_', rec) + '.txt'
            if not os.path.exists(file_name):
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)

                with open(data_dir + "/html_files/" + re.sub(' ', '_', rec) + '.txt', 'w') as text_file:
                    text_file.write(text)
        else:
            print('Could not load: %s' + rec)


def extract_ingredients_directions(data_dir: str,
                                   verbose: bool = False,
                                   service_context: Optional[ServiceContext] = None,
                                   directions_template: Optional[PromptTemplate] = None,
                                   ingredients_template: Optional[PromptTemplate] = None) -> None:
    """Function for extracting ingredient list and directions for recipe to a txt file. This is used next to extract
    knowledge graph triples.

    :param data_dir: Path to 'recipe_data'.
    :param verbose: If true, the extracted ingredients and directions are printed out for each recipe.
    :param service_context: Optional ServiceContext object for storing and setting llm.
    :param directions_template: PromptTemplate for extracting directions. If None, handled by function.
    :param ingredients_template: PromptTemplate for extracting ingredients. If None, handled by function.
    :return: None
    """
    if service_context is None:
        llm = OpenAI(temperature=0)
        service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

    if directions_template is None:
        directions_template_str = (
            "Context information is"
            " below.\n---------------------\n{context_str}\n---------------------\nUsing"
            " both the context information, answer"
            " the question: {query_str}\n"
            "Do not create a numbered list.\n"
            "Be as concise as possible.\n"

        )
        directions_template = PromptTemplate(directions_template_str)

    if ingredients_template is None:
        ingredients_template_str = (
            "Context information is"
            " below.\n---------------------\n{context_str}\n---------------------\nUsing"
            " both the context information, answer"
            " the question: {query_str}\n"
            "Return the answer as a list seperated by commas\n"
            "Include measurements\n"
            "Swap the words 'or' and 'and', with a commas.\n"
            "Do not repeat any ingredients in the response.\n"
        )
        ingredients_template = PromptTemplate(ingredients_template_str)

    html_docs = SimpleDirectoryReader(data_dir + '/html_files').load_data()
    vector_index = VectorStoreIndex.from_documents(html_docs, service_context=service_context)
    directions_engine = vector_index.as_query_engine(text_qa_template=directions_template)
    ingredients_engine = vector_index.as_query_engine(text_qa_template=ingredients_template)

    print('Extracting ingredients and directions from html data .... \n')

    for i in range(len(html_docs)):
        tmp = html_docs[i].metadata['file_name']
        rec = re.sub('_', ' ', tmp.split('.txt')[0])

        res = ingredients_engine.query('Give me a list of the ingredients for %s' % rec)
        ing_res = 'The ingredients for making ' + rec + ' are ' + res.response
        dir_res = directions_engine.query('Give me directions for making %s' % rec)

        if verbose:
            print(f'\n---------------------\n{ing_res}\nDirections: {dir_res.respnse}---------------------\n')

        with open(data_dir + "/txt_files/" + tmp, 'w') as text_file:
            text_file.write(ing_res + '\n' + dir_res.response)


def kg_triple_extractor(text: str,
                        entity_types: Dict[str, str],
                        relation_types: Dict[str, str],
                        model: Optional[str] = "gpt-3.5-turbo",
                        system_prompt: Optional[str] = None,
                        user_prompt: Optional[str] = None):
    """Get triples from recipe ingredients and directions. This has a call to OpenAI.

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

    If you want to change the system_prompt or user_prompt, please first see recipe_prompts.py. The user_prompt
    must specify to return the KG as detailed!

    :param text: Text to have entity, rel, entity extraction from
    :param entity_types: Dictionary detailing entities and their format
    :param relation_types: Dictionary detailing entities and their format
    :param model: model used for extracting relationships. Default to gpt-3.5-turbo
    :param system_prompt: Prompt for extraction task. See recipe_prompts.py for example.
    :param user_prompt: Prompt for how triples should be extracted. See recipe_prompts.py for example
    :return:
    """
    client = clientOpenAI(api_key=open_ai_key)

    if system_prompt is None or user_prompt is None:
        system_prompt, user_prompt = get_recipe_prompts()
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt.format(
              entity_types=entity_types,
              relation_types=relation_types,
              specification=text
            )
        }
        ]
    )

    return completion.choices[0].message.content


def extract_recipe_kg(entity_types: Dict[str, str],
                      relation_types: Dict[str, str],
                      data_dir: str,
                      verbose: bool = False,
                      model: Optional[str] = "gpt-3.5-turbo",
                      system_prompt: Optional[str] = None,
                      user_prompt: Optional[str] = None) -> None:
    """Extract and save recipe knowledge graphs in json files under the './data_dir/kg_files/' directory.

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

    If you want to change the system_prompt or user_prompt, please first see recipe_prompts.py. The user_prompt
    must specify to return the KG as detailed!

    :param entity_types: Dictionary detailing entities and their format
    :param relation_types: Dictionary detailing entities and their format
    :param data_dir: Default directory for storing recipe_kg.json files.
    :param verbose: If true, the triples will be printed out as they are extracted.
    :param model: model used for extracting relationships. Default to gpt-3.5-turbo
    :param system_prompt: Prompt for extraction task. See recipe_prompts.py for example.
    :param user_prompt: Prompt for how triples should be extracted. See recipe_prompts.py for example
    :return: None
    """

    txt_docs = SimpleDirectoryReader(data_dir + '/txt_files').load_data()
    for content in txt_docs:
        try:
            recipe_name = content.metadata['file_name'].split('.txt')[0]
            extracted_relations = kg_triple_extractor(content.text, entity_types, relation_types, model, system_prompt,
                                                      user_prompt)
            extracted_relations = json.loads(extracted_relations)
            with open(data_dir + '/kg_files/' + recipe_name + '.json', 'w') as f:
                json.dump(extracted_relations, f)

            if verbose:
                print(f'\n-----------\n{recipe_name}\n{extracted_relations}\n')
        except Exception as e:
            print(e)