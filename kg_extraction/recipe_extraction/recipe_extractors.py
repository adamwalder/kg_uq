"""Functions for scraping recipes as kg triples from https://www.allrecipes.com/."""

import os
open_ai_key = os.environ.get('OPENAI_API_KEY', None)
import asyncio
import re
import json
from typing import Dict, List, Optional, Tuple
import requests
from bs4 import *

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    VectorStoreIndex,
)

from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.llms import OpenAI as OpenAI
from llama_index.prompts import PromptTemplate
from kg_extraction.extraction_prompts import TripleExtractor

dash_str = '-'.join([' ' for _ in range(30)])

class AsyncRecipeQueryEngine:
    def __init__(self, query_engines: Dict[str, BaseQueryEngine], queries: Dict[str, str]):
        self.query_engines = query_engines
        self.queries = queries

    async def run_query(self, key: str, query: str):
        return await self.query_engines[key].aquery(query)

    def run_queries(self, recipe: str) -> str:
        tasks = [self.run_query(k, self.queries[k] % recipe) for k in self.queries.keys()]
        res = asyncio.run(asyncio.gather(*tasks))

        rec_str = f'The ingredients for making {recipe} are {res[0].response}.\n'
        rec_str += res[1].response
        rec_str += f'The following is preparation information for {recipe}, {res[2].response}.\n'
        rec_str += f'The nutritional content for {recipe} consists of {res[3].response}.\n'

        return rec_str

def extract_recipe_content(data_dir: str,
                           verbose: bool = False,
                           html_files: Optional[List[str]] = None,
                           return_files: bool = False,
                           service_context: Optional[ServiceContext] = None,
                           directions_template: Optional[PromptTemplate] = None,
                           ingredients_template: Optional[PromptTemplate] = None) -> Optional[List[str]]:
    """Function for extracting ingredient list and directions for recipe to a txt file. This is used next to extract
    knowledge graph triples.

    :param data_dir: Path to 'recipe_data'.
    :param verbose: If true, the extracted ingredients and directions are printed out for each recipe.
    :param html_files: A list of the html files scraped to read in. If none defaults to data_dir/recipe_data/html_files.
    :param return_files: If true, a list of the scraped websites paths are returned.
    :param service_context: Optional ServiceContext object for storing and setting llm.
    :param directions_template: PromptTemplate for extracting directions. If None, handled by function.
    :param ingredients_template: PromptTemplate for extracting ingredients. If None, handled by function.
    :return: None or List[str] containing paths to newly cleaned html files.

    """
    if service_context is None:
        llm = OpenAI(temperature=0)
        service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

    if directions_template is None:
        directions_template_str = (
            "Context information is"
            " below.\n---------------------\n{context_str}\n---------------------\nUsing"
            " both the context information, answer"
            " the question: {query_str}."
            "Do not create a numbered list."
            "Be as concise as possible."

        )
        directions_template = PromptTemplate(directions_template_str)

    if ingredients_template is None:
        ingredients_template_str = (
            "Context information is"
            " below.\n---------------------\n{context_str}\n---------------------\nUsing"
            " both the context information, answer"
            " the question: {query_str}."
            "Return the answer as a list separated by commas."
            "Include measurements."
            "Swap the words 'or' and 'and', with a commas."
            "Do not repeat any ingredients in the response."
        )
        ingredients_template = PromptTemplate(ingredients_template_str)

    nutrition_template_str = (
        "Context information is"
        " below.\n---------------------\n{context_str}\n---------------------\nUsing"
        " both the context information, answer"
        " the question: {query_str}."
        "Only return 'calories', 'fat', 'carbs', and 'protein'."
        "Do not create a numbered list."
        "Be as concise as possible."
        "Give result as a list separated by commas."

    )
    nutrition_template = PromptTemplate(nutrition_template_str)

    prep_template_str = (
        "Context information is"
        " below.\n---------------------\n{context_str}\n---------------------\nUsing"
        " both the context information, answer"
        " the question: {query_str}."
        "Only return 'prep time', 'cook time', 'total time', 'additional time', 'servings' and 'yield'."
        "Give result as a list separated by commas."
        "Do not create a numbered list."
        "Be as concise as possible."

    )
    prep_template = PromptTemplate(prep_template_str)

    if html_files is None:
        html_docs = SimpleDirectoryReader('/'.join([data_dir, 'html_files'])).load_data()
    else:
        html_docs = SimpleDirectoryReader(input_files=html_files).load_data()

    vector_index = VectorStoreIndex.from_documents(html_docs, service_context=service_context)
    query_engines = {
            'ingredients': vector_index.as_query_engine(text_qa_template=ingredients_template),
            'directions': vector_index.as_query_engine(text_qa_template=directions_template),
            'preparation': vector_index.as_query_engine(text_qa_template=prep_template),
            'nutrition': vector_index.as_query_engine(text_qa_template=nutrition_template)
    }
    queries = {
        'ingredients': 'Give me a list of the ingredients for %s.',
        'directions': 'Give me directions for making %s.',
        'preparation': 'Give me the preparation information for %s.',
        'nutrition': 'Give me the nutritional information for %s.'
    }

    async_query_engine = AsyncRecipeQueryEngine(query_engines=query_engines, queries=queries)

    print('Extracting content from html docs...')
    txt_files = []
    for i in range(len(html_docs)):
        tmp = html_docs[i].metadata['file_name']
        rec = re.sub('_', ' ', tmp.split('.txt')[0])
        file_name = '/'.join([data_dir, 'txt_files', tmp])

        if not os.path.exists(file_name):
            rec_str = async_query_engine.run_queries(rec)
            if verbose:
                tmp = '/'.join(['./recipe_data/txt_files', re.sub(' ', '_', rec) + '.txt'])
                print('\n'.join([dash_str + f'\n{repr(tmp)}', dash_str, f'{rec_str}\n']))

            with open(file_name, 'w') as f:
                f.write(rec_str)
        else:
            if verbose:
                tmp = '/'.join(['./recipe_data/txt_files', re.sub(' ', '_', rec) + '.txt'])
                print('\n'.join([dash_str + f'\n{repr(tmp)}', dash_str, f'Already extracted {rec}.\n']))
        txt_files.append(file_name)

    return txt_files if return_files else None


def extract_recipe_kg(data_dir: str,
                      entity_types: Dict[str, str],
                      relation_types: Dict[str, str],
                      txt_files: Optional[List[str]] = None,
                      verbose: bool = False,
                      model: Optional[str] = "gpt-3.5-turbo",
                      system_prompt: Optional[str] = None,
                      user_prompt: Optional[str] = None,
                      batch_size: int = 5) -> None:
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

    If you want to change the system_prompt or user_prompt, please first see extraction_prompts.py. The user_prompt
    must specify to return the KG as detailed!

    :param data_dir: Default directory for storing recipe_kg.json files.
    :param entity_types: Dictionary detailing entities and their format
    :param relation_types: Dictionary detailing entities and their format
    :param txt_files: List of strings to consider for triple extraction. Should be cleaned text.
    :param verbose: If true, the triples will be printed out as they are extracted.
    :param model: model used for extracting relationships. Default to gpt-3.5-turbo
    :param system_prompt: Prompt for extraction task. See extraction_prompts.py for example.
    :param user_prompt: Prompt for how triples should be extracted. See extraction_prompts.py for example.
    :param batch_size: Batch size for asynchronous call to LLM server for triple extraction.
    :return: None
    """
    batch_size = int(batch_size)
    batch_size = batch_size if batch_size > 0 else 1

    if not txt_files:
        txt_docs = SimpleDirectoryReader('/'.join([data_dir, 'txt_files'])).load_data()
    else:
        txt_docs = SimpleDirectoryReader(input_files=txt_files).load_data()

    if system_prompt is None:
        system_prompt = """
        You are an expert agent specialized in analyzing recipes and ingredients.
        Your task is to identify the entities and relations requested with the user prompt, from a set of recipe ingredients 
        and directions.
        You should capture relationships of the ingredients when possible.  
        You must generate the output in a JSON containing a list with JOSN objects having the following keys: "head", 
        "head_type", "relation", "tail", and "tail_type".
        The "head" key must contain the text of the extracted entity with one of the types from the provided list in the 
        user prompt, the "head_type"
        key must contain the type of the extracted head entity which must be one of the types from the provided user list,
        the "relation" key must contain the type of relation between the "head" and the "tail", the "tail" key must 
        represent the text of an
        extracted entity which is the tail of the relation, and the "tail_type" key must contain the type of the tail 
        entity. Attempt to extract as many entities and relations as you can.
        """
    triple_extractor = TripleExtractor(entity_types=entity_types, relation_types=relation_types,
                                       system_prompt=system_prompt, user_prompt=user_prompt, model=model)
    for i in range(0, len(txt_docs), batch_size):
        content = txt_docs[i:i+batch_size]
        recipe_name = [x.metadata['file_name'].split('.txt')[0] for x in content]
        file_name = ['/'.join([data_dir, 'kg_files', x + '.json']) for x in recipe_name]
        results = triple_extractor.extract_triples(text_prompts=content, file_names=file_name, kg_names=recipe_name)

        for res in results:
            if res[0]:
                with open(res[1], 'w') as f:
                    json.dump(res[0], f)
                if verbose:
                    tmp = '/'.join(['./recipe_data/kg_files', re.sub(' ', '_', res[2]) + '.json'])
                    print('\n'.join([dash_str + f'\n{repr(tmp)}', dash_str, f'{res[0]}\n']))
            else:
                tmp = '/'.join(['./recipe_data/kg_files', re.sub(' ', '_', res[2]) + '.json'])
                print('\n'.join([dash_str + f'\n{repr(tmp)}', dash_str, f'Already extracted {res[2]}.\n']))

def recipe_website_dict(path_to_data: str = './recipe_data', file_name: str = 'recipe_website_list.txt') -> \
        Dict[str, Tuple[List[str], List[str]]]:
    """Read in .txt file of recipe titles and websites separated by recipe type.

    By default uses ./recipe_data/recipe_website_list.txt as read in file. Format should be:
        'Cakes
        lemon pound cake, website.html
        ...
        chocolate cake, website.html

        Cookies
        chocolate chip, website.html
        ...
        vanilla cookies, website.html
        '
    Note each recipe type block is separated by a blank line!.

    :param path_to_data: Path to './recipe_data'.
    :param file_name: Defaults to recipe_website_list.txt, specify is you want to add your own.
    :return: Dictionary containing recipe types (e.g. cakes) as keys and a tuple containing a list of recipe names and
        websites.
    """
    recipe_dict = dict()
    with open('/'.join([path_to_data, file_name]), 'r') as f:
        lines = [x.strip() for x in f.readlines()]
        for line in lines:
            if len(line) > 0:
                line = line.split(', ')

                if len(line) == 2:
                    recipe_dict[recipe_title][0].append(line[0])
                    recipe_dict[recipe_title][1].append(line[1])
                else:
                    recipe_title = line[0]
                    recipe_dict[recipe_title] = ([], [])

    return recipe_dict


def scrape_recipe_websites(websites: List[str], recipes: List[str], data_dir: str, return_files: bool = False,
                           verbose: bool = False) -> \
        Optional[List[str]]:
    """Scrape websites for recipes. Stores all info html info in txt files.

    :param websites: List of websites.
    :param recipes: List of recipe titles corresponding to the websites.
    :param data_dir: Location for 'recipe_data' path.
    :param return_files: If true, a list of the scraped websites paths are returned.
    :param verbose: If True, print out the file scraped. If a file already exists print that out as well.
    :return: Path to html_files or None.
    """
    html_files = []
    for url, rec in zip(websites, recipes):
        response = requests.get(url)
        if response.status_code == 200:
            file_name = '/'.join([data_dir, 'html_files', re.sub(' ', '_', rec) + '.txt'])
            if not os.path.exists(file_name):
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)

                with open(file_name, 'w') as text_file:
                    text_file.write(text)

                if verbose:
                    tmp = '/'.join(['./recipe_data/html_files', re.sub(' ', '_', rec) + '.txt'])
                    print('\n'.join([dash_str + f'\n{repr(tmp)}', dash_str, f'Extracted html content for {rec}.\n']))
            else:
                if verbose:
                    tmp = '/'.join(['./recipe_data/html_files', re.sub(' ', '_', rec) + '.txt'])
                    print('\n'.join(
                        [dash_str + f'\n{repr(tmp)}', dash_str, f'Already extracted html content for {rec}.\n']))
            html_files.append(file_name)
        else:
            print('Could not load: %s' + rec)

    return html_files if return_files else None