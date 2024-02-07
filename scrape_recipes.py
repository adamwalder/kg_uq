import os
import sys

open_ai_key = '...'
os.environ['OPENAI_API_KEY'] = open_ai_key
sys.path = ['/Users/walder2/kg_uq/'] + sys.path
path_to_data = '/recipe_data'

from kg_extraction import *

entity_types = {
    "recipe": 'https://schema.org/Recipe',
    "ingredient": "https://schema.org/recipeIngredient",
    "measurement": "https://schema.org/QuantitativeValue",
    "nutrition": 'https://schema.org/nutrition',
}

relation_types = {
    "hasCharacteristic": "https://schema.org/additionalProperty",
    "hasColor": "https://schema.org/color",
    "hasMeasurement": "https://schema.org/hasMeasurement",
    "cookTime": "https://schema.org/cookTime",
    "recipeInstruction": "https://schema.org/recipeInstructions"

}

if __name__ == "__main__":

    recipe_dict = recipe_website_dict(path_to_data=path_to_data)
    cnt = 0
    tot = sum([len(v[1]) for v in recipe_dict.values()])

    for recipe_type in recipe_dict.keys():
        n = len(recipe_dict[recipe_type][1])

        print(f'Scraping {n} {recipe_type} recipes...')
        html_files = scrape_recipe_websites(websites=recipe_dict[recipe_type][1],
                                            recipes=recipe_dict[recipe_type][0],
                                            data_dir=path_to_data,
                                            return_files=True)
        print(f'Done scraping {recipe_type} websites.\n')
        txt_files = extract_ingredients_directions(data_dir=path_to_data, html_files=html_files, return_files=True)
        print(f'Done extracting ingredients and directions from html for {recipe_type}.')
        extract_recipe_kg(entity_types=entity_types, relation_types=relation_types, data_dir=path_to_data,
                          txt_files=txt_files)
        cnt += n
        print(f'Completed {cnt} out of {tot} kg recipe extractions.')
