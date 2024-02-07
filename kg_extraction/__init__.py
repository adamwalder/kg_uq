"""Import the functions needed to extract a knowledge graph by web-scraping recipes. """
__all__ = [
    "scrape_recipe_websites",
    "extract_ingredients_directions",
    "extract_recipe_kg",
    "kg_to_hetero",
    "get_recipe_kg",
    "recipe_website_dict"

]

from kg_extraction.recipe_kg_extraction import (
    scrape_recipe_websites, extract_ingredients_directions, extract_recipe_kg, recipe_website_dict
)

from kg_extraction.recipe_data_load import kg_to_hetero, get_recipe_kg

