"""Import the functions needed to extract a knowledge graph by web-scraping recipes. """
__all__ = [
    "scrape_recipe_websites",
    "extract_recipe_content",
    "extract_recipe_kg",
    "recipe_website_dict",
    "extract_kg"
]

from kg_extraction.kg_extraction import extract_kg
from kg_extraction.recipe_extraction import *


