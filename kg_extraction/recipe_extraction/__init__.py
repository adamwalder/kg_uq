"""Import the functions needed to extract a knowledge graph by web-scraping recipes. """
__all__ = [
    "scrape_recipe_websites",
    "extract_recipe_content",
    "extract_recipe_kg",
    "recipe_website_dict"

]

from kg_extraction.recipe_extraction.recipe_extractors import (
    extract_recipe_content,
    extract_recipe_kg,
    scrape_recipe_websites,
    recipe_website_dict
)

