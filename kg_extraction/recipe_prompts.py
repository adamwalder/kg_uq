from typing import Tuple
### credit for user prompt to
user_prompt = """Based on the following example, extract entities and relations from the provided text.
Use the following entity types:

# ENTITY TYPES:
{entity_types}

Use the following relation types:
{relation_types}

--> Beginning of example

# Specification
"The nutritional content for baked apples consists of The nutritional information for baked apples includes: 147 
calories, 1g fat, 37g carbs, and 1g protein. The following is preparation information for baked apples, prep time: 
15 mins, cook time: 45 mins, total time: 1 hr, servings: 6. The ingredients for making baked apples are butter 
(1 teaspoon), brown sugar (2 tablespoons), vanilla sugar (3 teaspoons), ground cinnamon (3 teaspoons), ground nutmeg 
(1 teaspoon), large apples (6), water (3 ½ tablespoons). To make baked apples, preheat the oven to 350 degrees F. 
Grease a large baking dish with butter. In a small bowl, mix brown sugar, vanilla sugar, cinnamon, and nutmeg. 
Layer sliced apples in the baking dish, sprinkling each layer with the sugar mixture. Bake for 30 minutes, then pour 
water over the apples and continue baking until tender, about 15 minutes more..."

################

# Output
[
    {"head": "baked apples", 
    "head_type": "recipe", 
    "relation": "hasCharacteristic", 
    "tail": "147 calories", 
    "tail_type": "nutrition"}, 
    {"head": "baked apples", 
    "head_type": "recipe", 
    "relation": "hasCharacteristic", 
    "tail": "1g fat", 
    "tail_type": "nutrition"},
    {"head": "baked apples", 
    "head_type": "recipe", 
    "relation": "hasCharacteristic", 
    "tail": "37g carbs", 
    "tail_type": "nutrition"}, 
    {"head": "baked apples", 
    "head_type": "recipe", 
    "relation": "hasCharacteristic", "
    tail": "1g protein", 
    "tail_type": "nutrition"}, 
    {"head": "baked apples", 
    "head_type": "recipe", 
    "relation": "cookTime", 
    "tail": "45 mins", "tail_type": 
    "measurement"}, 
    {"head": "baked apples", 
    "head_type": "recipe", 
    "relation": "recipeInstruction", 
    "tail": "preheat the oven to 350 degrees F", 
    "tail_type": "instruction"}, 
    {"head": "baked apples", 
    "head_type": "recipe", 
    "relation": "recipeInstruction", 
    "tail": "Grease a large baking dish with butter", 
    "tail_type": "instruction"}, 
    {"head": "baked apples", 
    "head_type": "recipe", 
    "relation": "recipeInstruction", 
    "tail": "mix brown sugar, vanilla sugar, cinnamon, and nutmeg", 
    "tail_type": "instruction"}, 
    {"head": "baked apples", 
    "head_type": "recipe", 
    "relation": "recipeInstruction", 
    "tail": "Layer sliced apples in the baking dish, sprinkling each layer with the sugar mixture", "
    tail_type": "instruction"}, 
    {"head": "baked apples", 
    "head_type": "recipe", 
    "relation": "recipeInstruction", 
    "tail": "Bake for 30 minutes", 
    "tail_type": "instruction"}, 
    {"head": "baked apples", 
    "head_type": "recipe", 
    "relation": "recipeInstruction", 
    "tail": "pour water over the apples and continue baking until tender, about 15 minutes more", 
    "tail_type": "instruction"}, 
    {"head": "butter", 
    "head_type": "ingredient", 
    "relation": "hasMeasurement", 
    "tail": "1 teaspoon", 
    "tail_type": "measurement"}, 
    {"head": "brown sugar", 
    "head_type": "ingredient", 
    "relation": "hasMeasurement", 
    "tail": "2 tablespoons", 
    "tail_type": "measurement"},
    {"head": "vanilla sugar", 
     "head_type": "ingredient", 
     "relation": "hasMeasurement", 
     "tail": "3 teaspoons", 
     "tail_type": "measurement"}, 
     {"head": "ground cinnamon", 
     "head_type": "ingredient", 
     "relation": "hasMeasurement", 
     "tail": "3 teaspoons", 
     "tail_type": "measurement"}, 
     {"head": "ground nutmeg", 
     "head_type": "ingredient", 
     "relation": "hasMeasurement", 
     "tail": "1 teaspoon", 
     "tail_type": "measurement"}, 
     {"head": "large apples", 
     "head_type": "ingredient", 
     "relation": "hasMeasurement", 
     "tail": "6", 
     "tail_type": "measurement"}, 
     {"head": "water", 
     "head_type": "ingredient", 
     "relation": "hasMeasurement", 
     "tail": "3 tablespoons", 
     "tail_type": "measurement"}
]

--> End of example

For the following specification, generate extract entities and relations as in the provided example.

# Specification
{specification}
################

# Output

"""

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

def get_recipe_prompts() -> Tuple[str, str]:
    """Returns the defualt system and user prompt for extracting recipe KG from ingredients and directions.

    :return: Tuple of system prompt string and user prompt string.

    """
    return system_prompt, user_prompt