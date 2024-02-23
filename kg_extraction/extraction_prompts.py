
import os
import json
import asyncio
from typing import Dict, Optional, Tuple, Sequence
from llama_index import Document
from openai import OpenAI as clientOpenAI
from openai import AsyncOpenAI

open_ai_key = os.environ.get('OPENAI_API_KEY', None)

DEFAULT_USER_PROMPT = """Based on the following example, extract entities and relations from the provided text.
Use the following entity types:

# ENTITY TYPES:
{entity_types}

Use the following relation types:
{relation_types}

--> Beginning of example

# Specification
"YUVORA 3D Brick Wall Stickers | PE Foam Fancy Wallpaper for Walls,
 Waterproof & Self Adhesive, White Color 3D Latest Unique Design Wallpaper for Home (70*70 CMT) -40 Tiles
 [Made of soft PE foam,Anti Children's Collision,take care of your family.Waterproof, moist-proof and sound insulated. 
 Easy clean and maintenance with wet cloth,economic wall covering material.,Self adhesive peel and stick wallpaper, 
 Easy paste And removement .Easy To cut DIY the shape according to your room area,The embossed 3d wall sticker offers 
 stunning visual impact. the tiles are light, water proof, anti-collision, they can be installed in minutes over a 
 clean and sleek surface without any mess or specialized tools, and never crack with time.,Peel and stick 3d wallpaper 
 is also an economic wall covering material, they will remain on your walls for as long as you wish them to be. The 
 tiles can also be easily installed directly over existing panels or smooth surface.,Usage range: Featured walls, 
 Kitchen,bedroom,living room, dinning room,TV walls,sofa background,office wall decoration,etc. Don't use in shower 
 and rugged wall surface]. Provide high quality foam 3D wall panels self adhesive peel and stick wallpaper, made of 
 soft PE foam,children's collision, waterproof, moist-proof and sound insulated,easy cleaning and maintenance with wet 
 cloth,economic wall covering material, the material of 3D foam wallpaper is SAFE, easy to paste and remove . Easy to 
 cut DIY the shape according to your decor area. Offers best quality products. This wallpaper we are is a real 
 wallpaper with factory done self adhesive backing. You would be glad that you it. Product features High-density 
 foaming technology Total Three production processes Can be use of up to 10 years Surface Treatment: 3D Deep 
 Embossing Damask Pattern."

################

# Output
[
  {{
    "head": "YUVORA 3D Brick Wall Stickers",
    "head_type": "product",
    "relation": "isProducedBy",
    "tail": "YUVORA",
    "tail_type": "manufacturer"
  }},
  {{
    "head": "YUVORA 3D Brick Wall Stickers",
    "head_type": "product",
    "relation": "hasCharacteristic",
    "tail": "Waterproof",
    "tail_type": "characteristic"
  }},
  {{
    "head": "YUVORA 3D Brick Wall Stickers",
    "head_type": "product",
    "relation": "hasCharacteristic",
    "tail": "Self Adhesive",
    "tail_type": "characteristic"
  }},
  {{
    "head": "YUVORA 3D Brick Wall Stickers",
    "head_type": "product",
    "relation": "hasColor",
    "tail": "White",
    "tail_type": "color"
  }},
  {{
    "head": "YUVORA 3D Brick Wall Stickers",
    "head_type": "product",
    "relation": "hasMeasurement",
    "tail": "70*70 CMT",
    "tail_type": "measurement"
  }},
  {{
    "head": "YUVORA 3D Brick Wall Stickers",
    "head_type": "product",
    "relation": "hasMeasurement",
    "tail": "40 tiles",
    "tail_type": "measurement"
  }},
  {{
    "head": "YUVORA 3D Brick Wall Stickers",
    "head_type": "product",
    "relation": "hasMeasurement",
    "tail": "40 tiles",
    "tail_type": "measurement"
  }}
]

--> End of example

For the following specification, generate extract entities and relations as in the provided example.

# Specification
{specification}
################

# Output

"""

DEFAULT_SYSTEM_PROMPT = """
You are an expert agent specialized in analyzing relationships among people, places, and things. 
Your task is to identify the entities and relations requested with the user prompt. 
You must generate the output in a JSON containing a list with JOSN objects having the following keys: "head", 
"head_type", "relation", "tail", and "tail_type".
The "head" key must contain the text of the extracted entity with one of the types from the provided list in the 
user prompt, the "head_type"
key must contain the type of the extracted head entity which must be one of the types from the provided user list,
the "relation" key must contain the type of relation between the "head" and the "tail", the "tail" key must 
represent the text of an
extracted entity which is the tail of the relation, and the "tail_type" key must contain the type of the tail 
entity. Attempt to extract as many entities and relations as you can.
When you encounter entities that look similar but are differently spelled (e.g. John Doe and J. Doe),
use one name for all occurrences, when extracting them from text.
"""

def get_prompts() -> Tuple[str, str]:
    """Returns the defualt system and user prompt for extracting recipe KG from ingredients and directions.

    :return: Tuple of system prompt string and user prompt string.

    """
    return DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT

class TripleExtractor:
    def __init__(self,
                 entity_types: Dict[str, str],
                 relation_types: Dict[str, str],
                 model: Optional[str] = "gpt-3.5-turbo",
                 system_prompt: Optional[str] = None,
                 user_prompt: Optional[str] = None) -> None:
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

        If you want to change the system_prompt or user_prompt, please first see extraction_prompts.py. The user_prompt
        must specify to return the KG as detailed!

        :param text: Text to have entity, rel, entity extraction from
        :param entity_types: Dictionary detailing entities and their format
        :param relation_types: Dictionary detailing entities and their format
        :param model: model used for extracting relationships. Default to gpt-3.5-turbo
        :param system_prompt: Prompt for extraction task. See extraction_prompts.py for example.
        :param user_prompt: Prompt for how triples should be extracted. See extraction_prompts.py for example
        :return:
        """
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        if user_prompt is None:
            _, user_prompt = get_prompts()

        if system_prompt is None:
            system_prompt, user_prompt = get_prompts()

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.entity_types = entity_types
        self.relation_types = relation_types
        self.model = model if model is not None else "gpt-3.5-turbo"

    async def run_query(self, text: str, file_name: str, kg_name: str) -> Tuple[Optional[str], str, str]:
        if not os.path.exists(file_name):
            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": self.user_prompt.format(
                            entity_types=self.entity_types,
                            relation_types=self.relation_types,
                            specification=text
                        )
                    }
                ]
            )

            return json.loads(completion.choices[0].message.content), file_name, kg_name

        else:
            return None, file_name, kg_name

    def extract_triples(self,
                        text_prompts: Sequence[Document],
                        file_names: Sequence[str],
                        kg_names: Sequence[str]) -> Sequence[Tuple[Optional[str], str, str]]:
        tasks = [self.run_query(x[0].text, x[1], x[2]) for x in zip(text_prompts, file_names, kg_names)]

        return asyncio.run(asyncio.gather(*tasks))


