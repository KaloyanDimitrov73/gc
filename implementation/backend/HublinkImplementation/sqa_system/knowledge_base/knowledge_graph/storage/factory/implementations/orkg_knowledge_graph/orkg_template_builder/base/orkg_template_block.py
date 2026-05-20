from abc import ABC
from typing import List


class OrkgTemplateBlock(ABC):
    """
    This block is responsible for adding contributions to the ORKG graph.
    It prepares the exact format that is required by the ORKG API to add contributions.
    """

    @staticmethod
    def create_entity_with_description(entities: List[dict], 
                                       label: str) -> List[dict]:
        """
        This prepares a dictionary for adding an entity with a description to the ORKG graph.
        This is an object that has a name and a description.

        Args:
            entities: A list of entities with their description. The structure should be:
                {
                    "name": "Entity Name",
                    "description": "Entity Description" // Optional
                }
            label: The label for the entity list that is used for naming the list in the
                orkg interface
                
        Returns:
            A list of dictionaries that can be used to add the entity with description
            to the ORKG graph.
        """
        entity_dicts = []
        for entity in entities:
            value = {
                "classes": ["C103013"],  # Entity with Description
                "values": {
                    "P15191": [
                        {
                            "text": entity["name"]
                        }
                    ],
                },
                "label": f"{label} Entity"
            }
            if "description" in entity:
                value["values"]["description"] = [
                    {
                        "text": entity["description"]
                    }
                ]
            entity_dicts.append(value)

        return [{
            "classes": ["C103012"],
            "values": {
                "P45132": entity_dicts
            },
            "label": f"{label} List"
        }]

    @staticmethod
    def create_list_of_text(text_list: List[str]) -> List[dict]:
        """
        A basic structure used as part of our template to add a list of text.
        
        Args:
            text_list: A list of strings that will be added to the ORKG graph.
        Returns:
            A list of dictionaries that can be used to add the list of text 
            to the ORKG graph.
        """
        texts = []
        for text in text_list:
            texts.append(
                {
                    "text": text
                }
            )
        return [{
            "classes": ["C103008"],
            "values": {
                "P162014": texts
            },
            "label": "",
        }]
