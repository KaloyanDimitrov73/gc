from typing import Dict
from pydantic import RootModel

from .taxonomy_category import TaxonomyType


class Taxonomy(RootModel):
    """
    Describes a taxonomy for question classification.
    """
    root: Dict[str, TaxonomyType]
