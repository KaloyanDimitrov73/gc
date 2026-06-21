from core.data.models.dataset import Dataset
from core.data.models.publication import Publication


class PublicationDataset(Dataset[Publication]):
    """
    A dataset implementation for handling publication data.

    This class represents a dataset specifically designed for handling publication data. 
    It is a subclass of the `Dataset` class and is parameterized with the `Publication` type.
    """

    def __init__(self, name: str, data: dict[str, Publication]):
        super().__init__(name=name, data=data)
