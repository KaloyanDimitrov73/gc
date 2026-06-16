import pandas as pd
from sqa_system.core.data.models.dataset.base.dataset import Dataset
from sqa_system.core.data.models.qa_pair import QAPair


class QADataset(Dataset[QAPair]):
    """
    A dataset implementation for storing QAPair objects.

    This class represents a dataset specifically designed for handling QA data. 
    It is a subclass of the `Dataset` class and is parameterized with the `QAPair` type.
    """

    def __init__(self, name: str, data: dict[str, QAPair]):
        super().__init__(name=name, data=data)

    @classmethod
    def from_qa_pairs(cls, name: str, qa_pairs: list[QAPair]) -> 'QADataset':
        """
        Creates a QADataset from a list of QAPairs.

        Args:
            name (str): The name of the dataset.
            qa_pairs (list[QAPair]): A list of QAPairs to be included in the dataset.

        Returns:
            QADataset: An instance of QADataset containing the provided QAPairs.
        """
        qa_dict = {}
        for qa_pair in qa_pairs:
            qa_dict[qa_pair.uid] = qa_pair
        return QADataset(name=name, data=qa_dict)

    def get_csv_string(self) -> str:
        """
        This method converts the dataset to a pandas DataFrame and then to a CSV string.

        Returns:
            str: The dataset represented as a CSV string.
        """
        qa_pairs_dict = [qa_pair.model_dump()
                         for qa_pair in self.get_all_entries()]
        df = pd.DataFrame(qa_pairs_dict)

        return df.to_csv(index=False)
