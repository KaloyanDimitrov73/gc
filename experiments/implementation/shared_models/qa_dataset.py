from datetime import datetime
from typing import Dict, Generator, List, Tuple
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field

from implementation.shared_models.qa_pair import QAPair


class QADataset(BaseModel):
    """
    Dataset for QAPair objects used in experiments.
    Standalone — does not inherit from the deployment Dataset class.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    data: Dict[str, QAPair] = Field(default_factory=dict)

    def __init__(self, name: str, data: Dict[str, QAPair] = None):
        super().__init__(name=name, data=data or {})

    @classmethod
    def from_qa_pairs(cls, name: str, qa_pairs: List[QAPair]) -> "QADataset":
        return cls(name=name, data={qa.uid: qa for qa in qa_pairs})

    def add_entry(self, entry_id: str, entry: QAPair) -> None:
        if entry_id in self.data:
            raise KeyError(f"Entry {entry_id} already exists in dataset {self.name}.")
        self.data[entry_id] = entry
        self.updated_at = datetime.now()

    def add_entries(self, entries: Dict[str, QAPair]) -> None:
        for key in set(entries.keys()) & set(self.data.keys()):
            del entries[key]
        self.data.update(entries)
        self.updated_at = datetime.now()

    def update_entry(self, entry_id: str, entry: QAPair) -> None:
        if entry_id not in self.data:
            raise KeyError(f"Entry {entry_id} not found in dataset {self.name}.")
        self.data[entry_id] = entry
        self.updated_at = datetime.now()

    def get_entry(self, entry_id: str) -> QAPair:
        if entry_id not in self.data:
            raise KeyError(f"Entry {entry_id} not found in dataset {self.name}.")
        return self.data[entry_id]

    def get_all_entries(self) -> List[QAPair]:
        return list(self.data.values())

    def iterate_entries(self) -> Generator[Tuple[str, QAPair], None, None]:
        yield from self.data.items()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([qa.model_dump() for qa in self.get_all_entries()])

    def get_csv_string(self) -> str:
        return self.to_dataframe().to_csv(index=False)
