from typing import List
from pydantic import BaseModel
from implementation.shared_models.pipe_io_data import PipeIOData
from implementation.core import LLMStats
from implementation.resource_management.emission_tracker_manager import EmissionsTrackingData


class PipelineData(BaseModel):
    """
    Data class that holds all tracking data for a single pipeline run.
    """
    pipe_io_data: PipeIOData
    runtime: float
    llm_stats: LLMStats
    emissions_data: EmissionsTrackingData
    weave_url: str


class PipelineDataCollector:
    """
    A class that collects tracking data from a pipeline. It is appended to the 
    pipeline object which then fills the data into the pipeline data collector.
    """
    data: List[PipelineData]

    def __init__(self):
        self.data = []

    def add(self, data: PipelineData):
        """
        Adds a new entry to the data.
        
        Args:
            data (PipelineData): The data to be added.
        """
        self.data.append(data)

    def get_all_entries(self) -> List[PipelineData]:
        """
        Returns all elements as a list.
        
        Returns:
            List[PipelineData]: The list of all entries.
        """
        return self.data
