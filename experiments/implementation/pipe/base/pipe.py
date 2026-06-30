from abc import ABC, abstractmethod
from langchain_core.runnables import RunnableLambda

from implementation.shared_models.pipe_io_data import PipeIOData
from core.progress.progress_handler import ProgressHandler


class Pipe(ABC):
    """
    Interface for a pipe in the 'RetrievalPipeline' object. It is chaied
    with other pipes to form a pipeline that processes the input data.
    In each Pipe, the 'PipeIOData' is processed and extended with new data.

    This interface acts as an adapter for a Runnable LangChain object.
    """

    def process(self, input_data: PipeIOData) -> PipeIOData:
        """
        Process the input data and extend or manipulate the data.

        Args:
            input_data (PipeIOData): The input data that will be processed.
                It is extended or manipulated by the pipe.

        Returns:
            PipeIOData: The processed input data.
        """
        result = self._process(input_data)
        self._update_progress(input_data)
        return result

    @abstractmethod
    def _process(self, input_data: PipeIOData) -> PipeIOData:
        """
        Abstract method that subclasses must implement to process the input.

        Args:
            input_data (PipeIOData): The input data that will be processed.
                It is extended or manipulated by the pipe.

        Returns:
            PipeIOData: The processed input data.
        """

    def get_runnable(self):
        """Returns a LangChain runnable that wraps the process function of the pipe."""
        return RunnableLambda(self.process)

    def _update_progress(self, input_data: PipeIOData):
        """
        To allow the user to see progress of the pipeline, this method allows to update
        the progress bar if a progress_bar_id is provided.

        This is handled by the 'ProgressHandler' class.

        Args:
            input_data (PipeIOData): The input data that will be processed.
                It is extended or manipulated by the pipe.
        """
        if input_data.progress_bar_id is not None:
            ProgressHandler().update_task_by_string_id(input_data.progress_bar_id, 1)
