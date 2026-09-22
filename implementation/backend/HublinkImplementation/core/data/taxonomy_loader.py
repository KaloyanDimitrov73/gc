import json

from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.data.models.taxonomy.taxonomy import Taxonomy


class TaxonomyLoader:
    """
    This loader loads the dimensions of the question Taxonomy as JSON files.
    The locations of the JSON files are specified in the FilePathManager.

    Note that the Knowledge Organization Dimension does not need a explcit
    taxonomy file as its classification is based on the graph and not the
    question.
    """

    def __init__(self):
        self.file_path_manager = FilePathManager()

    def load_answer_content_taxonomy(self) -> Taxonomy:
        """
        Loads the answer content dimension.

        Returns:
            Taxonomy: The loaded answer content taxonomy.
        """
        loaded_taxonomy = self._load_taxonomy("answer_contents.json")
        taxonomy_json = json.dumps(loaded_taxonomy)
        return Taxonomy.model_validate_json(taxonomy_json)

    def load_answer_format_taxonomy(self) -> Taxonomy:
        """
        Loads the answer format dimension.

        Returns:
            Taxonomy: The loaded answer format taxonomy.
        """
        loaded_taxonomy = self._load_taxonomy("answer_format.json")
        taxonomy_json = json.dumps(loaded_taxonomy)
        return Taxonomy.model_validate_json(taxonomy_json)

    def load_retrieval_operation_taxonomy(self) -> Taxonomy:
        """
        Loads the retrieval operation dimension.

        Returns:
            Taxonomy: The loaded question type taxonomy.
        """
        loaded_taxonomy = self._load_taxonomy("retrieval_operation.json")
        taxonomy_json = json.dumps(loaded_taxonomy)
        return Taxonomy.model_validate_json(taxonomy_json)

    def load_intention_count_taxonomy(self) -> Taxonomy:
        """
        Loads the intention count dimension.

        Returns:
            Taxonomy: The loaded question intention taxonomy.
        """
        loaded_taxonomy = self._load_taxonomy("intention_count.json")
        taxonomy_json = json.dumps(loaded_taxonomy)
        return Taxonomy.model_validate_json(taxonomy_json)

    def load_answer_constraints_taxonomy(self) -> Taxonomy:
        """
        Loads the answer constraints dimension.

        Returns:
            Taxonomy: The loaded answer constraints taxonomy.
        """
        loaded_taxonomy = self._load_taxonomy("answer_constraints.json")
        taxonomy_json = json.dumps(loaded_taxonomy)
        return Taxonomy.model_validate_json(taxonomy_json)

    def load_question_goal_taxonomy(self) -> Taxonomy:
        """
        Loads the question goal dimension.

        Returns:
            Taxonomy: The loaded question goal taxonomy.
        """
        loaded_taxonomy = self._load_taxonomy("question_goal.json")
        taxonomy_json = json.dumps(loaded_taxonomy)
        return Taxonomy.model_validate_json(taxonomy_json)

    def load_content_domain_taxonomy(self) -> Taxonomy:
        """
        Loads the content domain dimension.

        Returns:
            Taxonomy: The loaded content domain taxonomy.
        """
        loaded_taxonomy = self._load_taxonomy("content_domain.json")
        taxonomy_json = json.dumps(loaded_taxonomy)
        return Taxonomy.model_validate_json(taxonomy_json)

    def load_answer_credibility_taxonomy(self) -> Taxonomy:
        """
        Loads the answer credibility dimension.

        Returns:
            Taxonomy: The loaded answer credibility taxonomy.
        """
        loaded_taxonomy = self._load_taxonomy("answer_credibility.json")
        taxonomy_json = json.dumps(loaded_taxonomy)
        return Taxonomy.model_validate_json(taxonomy_json)

    def _load_taxonomy(self, taxonomy_name: str) -> dict:
        """
        Loads a JSON taxonomy file by using the file path manager.

        Args:
            taxonomy_name (str): The name of the taxonomy file.

        Returns:
            dict: The loaded taxonomy data.
        """
        path = self.file_path_manager.get_path(taxonomy_name)
        with open(path, "r", encoding="utf-8") as file:
            taxonomy_json = json.load(file)
        return taxonomy_json
