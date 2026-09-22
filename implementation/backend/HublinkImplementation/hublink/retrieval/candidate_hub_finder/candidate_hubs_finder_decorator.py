from typing import List
from typing_extensions import override

from ..models.processed_question import ProcessedQuestion
from .candidate_hubs_finder import CandidateHubsFinder
from ...core.models.hub_path import HubPath


class CandidateHubsFinderDecorator(CandidateHubsFinder):
    """
    Base decorator for CandidateHubsFinder. Wraps another CandidateHubsFinder
    and by default just delegates to it. Concrete decorators subclass this
    and augment the delegated result (e.g. fusing in another search channel).
    """

    def __init__(self, candidate_hub_finder: CandidateHubsFinder):
        self.candidate_hub_finder = candidate_hub_finder

    @override
    def find_candidate_hubs(self, processed_question: ProcessedQuestion) -> dict[str, List[HubPath]]:
        return self.candidate_hub_finder.find_candidate_hubs(processed_question)
