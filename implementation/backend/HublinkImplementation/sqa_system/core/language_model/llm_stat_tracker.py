from sqa_system.core.data.models.llm_stats import LLMStats


class LLMStatTracker:
    """
    Class that tracks stats used by LLMs in this project.
    Implements the singleton pattern.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMStatTracker, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.stats: LLMStats = LLMStats()

    def reset(self):
        """Resets the stats"""
        self.stats = LLMStats()

    def get_stats(self) -> LLMStats:
        """Returns the stats"""
        return self.stats

    def add_stats(self, stats: LLMStats):
        """Adds stats to the current stats"""
        self.stats.completion_tokens += stats.completion_tokens
        self.stats.total_tokens += stats.total_tokens
        self.stats.prompt_tokens += stats.prompt_tokens
        self.stats.cost += stats.cost
