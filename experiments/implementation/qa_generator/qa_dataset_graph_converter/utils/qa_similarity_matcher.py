import difflib


class QASimilarityMatcher:
    """
    Class responsible for matching and selecting the most relevant triples and entities
    based on their string representations.
    """

    def select_most_matching_candidate(self,
                                       candidates: list[any],
                                       target: str,
                                       string_replacements: dict[str, str] = None) -> any:
        """
        Given a list of candidates this function will select the best candidate
        based on the similarity of the target string to the string representation
        of the candidate.

        Args:
            candidates (list[any]): A list of candidates to choose from.
            target (str): The target string to match against.
            string_replacements (dict[str, str], optional): A dictionary of
                string replacements to be applied to the candidates before
                matching. Has to be in lower case. They key is the string to
                be replaced and the value is the string to replace it with.

        Returns:
            any: The best matching candidate.
        """
        if not candidates:
            return None

        if not string_replacements:
            string_replacements = {}

        string_candidates = []
        for candidate in candidates:
            string_candidate = str(candidate).lower()
            if string_replacements:
                for old, new in string_replacements.items():
                    string_candidate = string_candidate.replace(old.lower(), new.lower())
            string_candidates.append(string_candidate)

        scores = []
        for index, candidate in enumerate(string_candidates):
            # Calculate the similarity score
            score = difflib.SequenceMatcher(
                None, candidate, target.lower()).ratio()
            scores.append((candidates[index], score))

        # Select the candidate with the highest score
        best_candidate, _ = max(scores, key=lambda x: x[1])
        return best_candidate
