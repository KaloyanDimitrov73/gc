from sqa_system.core.data.models import Triple, Knowledge


class QACandidateExtractor:
    """
    Class responsible for extracting candidates from the knowledge graph
    based on the given triples and topic entity value.

    Args:
        string_replacements (dict[str, str], optional): A dictionary of
                string replacements to be applied to the candidates before
                matching. Has to be in lower case. They key is the string to
                be replaced and the value is the string to replace it with.
    """

    def __init__(self, predicate_mappings: dict[str, str]):
        self.string_replacements = predicate_mappings

    def get_topic_candidate_from_triple(self,
                                        triple: Triple,
                                        topic_entity_value: str) -> Knowledge | None:
        """
        Checks whether the given triple is a candidate for a topic entity
        based on the topic entity value. It returns the entity candidate
        if it matches the topic entity value else None.

        Args:
            triple (Triple): The triple to check.
            topic_entity_value (str): The topic entity value to match against.

        Returns:
            Knowledge | None: The entity candidate if it matches the topic
                entity value, else None.
        """
        entity_subject = triple.entity_subject.text
        entity_object = triple.entity_object.text
        if entity_object and entity_object.lower() in topic_entity_value.lower():
            return triple.entity_object
        if entity_subject and entity_subject.lower() in topic_entity_value.lower():
            return triple.entity_subject
        return None

    def update_golden_triple_candidates_with_triple(self,
                                                    triple: Triple,
                                                    golden_triples: list[str],
                                                    new_golden_triple_candidates: dict[str, set[Triple]]):
        """
        This function checks whether the given triple is a candidate for a golden
        triple based on the golden triples. It returns the entity candidate
        if it is a candidate for the golden triple else None.

        Args:
            triple (Triple): The triple to check.
            golden_triples (list[str]): The list of golden triples to match against.
            new_golden_triple_candidates (dict[str, set]): A dictionary to store
                the new golden triple candidates.
        """
        entity_subject = triple.entity_subject.text
        entity_object = triple.entity_object.text
        predicate = triple.predicate
        # Replace the predicate with the mapped predicate
        for pr, replacement in self.string_replacements.items():
            entity_subject = entity_subject.lower().replace(
                pr.lower(), replacement.lower())
            entity_object = entity_object.replace(
                pr.lower(), replacement.lower())
            predicate = predicate.lower().replace(pr.lower(), replacement.lower())

        for golden_triple in golden_triples:
            converted_triple = Triple.from_string(golden_triple)
            if not converted_triple:
                raise ValueError(
                    f"Could not parse golden triple: {golden_triple}")

            is_candidate = False

            if self._check_for_candidate_with_boolean_object(
                    converted_triple,
                    predicate,
                    entity_subject,
                    entity_object):
                is_candidate = True

            elif self._check_for_golden_with_boolean_object(
                    converted_triple,
                    predicate):
                is_candidate = True

            elif self._check_only_object_match(
                    converted_triple,
                    entity_object):
                is_candidate = True

            if not is_candidate:
                continue

            # We found a candidate for the golden triple
            if golden_triple not in new_golden_triple_candidates:
                new_golden_triple_candidates[golden_triple] = set()
            new_golden_triple_candidates[golden_triple].add(triple)

    def _check_only_object_match(self,
                                 converted_triple: Triple,
                                 entity_object: str) -> bool:
        """
        Checks if the given triple is a candidate when only the object of the
        triple matches.

        Args:
            converted_triple (Triple): The golden triple that needs to be 
                converted.
            entity_object (str): The object of the potential candidate.

        Returns:
            bool: True if the triple is a candidate, False otherwise.
        """
        # Simple match where only the object matches
        # e.g. (Uses Input Data, None, Input Data) -> (Uses Input Data, None, Input Data)
        if entity_object and entity_object.lower() in converted_triple.entity_object.text.lower():
            return True
        return False

    def _check_for_golden_with_boolean_object(self,
                                              converted_triple: Triple,
                                              predicate: str) -> bool:
        """
        Checks if the given triple is a candidate when the object of the triple
        is a boolean value.

        Args:
            converted_triple (Triple): The golden triple that needs to be 
                converted.
            predicate (str): The predicate of the potential candidate.

        Returns:
            bool: True if the triple is a candidate, False otherwise.
        """
        # The golden triple has an object that is a boolean value and needs
        # to be converted to a string
        # e.g. (paper class, Validation Research, True) ->
        # (Classification 1, paper class, Validation Research)
        if ("true" in converted_triple.entity_object.text.lower() and
                predicate.lower() in converted_triple.entity_subject.text.lower()):
            return True
        return False

    def _check_for_candidate_with_boolean_object(self,
                                                 converted_triple: Triple,
                                                 predicate: str,
                                                 entity_subject: str,
                                                 entity_object: str) -> bool:
        """
        Checks if the given triple is a candidate when the object of the triple
        is a boolean value.

        Args:
            converted_triple (Triple): The golden triple that needs to be 
                converted.
            predicate (str): The predicate of the potential candidate.
            entity_subject (str): The subject of the potential candidate.
            entity_object (str): The object of the potential candidate.

        Returns:
            bool: True if the triple is a candidate, False otherwise.
        """
        if entity_object and (entity_object.lower() == "true" or entity_object.lower() == "false"):
            # Simple predicate and object match
            # e.g. (Uses Input Data, None, True) -> (Uses Input Data, None, True)
            if (predicate.lower() == converted_triple.predicate.lower() and
                    entity_object.lower() in converted_triple.entity_object.text.lower()):
                return True
            # When the text of the object is different
            if entity_subject and entity_object and (entity_object.lower() == "true"):
                # e.g. (Classifications_1, Uses Tool Support, Tool was used but is not available) ->
                # (Uses Tool Support, None, True)
                if (converted_triple.predicate.lower() in entity_subject.lower() and
                        converted_triple.entity_object.text.lower() in entity_object.lower()):
                    return True
        return False
