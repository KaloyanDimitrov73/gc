import pytest

from sqa_system.core.data.models import Triple, Knowledge
from sqa_system.qa_generator import QACandidateExtractor, QASimilarityMatcher  

def test_get_topic_candidate_from_triple():
    """Test whether the topic entity candidate can be retrieved from either the object or the subject of a triple."""
    extractor = QACandidateExtractor(predicate_mappings={})

    old_topic_entity_value = "old_topic_entity"
    topic_is_subject = Triple(
        entity_subject=Knowledge(uid="1", text="old_topic_entity"),
        entity_object=Knowledge(uid="2", text="object"),
        predicate="predicate"
    )
    topic_is_object = Triple(
        entity_subject=Knowledge(uid="3", text="subject"),
        entity_object=Knowledge(uid="4", text="old_topic_entity"),
        predicate="predicate"
    )
    topic_is_none = Triple(
        entity_subject=Knowledge(uid="5", text="subject"),
        entity_object=Knowledge(uid="6", text="object"),
        predicate="predicate"
    )

    assert extractor.get_topic_candidate_from_triple(
        topic_is_subject, old_topic_entity_value) == topic_is_subject.entity_subject
    assert extractor.get_topic_candidate_from_triple(
        topic_is_object, old_topic_entity_value) == topic_is_object.entity_object
    assert extractor.get_topic_candidate_from_triple(
        topic_is_none, old_topic_entity_value) is None


def test_triple_simple_match():
    """Test whether the triple is correctly matched with the golden triples."""
    extractor = QACandidateExtractor(predicate_mappings={})

    triple = Triple(
        entity_subject=Knowledge(uid="R9999", text="subject"),
        entity_object=Knowledge(uid="R9991", text="object"),
        predicate="predicate"
    )

    golden_triples = ["(R123: Subject, predicate, R456: Object)"]
    new_golden_triple_candidates = {}

    extractor.update_golden_triple_candidates_with_triple(
        triple, golden_triples, new_golden_triple_candidates)

    set_of_candidates = list(new_golden_triple_candidates.values())[0]
    assert len(set_of_candidates) == 1
    assert set_of_candidates.pop() == triple


def test_triple_no_match():
    """Test whether the triple is not matched with the golden triples."""
    extractor = QACandidateExtractor(predicate_mappings={})

    triple = Triple(
        entity_subject=Knowledge(uid="R9999", text="I am Title A"),
        entity_object=Knowledge(uid="R9991", text="I am title B"),
        predicate="predicate"
    )

    golden_triples = ["(R123: Subject, predicate 2, R456: Object)"]
    new_golden_triple_candidates = {}

    extractor.update_golden_triple_candidates_with_triple(
        triple, golden_triples, new_golden_triple_candidates)

    assert len(new_golden_triple_candidates) == 0


def test_triple_complex_triple():
    """Test whether the triple is correctly matched with the golden triples, even with complex text."""
    extractor = QACandidateExtractor(predicate_mappings={})

    triple = Triple(
        entity_subject=Knowledge(
            uid="R9999", text="This, is, a , very, weird. Andcomplex, title!"),
        entity_object=Knowledge(
            uid="R9991", text="This, is, a, ver, weid and COMPLEx txt...,,,,"),
        predicate="predicate"
    )

    golden_triples = ["(R123: This, is, a , very, weird. Andcomplex, title!, predicate, R456: This, is, a, ver, weid and COMPLEx txt...,,,,)",
                      "(R94: Subject, predicate, R91: Object)"]
    new_golden_triple_candidates = {}

    extractor.update_golden_triple_candidates_with_triple(
        triple, golden_triples, new_golden_triple_candidates)

    set_of_candidates = list(new_golden_triple_candidates.values())[0]
    assert triple in set_of_candidates
    
def test_triple_multiple_matches():
    """Test whether the triple is correctly matched with multiple golden triples."""
    extractor = QACandidateExtractor(predicate_mappings={})

    triples = [
        Triple(
            entity_subject=Knowledge(uid="R9999", text="I contain the Test"),
            entity_object=Knowledge(uid="R9991", text="object"),
            predicate="predicate"
        ),
        Triple(
            entity_subject=Knowledge(uid="R8888", text="contain"),
            entity_object=Knowledge(uid="R8881", text="Object"),
            predicate="predicate"
        )
    ]

    golden_triples = ["(R123: contain, predicate, R456: Object)"]
    new_golden_triple_candidates = {}
    
    for triple in triples:
        extractor.update_golden_triple_candidates_with_triple(
            triple, golden_triples, new_golden_triple_candidates)

    set_of_candidates = new_golden_triple_candidates[golden_triples[0]]
    
    assert len(set_of_candidates) == 2
    
    similarity_matcher = QASimilarityMatcher()
    best_candidate = similarity_matcher.select_most_matching_candidate(
        candidates=list(set_of_candidates),
        target=golden_triples[0]
    )
    assert best_candidate == triples[1]
    
def test_triple_boolean_match():
    """Test whether the triple is correctly matched with the golden triples, even with boolean values."""
    extractor = QACandidateExtractor(predicate_mappings={})

    triple = Triple(
        entity_subject=Knowledge(uid="R9999", text="Uses Input Data"),
        entity_object=Knowledge(uid="R9991", text="True"),
        predicate="None"
    )

    golden_triples = ["(R868919:Uses Input Data, None, L1521123:True)"]
    new_golden_triple_candidates = {}

    extractor.update_golden_triple_candidates_with_triple(
        triple, golden_triples, new_golden_triple_candidates)

    set_of_candidates = list(new_golden_triple_candidates.values())[0]
    
    assert len(set_of_candidates) == 1
    assert set_of_candidates.pop() == triple


def test_triple_boolean_to_textmatch():
    """Test whether the triple is correctly matched with the golden triples, even with boolean values."""
    extractor = QACandidateExtractor(predicate_mappings={
        
    })

    triple = Triple(
        entity_subject=Knowledge(uid="R9999", text="Evidence 1"),
        entity_object=Knowledge(uid="R9991", text="Input data was used but is not available"),
        predicate="Uses Input Data"
    )

    golden_triples = ["(R868919:Uses Input Data, None, L1521123:True)"]
    new_golden_triple_candidates = {}

    extractor.update_golden_triple_candidates_with_triple(
        triple, golden_triples, new_golden_triple_candidates)

    set_of_candidates = list(new_golden_triple_candidates.values())[0]
    
    assert len(set_of_candidates) == 1
    assert set_of_candidates.pop() == triple
    
def test_boolean_match():
    """Test whether the triple is correctly matched with the golden triples, even with boolean values."""
    extractor = QACandidateExtractor(predicate_mappings={
        "Has Evaluation Guideline": "Has Guideline",
        "Threats to validity": "threat to validity",
        "Tool is available": "available",
        "Tool was used but is not available": "used",
        "No tool used": "none",
        "Uses Tool Support": "Tool Support",
        "Input data was used but is not available": "used",
        "Input data is available": "available",
        "No input data used": "none",
        "Uses Input Data": "Input Data"
    })

    triple = Triple(
        entity_subject=Knowledge(uid="R9999", text="Classifications_1"),
        entity_object=Knowledge(uid="R9991", text="False"),
        predicate="Evaluation method"
    )

    golden_triples = ["(R869280:Evaluation, Evaluation method, L1521903:False)"]
    new_golden_triple_candidates = {}

    extractor.update_golden_triple_candidates_with_triple(
        triple, golden_triples, new_golden_triple_candidates)

    set_of_candidates = list(new_golden_triple_candidates.values())[0]
    
    assert len(set_of_candidates) == 1
    assert set_of_candidates.pop() == triple
    
def test_predicate_missing():
    """Test whether the triple is correctly matched with the golden triples, even with missing predicates."""
    extractor = QACandidateExtractor(predicate_mappings={
        "Has Evaluation Guideline": "Has Guideline",
        "Threats to validity": "threat to validity",
        "Tool is available": "available",
        "Tool was used but is not available": "used",
        "No tool used": "none",
        "Uses Tool Support": "Tool Support",
        "Input data was used but is not available": "used",
        "Input data is available": "available",
        "No input data used": "none",
        "Uses Input Data": "Input Data"
    })

    triple = Triple(
        entity_subject=Knowledge(uid="R9999", text="Evidence 1"),
        entity_object=Knowledge(uid="R9991", text="Case Study"),
        predicate="Evaluation method"
    )

    golden_triples = ["(R868919:Entity, Name, L1521123:Case Study)"]
    new_golden_triple_candidates = {}

    extractor.update_golden_triple_candidates_with_triple(
        triple, golden_triples, new_golden_triple_candidates)

    set_of_candidates = list(new_golden_triple_candidates.values())[0]
    
    assert len(set_of_candidates) == 1
    assert set_of_candidates.pop() == triple
    
def test_predicate_mapping():
    """Test whether the predicate mapping works as expected."""
    extractor = QACandidateExtractor(predicate_mappings={
        "Has Evaluation Guideline": "Has Guideline",
        "Threats to validity": "threat to validity",
        "Tool is available": "available",
        "Tool was used but is not available": "used",
        "No tool used": "none",
        "Uses Tool Support": "Tool Support",
        "Input data was used but is not available": "used",
        "Input data is available": "available",
        "No input data used": "none",
        "Uses Input Data": "Input Data"
    })
    triple = Triple(
        entity_subject=Knowledge(uid="R9999", text="Classification 1"),
        entity_object=Knowledge(uid="R9991", text="False"),
        predicate="Has Evaluation Guideline"
    )

    golden_triples = ["(R868919:Evaluation, Has Guideline, L1521123:False)"]
    new_golden_triple_candidates = {}

    extractor.update_golden_triple_candidates_with_triple(
        triple, golden_triples, new_golden_triple_candidates)

    set_of_candidates = list(new_golden_triple_candidates.values())[0]
    
    assert len(set_of_candidates) == 1
    assert set_of_candidates.pop() == triple
    
def test_predicate_object_switch():
    """Test cases where the predicate and object is switched."""
    extractor = QACandidateExtractor(predicate_mappings={
        "Has Evaluation Guideline": "Has Guideline",
        "Threats to validity": "threat to validity",
        "Tool is available": "available",
        "Tool was used but is not available": "used",
        "No tool used": "none",
        "Uses Tool Support": "Tool Support",
        "Input data was used but is not available": "used",
        "Input data is available": "available",
        "No input data used": "none",
        "Uses Input Data": "Input Data"
    })
    triple = Triple(
        entity_subject=Knowledge(uid="R9999", text="Classification 1"),
        entity_object=Knowledge(uid="R9991", text="Validation Research"),
        predicate="paper class"
    )

    golden_triples = ["(R868919:paper class, Validation Research, L1521123:True)"]
    new_golden_triple_candidates = {}

    extractor.update_golden_triple_candidates_with_triple(
        triple, golden_triples, new_golden_triple_candidates)

    set_of_candidates = list(new_golden_triple_candidates.values())[0]
    
    assert len(set_of_candidates) == 1
    assert set_of_candidates.pop() == triple
    
def test_boolean_predicate_object_switch():
    """Tests the case where booleans are switched to predicates."""
    extractor = QACandidateExtractor(predicate_mappings={
        "Has Evaluation Guideline": "Has Guideline",
        "Threats to validity": "threat to validity",
        "Tool is available": "available",
        "Tool was used but is not available": "used",
        "No tool used": "none",
        "Uses Tool Support": "Tool Support",
        "Input data was used but is not available": "used",
        "Input data is available": "available",
        "No input data used": "none",
        "Uses Input Data": "Input Data"
    })
    triple = Triple(
        entity_subject=Knowledge(uid="R9999", text="Classification 1"),
        entity_object=Knowledge(uid="R9991", text="external validity"),
        predicate="threat to validity"
    )

    golden_triples = ["(R868919:Threat to Validity, external validity, L1521123:True)"]
    new_golden_triple_candidates = {}

    extractor.update_golden_triple_candidates_with_triple(
        triple, golden_triples, new_golden_triple_candidates)

    set_of_candidates = list(new_golden_triple_candidates.values())[0]
    
    assert len(set_of_candidates) == 1
    assert set_of_candidates.pop() == triple
    
def test_boolean_predicate_object_switch_2():
    """"Tests the case where booleans are switched to predicates."""
    extractor = QACandidateExtractor(predicate_mappings={
        "Has Evaluation Guideline": "Has Guideline",
        "Threats to validity": "threat to validity",
        "Tool is available": "available",
        "Tool was used but is not available": "used",
        "No tool used": "none",
        "Uses Tool Support": "Tool Support",
        "Input data was used but is not available": "used",
        "Input data is available": "available",
        "No input data used": "none",
        "Uses Input Data": "Input Data"
    })
    triple = Triple(
        entity_subject=Knowledge(uid="R9999", text="Classification 1"),
        entity_object=Knowledge(uid="R9991", text="Input data is available"),
        predicate="Uses Input Data"
    )

    golden_triples = ["(R872250:Input Data, available, L1528103:True)"]
    new_golden_triple_candidates = {}

    extractor.update_golden_triple_candidates_with_triple(
        triple, golden_triples, new_golden_triple_candidates)

    set_of_candidates = list(new_golden_triple_candidates.values())[0]
    
    assert len(set_of_candidates) == 1
    assert set_of_candidates.pop() == triple
    
def test_predicate_subject_switch():
    """Test cases where the predicate and subject is switched."""
    extractor = QACandidateExtractor(predicate_mappings={
        "Has Evaluation Guideline": "Has Guideline",
        "Threats to validity": "threat to validity",
        "Tool is available": "available",
        "Tool was used but is not available": "used",
        "No tool used": "none",
        "Uses Tool Support": "Tool Support",
        "Input data was used but is not available": "used",
        "Input data is available": "available",
        "No input data used": "none",
        "Uses Input Data": "Input Data"
    })
    triple = Triple(
        entity_subject=Knowledge(uid="R9999", text="Classification 1"),
        entity_object=Knowledge(uid="R9991", text="Architecture Optimization"),
        predicate="Research Object"
    )

    golden_triples = ["(R868919:Research Object Entity, Name, L1521123:Architecture Optimization)"]
    new_golden_triple_candidates = {}

    extractor.update_golden_triple_candidates_with_triple(
        triple, golden_triples, new_golden_triple_candidates)

    set_of_candidates = list(new_golden_triple_candidates.values())[0]
    
    assert len(set_of_candidates) == 1
    assert set_of_candidates.pop() == triple

def test_multiple_triples():
    """Test whether multiple triples are correctly matched with the golden triples."""
    extractor = QACandidateExtractor(predicate_mappings={
        "Has Evaluation Guideline": "Has Guideline",
        "Threats to validity": "threat to validity",
        "Tool is available": "available",
        "Tool was used but is not available": "used",
        "No tool used": "none",
        "Uses Tool Support": "Tool Support",
        "Input data was used but is not available": "used",
        "Input data is available": "available",
        "No input data used": "none",
        "Uses Input Data": "Input Data"
    })
    triples = [
        Triple(
            entity_subject=Knowledge(uid="R9999", text="Classifications_1"),
            entity_object=Knowledge(uid="L9991", text="No tool used"),
            predicate="Uses Tool Support"
        ),
        Triple(
            entity_subject=Knowledge(uid="R9999", text="Classifications_1"),
            entity_object=Knowledge(uid="L9991", text="Tool was used but is not available"),
            predicate="Uses Tool Support"
        ),
        Triple(
            entity_subject=Knowledge(uid="R9999", text="Classifications_1"),
            entity_object=Knowledge(uid="L9991", text="Tool is available"),
            predicate="Uses Tool Support"
        )
    ]

    golden_triples = ["(R868919:Tool Support, None, L1521123:True)"]
    new_golden_triple_candidates = {}
    for triple in triples:
        extractor.update_golden_triple_candidates_with_triple(
            triple, golden_triples, new_golden_triple_candidates)

    set_of_candidates = list(new_golden_triple_candidates.values())[0]
    
    assert len(set_of_candidates) == 1
    assert set_of_candidates.pop() == triples[0]
    
    golden_triples = ["(R873675:Tool Support, available, L1530998:True)"]
    new_golden_triple_candidates = {}
    for triple in triples:
        extractor.update_golden_triple_candidates_with_triple(
            triple, golden_triples, new_golden_triple_candidates)

    set_of_candidates = list(new_golden_triple_candidates.values())[0]
    
    assert len(set_of_candidates) == 1
    assert set_of_candidates.pop() == triples[2]
    


if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])
