import pytest

from sqa_system.core.data.models import QAPair
from sqa_system.qa_generator import QADatasetToGraphConverter
from sqa_system.core.config.models import KnowledgeGraphConfig

# This test is designed to check whether the conversion works correctly
# Note, that if the IDs in the ORKG change this test will need to be updated
# with the new IDs to work correctly.

def test_source_ids_wrong():
    """When the source ids are not provided or wrong the function should return None"""
    research_field = "R659055"
    test_kg_config = KnowledgeGraphConfig.from_dict({
        "additional_params": {
            "contribution_building_blocks": {
                "Classifications_1": [
                    "paper_class_flattened",
                    "research_level_flattened",
                    "all_research_objects_flattened",
                    "validity_flattened",
                    "evidence_flattened"
                ]
            },
            "force_cache_update": True,
            "force_publication_update": False,
            "subgraph_root_entity_id": research_field,
            "orkg_base_url": "https://sandbox.orkg.org"
        },
        "graph_type": "orkg",
        "dataset_config": {
            "additional_params": {},
            "file_name": "merged_ecsa_icsa.json",
            "loader": "JsonPublicationLoader",
            "loader_limit": -1
        }
    })
    
    qa_pair = QAPair(
        question="Among publications that investigate architectural assumptions, how are they distributed per year?",
        golden_answer="Answer",
        golden_triples=[
            "(R873620:Research Object Entity, Name, L1530876:Architectural Assumptions)",
            "(R873599:Improving the Consistency and Usefulness of Architecture Descriptions: Guidelines for Architects, publication year, L1530833:2019)",
            "(R873393:Research Object Entity, Name, L1530418:Architectural Assumptions)",
            "(R873379:Predicting the Performance of Privacy-Preserving Data Analytics Using Architecture Modelling and Simulation, publication year, L1530392:2018)",
            "(R868669:Research Object Entity, Name, L1520592:Architectural Assumptions)",
            "(R868651:Architectural Assumptions and Their Management in Industry – An Exploratory Study, publication year, L1520562:2017)"
        ],
        hops=6,
        is_generated_with="test",
        source_ids=['10.1109/ICSA.2019.IAMBROKEN', '10.1109/ICSA.2019.00024', '10.1109/ICSA.2018.00026', '10.1109/ICSA.2018.00026', '10.1007/978-3-319-65831-5_14', '10.1007/978-3-319-65831-5_14'],
        topic_entity_id="R659055",
        topic_entity_value="Software Architecture and Design",
    )
     
    
    converter = QADatasetToGraphConverter(
        kg_config=test_kg_config,
        string_replacements={
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
        }
    )
    updated_qa_pair = converter._update_qa_pair(qa_pair, converter.get_doi_to_entity_mapping(research_field))
    assert updated_qa_pair == None


def test_deep_to_flat():
    """Test whether the conversion from the deep config to the flat config works."""
    research_field = "R659055"
    test_kg_config = KnowledgeGraphConfig.from_dict({
        "additional_params": {
            "contribution_building_blocks": {
                "Classifications_1": [
                    "paper_class_flattened",
                    "research_level_flattened",
                    "all_research_objects_flattened",
                    "validity_flattened",
                    "evidence_flattened"
                ]
            },
            "force_cache_update": True,
            "force_publication_update": False,
            "subgraph_root_entity_id": research_field,
            "orkg_base_url": "https://sandbox.orkg.org"
        },
        "graph_type": "orkg",
        "dataset_config": {
            "additional_params": {},
            "file_name": "merged_ecsa_icsa.json",
            "loader": "JsonPublicationLoader",
            "loader_limit": -1
        }
    })
    

    qa_pairs = [
        # Research Object
        QAPair(
            question="Among publications that investigate architectural assumptions, how are they distributed per year?",
            golden_answer="Answer",
            golden_triples=[
                "(R873620:Research Object Entity, Name, L1530876:Architectural Assumptions)",
                "(R873599:Improving the Consistency and Usefulness of Architecture Descriptions: Guidelines for Architects, publication year, L1530833:2019)",
                "(R873393:Research Object Entity, Name, L1530418:Architectural Assumptions)",
                "(R873379:Predicting the Performance of Privacy-Preserving Data Analytics Using Architecture Modelling and Simulation, publication year, L1530392:2018)",
                "(R868669:Research Object Entity, Name, L1520592:Architectural Assumptions)",
                "(R868651:Architectural Assumptions and Their Management in Industry – An Exploratory Study, publication year, L1520562:2017)"
            ],
            hops=6,
            is_generated_with="test",
            source_ids=['10.1109/ICSA.2019.00024', '10.1109/ICSA.2019.00024', '10.1109/ICSA.2018.00026', '10.1109/ICSA.2018.00026', '10.1007/978-3-319-65831-5_14', '10.1007/978-3-319-65831-5_14'],
            topic_entity_id="R659055",
            topic_entity_value="Software Architecture and Design",
        ),
        # Input Data
        QAPair(
            question="What are the research objects that are investigated in publications from Davide Arcelli that have not used input data?",
            golden_answer="Answer",
            golden_triples=[
                "(R871306:authors list, has list element, L1526166:Davide Arcelli)",
                "(R871323:Input Data, None, L1526208:True)",
                "(R871815:authors list, has list element, L1527181:Davide Arcelli)",
                "(R871833:Input Data, None, L1527223:True)",
                "(R871320:Research Object Entity, Name, L1526198:Architecture Optimization Method)",
                "(R871828:Research Object Entity, Name, L1527212:Architecture Analysis Method)",
                "(R871830:Research Object Entity, Name, L1527213:Architecture Design Method)"
            ],
            hops=6,
            is_generated_with="test",
            source_ids=['10.1109/ICSA.2018.00020', '10.1109/ICSA.2018.00020', '10.1109/ICSA.2019.00017', '10.1109/ICSA.2019.00017', '10.1109/ICSA.2018.00020', '10.1109/ICSA.2019.00017', '10.1109/ICSA.2019.00017'],
            topic_entity_id="R659055",
            topic_entity_value="Software Architecture and Design",
        ),
        # Paper Class, Property
        QAPair(
            question="What is the paper class that papers are most frequently classified as when they investigate the property with the name Reliability in their evaluations?",
            golden_answer="Answer",
            golden_triples=[
                "(R873741:Property, Name, L1531133:Reliability)",
                "(R873747:Paper Class, evaluation research, L1531144:True)",
                "(R871741:Property, Name, L1527056:Reliability)",
                "(R871751:Paper Class, evaluation research, L1527070:True)",
                "(R870027:Property, Name, L1523497:Reliability)",
                "(R870032:Paper Class, validation research, L1523502:True)",
                "(R871116:Property, Name, L1525763:Reliability)",
                "(R871119:Paper Class, proposal of solution, L1525768:True)",
                "(R871539:Property, Name, L1526638:Reliability)",
                "(R871548:Paper Class, evaluation research, L1526651:True)"
            ],
            hops=7,
            is_generated_with="test",
            source_ids=['10.1007/978-3-030-58923-3_23', '10.1007/978-3-030-58923-3_23', '10.1109/ICSA.2019.00019', '10.1109/ICSA.2019.00019', '10.1109/ICSA.2019.00013', '10.1109/ICSA.2019.00013', '10.1109/ICSA.2017.43', '10.1109/ICSA.2017.43', '10.1109/ICSA.2018.00017', '10.1109/ICSA.2018.00017'],
            topic_entity_id="R659055",
            topic_entity_value="Software Architecture and Design",
        ),
        # Evaluation Method
        QAPair(
            question="What is the evaluation method used in the paper 'Enforcing Architectural Security Decisions'?",
            golden_answer="The evaluation method used in the paper is Case Study.",
            golden_triples=[
                "(R872888:Evaluation Method Entity, Name, L1529410:Case Study)",
            ],
            hops=6,
            is_generated_with="test",
            source_ids=['10.1109/ICSA47634.2020.00012'],
            topic_entity_id="R659055",
            topic_entity_value="Software Architecture and Design",
        ),
        # Research Level
        QAPair(
            question="Which papers have the research level secondary research?",
            golden_answer="Answer",
            golden_triples=[
                '(R869572:Research Level, secondary research, L1522540:True)', '(R874180:Research Level, secondary research, L1532059:True)', '(R868829:Research Level, secondary research, L1520953:True)'
            ],
            hops=4,
            is_generated_with="test",
            source_ids=['10.1007/978-3-030-58923-3_1', '10.1109/ICSA.2019.00023', '10.1109/ICSA.2019.00025'],
            topic_entity_id="R659055",
            topic_entity_value="Software Architecture and Design",    
        ),
        # Threats to Validity
        QAPair(
            question="What are the threats to validity of the paper 'Predicting the Performance of Privacy-Preserving Data Analytics Using Architecture Modelling and Simulation'?",
            golden_answer="answer",
            golden_triples=['(R873384:Threat to Validity, external validity, L1530409:True)', '(R873384:Threat to Validity, internal validity, L1530408:True)'],
            hops=5,
            is_generated_with="test",
            source_ids=['10.1109/ICSA.2018.00026', '10.1109/ICSA.2018.00026'],
            topic_entity_id="R659055",
            topic_entity_value="Software Architecture and Design",  
        ),
        # Replication Package Link
        QAPair(
            question="What is the replication package link of the paper with the title 'A Quantitative Approach for the Assessment of Microservice Architecture Deployment Alternatives by Automated Performance Testing'?",
            golden_answer="Answer",
            golden_triples=[
                "(R872366:Evidence, Replication Package Link, L1528376:https://doi.org/10.5281/zenodo.1256467)"
            ],
            hops=4,
            is_generated_with="test",
            source_ids=['10.1007/978-3-030-00761-4_11'],
            topic_entity_id="R659055",
            topic_entity_value="Software Architecture and Design",
        ),
        # Tool Support
        QAPair(
            question="Is a tool applied in the paper 'An Empirical Study of Architectural Decay in Open-Source Software'?",
            golden_answer="Yes, the paper indicates that tool support is available for the study of architectural decay in open-source software.",
            golden_triples=[
                "(R870120:Tool Support, available, L1523692:True)"
            ],
            hops=4,
            is_generated_with="test",
            source_ids=['10.1109/ICSA.2018.00027'],
            topic_entity_id="R659055",
            topic_entity_value="Software Architecture and Design",
        ),
        # Evaluation Guideline
        QAPair(
            question="What are the names of the properties that are evaluated in papers that investigate the research object with the name Technical Debt without adhering to a evaluation guideline?",
            golden_answer="Answer",
            golden_triples=[
                "(R872794:Research Object Entity, Name, L1529212:Technical Debt)",
                "(R872788:Evaluation, Has Guideline, L1529209:False)",
                "(R869118:Research Object Entity, Name, L1521551:Technical Debt)",
                "(R869112:Evaluation, Has Guideline, L1521548:False)",
                "(R872792:Property, Name, L1529210:Accuracy)",
                "(R869116:Property, Name, L1521549:Maintainability)"
            ],
            hops=7,
            is_generated_with="test",
            source_ids=['10.1109/ICSA51549.2021.00017', '10.1109/ICSA51549.2021.00017', '10.1007/978-3-319-65831-5_4', '10.1007/978-3-319-65831-5_4', '10.1109/ICSA51549.2021.00017', '10.1007/978-3-319-65831-5_4'],
            topic_entity_id="R659055",
            topic_entity_value="Software Architecture and Design",
        )
    ]
    
    converter = QADatasetToGraphConverter(
        kg_config=test_kg_config,
        string_replacements={
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
        }
    )
    updated_qa_pairs = []
    for qa_pair in qa_pairs:
        updated_qa_pair = converter._update_qa_pair(qa_pair, converter.get_doi_to_entity_mapping(research_field))
        updated_qa_pairs.append(updated_qa_pair)
    
    assert updated_qa_pairs[0].hops == 3
    assert set(updated_qa_pairs[0].golden_triples) == set(['(R873601:Classifications_1, Research Object, L1530847:Architectural Assumptions)', '(R873599:Improving the Consistency and Usefulness of Architecture Descriptions: Guidelines for Architects, publication year, L1530833:2019)', '(R873381:Classifications_1, Research Object, L1530401:Architectural Assumptions)', '(R873379:Predicting the Performance of Privacy-Preserving Data Analytics Using Architecture Modelling and Simulation, publication year, L1530392:2018)', '(R868653:Classifications_1, Research Object, L1520573:Architectural Assumptions)', '(R868651:Architectural Assumptions and Their Management in Industry – An Exploratory Study, publication year, L1520562:2017)'])
    
    assert updated_qa_pairs[1].hops == 3
    assert set(updated_qa_pairs[1].golden_triples) == set(['(R871306:authors list, has list element, L1526166:Davide Arcelli)', '(R871307:Classifications_1, Uses Input Data, L1526173:No input data used)', '(R871815:authors list, has list element, L1527181:Davide Arcelli)', '(R871816:Classifications_1, Uses Input Data, L1527189:No input data used)', '(R871307:Classifications_1, Research Object, L1526179:Architecture Optimization Method)', '(R871816:Classifications_1, Research Object, L1527194:Architecture Analysis Method)', '(R871816:Classifications_1, Research Object, L1527195:Architecture Design Method)'])
    
    assert updated_qa_pairs[2].hops == 3
    assert set(updated_qa_pairs[2].golden_triples) == set(['(R873730:Classifications_1, Evaluation Sub-Property, L1531115:Reliability)', '(R873730:Classifications_1, paper class, L1531121:Evaluation Research)', '(R871728:Classifications_1, Evaluation Sub-Property, L1527037:Reliability)', '(R871525:Classifications_1, paper class, L1526623:Evaluation Research)', '(R870016:Classifications_1, Evaluation Sub-Property, L1523480:Reliability)', '(R870016:Classifications_1, paper class, L1523485:Validation Research)', '(R871106:Classifications_1, Evaluation Sub-Property, L1525749:Reliability)', '(R871106:Classifications_1, paper class, L1525752:Proposal of Solution)', '(R871525:Classifications_1, Evaluation Sub-Property, L1526618:Reliability)', '(R871728:Classifications_1, paper class, L1527042:Evaluation Research)'])
    
    assert updated_qa_pairs[3].hops == 3
    assert set(updated_qa_pairs[3].golden_triples) == set(['(R872880:Classifications_1, Evaluation method, L1529388:Case Study)'])
    
    assert updated_qa_pairs[4].hops == 3
    assert set(updated_qa_pairs[4].golden_triples) == set(['(R869568:Classifications_1, research level, L1522528:Secondary Research)', '(R874176:Classifications_1, research level, L1532050:Secondary Research)', '(R868825:Classifications_1, research level, L1520940:Secondary Research)'])
    
    assert updated_qa_pairs[5].hops == 3
    assert set(updated_qa_pairs[5].golden_triples) == set(['(R873381:Classifications_1, threat to validity, L1530398:External Validity)', '(R873381:Classifications_1, threat to validity, L1530399:Internal Validity)'])
    
    assert updated_qa_pairs[6].hops == 3
    assert set(updated_qa_pairs[6].golden_triples) == set(['(R872352:Classifications_1, Replication Package Link, L1528340:https://doi.org/10.5281/zenodo.1256467)'])
    
    assert updated_qa_pairs[7].hops == 3
    assert set(updated_qa_pairs[7].golden_triples) == set(['(R870104:Classifications_1, Uses Tool Support, L1523659:Tool is available)'])
    
    assert updated_qa_pairs[8].hops == 3
    assert set(updated_qa_pairs[8].golden_triples) == set(['(R872782:Classifications_1, Research Object, L1529195:Technical Debt)', '(R872782:Classifications_1, Has Evaluation Guideline, L1529186:False)', '(R869106:Classifications_1, Research Object, L1521537:Technical Debt)', '(R869106:Classifications_1, Has Evaluation Guideline, L1521529:False)', '(R872782:Classifications_1, Evaluation Sub-Property, L1529196:Accuracy)', '(R869106:Classifications_1, Evaluation Sub-Property, L1521524:Maintainability)'])
    
    
def test_deep_to_deep_distributed():
    """Test whether the conversion from the deep config to the deep distributed config works."""
    research_field = "R659055"
    test_kg_config = KnowledgeGraphConfig.from_dict({
        "additional_params": {
            "contribution_building_blocks": {
                "Paper Class 2": [
                    "paper_class"
                ],
                "Research Level 2": [
                    "research_level"
                ],
                "First Research Object 2": [
                    "first_research_object"
                ],
                "Second Research Object 2": [
                    "second_research_object"
                ],
                "Validity 2": [
                    "validity"
                ],
                "Evidence 2": [
                    "evidence"
                ]
            },
            "force_cache_update": True,
            "force_publication_update": False,
            "subgraph_root_entity_id": research_field,
            "orkg_base_url": "https://sandbox.orkg.org"
        },
        "graph_type": "orkg",
        "dataset_config": {
            "additional_params": {},
            "file_name": "merged_ecsa_icsa.json",
            "loader": "JsonPublicationLoader",
            "loader_limit": -1
        }
    })
    

    qa_pairs = [
        QAPair(
            question="What is the property that is evaluated the most often in papers that investigate architecture analysis methods published by Romina Eramo?",
            golden_answer="Answer",
            golden_triples=[
                "(R872742:authors list, has list element, L1529100:Romina Eramo)",
                "(R872755:Research Object Entity, Name, L1529127:Architecture Analysis Method)",
                "(R872753:Property, Name, L1529125:Functional Suitability)",
                "(R871815:authors list, has list element, L1527184:Romina Eramo)",
                "(R871828:Research Object Entity, Name, L1527212:Architecture Analysis Method)",
                "(R871826:Property, Name, L1527210:Effectiveness)"
            ],
            hops=7,
            is_generated_with="test",
            source_ids=['10.1109/ICSA.2018.00022', '10.1109/ICSA.2018.00022', '10.1109/ICSA.2018.00022', '10.1109/ICSA.2019.00017', '10.1109/ICSA.2019.00017', '10.1109/ICSA.2019.00017'],
            topic_entity_id="R659055",
            topic_entity_value="Software Architecture and Design",
        ),
    ]
    
    converter = QADatasetToGraphConverter(
        kg_config=test_kg_config,
        string_replacements={}
    )
    updated_qa_pairs = []
    for qa_pair in qa_pairs:
        updated_qa_pair = converter._update_qa_pair(qa_pair, converter.get_doi_to_entity_mapping(research_field))
        updated_qa_pairs.append(updated_qa_pair)
    
    assert updated_qa_pairs[0].hops == 6
    assert set(updated_qa_pairs[0].golden_triples) == set(['(R872742:authors list, has list element, L1529100:Romina Eramo)', '(R872769:Research Object Entity, Name, L1529153:Architecture Analysis Method)', '(R872767:Property, Name, L1529151:Functional Suitability)', '(R871815:authors list, has list element, L1527184:Romina Eramo)', '(R871844:Research Object Entity, Name, L1527239:Architecture Analysis Method)', '(R871842:Property, Name, L1527237:Effectiveness)'])
    
    
def test_deep_to_flat_distributed():
    """Test whether the conversion from the deep config to the flat distributed config works."""
    research_field = "R659055"
    test_kg_config = KnowledgeGraphConfig.from_dict({
        "additional_params": {
            "contribution_building_blocks": {
                "Paper Class 1": [
                "paper_class_flattened"
                ],
                "Research Level 1": [
                    "research_level_flattened"
                ],
                "First Research Object 1": [
                    "first_research_object_flattened"
                ],
                "Second Research Object 1": [
                    "second_research_object_flattened"
                ],
                "Validity 1": [
                    "validity_flattened"
                ],
                "Evidence 1": [
                    "evidence_flattened"
                ]
            },
            "force_cache_update": True,
            "force_publication_update": False,
            "subgraph_root_entity_id": research_field,
            "orkg_base_url": "https://sandbox.orkg.org"
        },
        "graph_type": "orkg",
        "dataset_config": {
            "additional_params": {},
            "file_name": "merged_ecsa_icsa.json",
            "loader": "JsonPublicationLoader",
            "loader_limit": -1
        }
    })
    

    qa_pairs = [
        QAPair(
            question="Is a tool applied in the paper 'An Empirical Study of Architectural Decay in Open-Source Software'?",
            golden_answer="Yes, the paper indicates that tool support is available for the study of architectural decay in open-source software.",
            golden_triples=[
                "(R870120:Tool Support, available, L1523692:True)"
            ],
            hops=4,
            is_generated_with="test",
            source_ids=['10.1109/ICSA.2018.00027'],
            topic_entity_id="R659055",
            topic_entity_value="Software Architecture and Design",
        )
    ]
    
    converter = QADatasetToGraphConverter(
        kg_config=test_kg_config,
        string_replacements={
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
        }
    )
    updated_qa_pairs = []
    for qa_pair in qa_pairs:
        updated_qa_pair = converter._update_qa_pair(qa_pair, converter.get_doi_to_entity_mapping(research_field))
        updated_qa_pairs.append(updated_qa_pair)
    
    assert updated_qa_pairs[0].hops == 3
    assert set(updated_qa_pairs[0].golden_triples) == set(['(R870140:Evidence 1, Uses Tool Support, L1523734:Tool is available)'])
    
if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])