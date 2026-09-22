import pytest
from sqa_system.knowledge_base.knowledge_graph.storage.utils.path_builder import PathBuilder
from sqa_system.core.data.models import Knowledge, Triple, Subgraph


def test_build_all_paths():
    """Test whether all paths are built correctly from the subgraph."""
    # Create test entities
    n1 = Knowledge(uid="n1", text="n1")
    n2 = Knowledge(uid="n2", text="n2")
    n3 = Knowledge(uid="n3", text="n3")
    n4 = Knowledge(uid="n4", text="n4")
    n5 = Knowledge(uid="n5", text="n5")
    n6 = Knowledge(uid="n6", text="n6")
    n7 = Knowledge(uid="n7", text="n7")
    n8 = Knowledge(uid="n8", text="n8")

    # Create test relations forming a connected graph
    # n1 -> n2 -> n3
    #       n2 -> n8
    #       n2 -> n4 -> n6
    #                   n7
    # n1 <- n5
    r1 = Triple(entity_subject=n1, entity_object=n2, predicate="rel1")
    r2 = Triple(entity_subject=n2, entity_object=n3, predicate="rel2")
    r3 = Triple(entity_subject=n2, entity_object=n4, predicate="rel3")
    r4 = Triple(entity_subject=n4, entity_object=n6, predicate="rel4")
    r5 = Triple(entity_subject=n4, entity_object=n7, predicate="rel5")
    r6 = Triple(entity_subject=n5, entity_object=n1, predicate="rel6")
    r7 = Triple(entity_subject=n2, entity_object=n8, predicate="rel2")

    # Expected paths
    # n5 -> n1 -> n2 -> n3
    # n5 -> n1 -> n2 -> n8
    # n5 -> n1 -> n2 -> n4 -> n6
    # n5 -> n1 -> n2 -> n4 -> n7
    p1 = [r6, r1, r2]
    p2 = [r6, r1, r3, r4]
    p3 = [r6, r1, r3, r5]
    p4 = [r6, r1, r7]

    expected_paths_as_string_list = [
        Triple.convert_list_to_string(p1),
        Triple.convert_list_to_string(p2),
        Triple.convert_list_to_string(p3),
        Triple.convert_list_to_string(p4)
    ]

    # Create paths
    subgraph = [r1, r2, r6, r5, r3, r4, r7]
    path_builder = PathBuilder(Subgraph(root=subgraph))
    paths = path_builder.build_all_paths(
        n5,
        include_tails=True,
        include_against_direction=True)

    # Check if paths are correct
    assert len(paths) == 4

    for path in paths:
        assert Triple.convert_list_to_string(
            path) in expected_paths_as_string_list, f"Path {str(path)} not in expected paths"


if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])
