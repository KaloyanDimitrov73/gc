def compute_sparse_index_key(config) -> str:
    """Returns the cache identity shared by all sparse index channels."""
    return (
        f"{config.knowledge_graph_config.config_hash}_"
        f"{config.index_llm_config.config_hash}"
    )
