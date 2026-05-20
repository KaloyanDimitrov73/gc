# QA Module Configuration

This module drives the HubLink-based retrieval pipeline. Configuration is split across two files:

- **`.env`** (at the backend root) — runtime secrets and path overrides
- **`configs/default_hublink_config.json`** — retrieval behaviour and update strategy

---

## `.env` Parameters

| Variable | Required | Description |
|---|---|---|
| `ORKG_EMAIL` | For setup only | ORKG sandbox account email. Only needed when running `init_graph_cache.py`. Not needed at normal backend startup. |
| `ORKG_PASSWORD` | For setup only | ORKG sandbox account password. Same as above. |
| `VDL_API_KEY` | Yes | API key for the VDL LLM endpoint used during retrieval. |
| `OPENAI_API_KEY` | Fallback | Used if `VDL_API_KEY` is not set. |
| `RETRIEVAL_KG_CONFIG_PATH` | No | Override the path to the HubLink config JSON. Defaults to `configs/test_on_publication_dataset/default_hublink_config_deep_distributed.json`. |

---

## `configs/default_hublink_config.json`

### Knowledge Graph Parameters (`knowledge_graph_config.additional_params`)

These control how the local graph cache is managed.

| Parameter | Default | Description |
|---|---|---|
| `orkg_base_url` | `"https://sandbox.orkg.org"` | The ORKG instance to connect to. Must point to the instance where papers were uploaded. |
| `subgraph_root_entity_id` | `"R659055"` | The root entity ID that defines the boundary of the subgraph to cache. |
| `force_cache_update` | `false` | When `true`, re-downloads the full subgraph from ORKG on the next backend start, overwriting the local cache. Reset to `false` after restarting. |
| `force_publication_update` | `false` | When `true`, deletes and re-uploads all papers to ORKG before caching. Only meaningful when running `init_graph_cache.py`, not at normal backend startup. |
| `contribution_building_blocks` | see config | Defines which ORKG contribution names are included when caching the subgraph. Must match the contribution names used when papers were uploaded. |

> **Note on config hash stability:** `force_cache_update` and `force_publication_update` are intentionally excluded from the config hash computation. The cache file names (JSON graph cache, ChromaDB vector store directory) are derived from this hash, so including these flags would cause the backend to miss existing caches whenever the flags are toggled. `force_index_update` and `check_updates_during_retrieval` are not part of the hash formula either (the vector store path uses only the KG, embedding, and LLM config hashes), so toggling them is also safe.


### Retrieval Parameters (`additional_params`)

These control how HubLink indexes and searches the graph. Only the most relevant parameters are documented here. For the full list with descriptions of every parameter, see [`HublinkImplementation/sqa_system/retrieval/implementations/HubLink/models/hub_link_settings.py`](../../../../../HublinkImplementation/sqa_system/retrieval/implementations/HubLink/models/hub_link_settings.py).

**Update strategy**

| Parameter | Default | Description |
|---|---|---|
| `force_index_update` | `false` | When `true`, each hub is re-traversed at startup to check whether its content has changed (via path hash comparison). Contacts the live ORKG instance, but results are limited to triples already present in the local cache (`only_subgraph_mode`). Changes to existing cached triples are reflected; completely new triples in ORKG are not picked up — use `force_cache_update` first for that. Embeddings are only rebuilt for hubs whose content actually changed. Reset to `false` after restarting. |
| `check_updates_during_retrieval` | `false` | When `true`, hubs visited during a query are re-checked instead of using cached paths (same source as `force_index_update`, but triggered per query rather than at startup). Contacts the live ORKG instance per query, but same constraint applies: only triples already in the local cache are used. New ORKG triples require `force_cache_update` first. Slower per query but avoids a full re-index. Only works when `use_topic_if_given: true`. |

**Hub classification**

| Parameter | Default | Description |
|---|---|---|
| `hub_types` | `["Paper"]` | Entity types in the graph that are classified as hub roots. Can be type name strings or `(predicate, object)` tuples. |
| `hub_edges` | `-1` | Minimum number of outgoing edges an entity must have to be classified as a hub. `-1` disables edge-count classification. |

**Indexing**

| Parameter | Default | Description |
|---|---|---|
| `embedding_config` | see config | The embedding model used to embed hub paths into the vector store. |
| `distance_metric` | `"cosine"` | Distance metric for the vector store. Options: `cosine`, `l2`, `ip`. |
| `indexing_root_entity_ids` | `["R659055"]` | Entity IDs from which hub indexing starts. |
| `indexing_root_entity_types` | `null` | (Optional) Entity types from which hub indexing starts. Used 

**Retrieval**

| Parameter | Default | Description |
|---|---|---|
| `number_of_hubs` | `30` | How many hub candidates are compared per query. |
| `top_paths_to_keep` | `10` | How many top-scoring hub paths are passed to the answer generation step. |

**Topic-guided traversal** (only active when `use_topic_if_given: true`)

| Parameter | Default | Description |
|---|---|---|
| `use_topic_if_given` | `false` | When `true` and a topic entity is provided with the query, traversal starts from that entity rather than from all indexed hubs. Enables `max_level`, `compare_hubs_with_same_hop_amount`, and `check_updates_during_retrieval`. |


**Output**

| Parameter | Default | Description |
|---|---|---|
| `filter_output_context` | `true` | When `true`, an LLM filters the retrieved context after answer generation to remove irrelevant entries. Does not affect the answer itself. |
---

## Common Scenarios

| Goal | What to change |
|---|---|
| **Normal startup** (use existing cache) | Leave all flags at `false` (defaults). |
| **Re-download graph from ORKG** (e.g. ORKG was updated) | Set `force_cache_update: true`. Restart backend. Reset to `false` afterwards. |
| **Rebuild hub embeddings** (e.g. graph content changed) | Set `force_index_update: true`. Restart backend. Reset to `false` afterwards. |
| **Full refresh** (new graph data + new embeddings) | Set both `force_cache_update: true` and `force_index_update: true`. Restart. Reset both. |
| **Change embedding model** | Delete the vector store contents manually, then set `force_index_update: true` and restart. `force_index_update` alone does not re-embed if graph content is unchanged. |
| **Lazy freshness without re-indexing** | Set `check_updates_during_retrieval: true` (also requires `use_topic_if_given: true`). Hubs are re-checked on each query. |

> **Note:** Setting `force_cache_update: true` without `force_index_update: true` re-downloads the graph but leaves the hub embeddings stale. The retrieval output will not change until `force_index_update` is also set.
