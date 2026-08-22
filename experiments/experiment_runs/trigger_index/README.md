# HubLink index triggers

These scripts build the indexes required by HubLink experiment runs. Run them
from the `experiments/` directory after installing the experiment and HubLink
backend packages.

The supplied JSON file must contain one HubLink KG retrieval configuration. It
can be a baseline/pipeline config or a full experiment config.

## 1. Build the dense hub index

```powershell
python experiment_runs/trigger_index/index_hubs.py `
  experiment_runs/1_experiment/base_configs/base_config_kit.json
```

Use `--force-update` to rebuild hubs even when cached hub data exists:

```powershell
python experiment_runs/trigger_index/index_hubs.py <config.json> --force-update
```

## 2. Build the sparse path indexes

Run sparse indexing only after dense hub indexing, because it reads the cached
hub paths produced by the first step.

```powershell
python experiment_runs/trigger_index/index_sparse.py `
  experiment_runs/1_experiment/base_configs/base_config_kit.json
```

Without flags, the script builds the sparse channels enabled in the config.
Use flags to select channels explicitly:

```powershell
python experiment_runs/trigger_index/index_sparse.py <config.json> --bm25
python experiment_runs/trigger_index/index_sparse.py <config.json> --splade
python experiment_runs/trigger_index/index_sparse.py <config.json> --bm25 --splade
```

Use the same config for both commands so the dense and sparse index keys match.

## 3. Compare BM25 and SPLADE results

After both sparse indexes have been built, compare the top hub-level results
for the full-sentence query and two keyword-style queries:

```powershell
python experiment_runs/trigger_index/query_sparse.py
```

This command automatically uses
`experiment_runs/1_experiment/base_configs/base_config_kit.json`.

The four default queries are:

1. `Which papers did Georg Buchgeher publish in 2017?`
2. `author: Georg Buchgeher`
3. `publication year: 2017`
4. `How frequently is the research object Technical Debt investigated per publication year?`

The output includes ordinary dense ANN ("normal hub") results, BM25's analyzed
terms, SPLADE's highest-weighted terms, the ranked hubs and best matching path
from each index, and hub overlap between the query formulations. Dense ANN uses
one embedding of each supplied query text so the formulations remain directly
comparable. Repeat `--query` to supply a different comparison:

```powershell
python experiment_runs/trigger_index/query_sparse.py <config.json> `
  --query "Which papers did Georg Buchgeher publish in 2017?" `
  --query "author: Georg Buchgeher" `
  --query "publication year: 2017" `
  --limit 5
```
