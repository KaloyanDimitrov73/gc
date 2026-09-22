# experiments

Standalone Python package for running and tracking HubLink retrieval experiments.
Results are tracked with [Weave (W&B)](https://weave-docs.wandb.ai/).

---

## Setup

### 1. Create and activate a virtual environment

From the `experiments/` directory:

```bash
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **macOS / Linux:** `source .venv/bin/activate`

### 2. Set up environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your-openai-api-key
VDL_API_KEY=your-vdl-api-key
```

The `.env` file is loaded automatically at runtime via `python-dotenv`. Never commit `.env` — it contains secrets.

### 3. Install HubLink backend

The experiments package depends on `HubLink-Backend`, which must be installed first
from the local source:

```bash
pip install -e ../implementation/backend/HublinkImplementation
```

### 4. Install the experiments package

From this directory (`experiments/`):

```bash
pip install -e .
```

This makes `from implementation.*` imports work in all run scripts.

### 5. Authenticate with Weave / W&B (optional)

Experiments are tracked on [wandb.ai](https://wandb.ai). Log in once on the machine:

```bash
wandb login
```

You will be prompted for your API key (found at wandb.ai → Settings → API Keys).
Credentials are stored in `~/.netrc` and reused automatically on future runs.

The `weave_project_name` field in each run script (e.g.
`"kastel-sdq-meta-research/hublink_parameter_selection_ollama"`) controls the W&B
entity and project where results are posted. Make sure your account has access to
that entity.

#### Running without W&B

If you want to run experiments locally without W&B tracking (e.g. no account or
offline), set `skip_weave=True` in `ExperimentRunnerSettings`:

```python
ExperimentRunnerSettings(
    skip_weave=True,
    execution_strategy=ExecutionStrategyType.NO_EVALUATION,
    ...
)
```

Prediction results are still saved to CSV as usual. Metrics can be computed
afterwards using `FileEvaluator` (see `implementation/experimentation/file_evaluator/`).

---

## Running an experiment

### Option A — Inline config (quick test)

Edit and run [`experiment_runs/1_experiment/runs/0_test_runs/HubLink/test_run_openai_cached_inline.py`](experiment_runs/1_experiment/runs/0_test_runs/HubLink/test_run_openai_cached_inline.py).
The full `ExperimentConfig` is defined as a Python dict inside the script — no JSON
files needed.

```bash
python experiment_runs/1_experiment/runs/0_test_runs/HubLink/test_run_openai_cached_inline.py
```

Adjust the config dict directly in the file to change pipeline parameters, LLM
endpoints, or evaluators.

### Option B — Builder config (recommended)

Edit and run [`experiment_runs/1_experiment/runs/0_test_runs/HubLink/test_run_openai_cached_builder.py`](experiment_runs/1_experiment/runs/0_test_runs/HubLink/test_run_openai_cached_builder.py).
Use `ExperimentConfigBuilder` to assemble the config from separate JSON files:

```bash
python experiment_runs/1_experiment/runs/0_test_runs/HubLink/test_run_openai_cached_builder.py
```

The builder loads two files:
| File | Purpose |
|---|---|
| `base_configs/HubLink/base_config_openai.json` | Baseline pipeline (one `PipelineConfig`) |
| `evaluator_configs.json` (four levels up) | List of `EvaluatorConfig` objects |

To add a parameter sweep, also load a tuning parameters file via `load_parameter_ranges_from_path()`:
```json
[
  {
    "config_name": "retrieval_config",
    "parameter_name": "number_of_hubs",
    "values": [5, 10, 20]
  }
]
```

---

## Adding a new experiment

1. Create a new folder under `experiment_runs/`, e.g.
   `experiment_runs/2_experiment/runs/0_test_runs/MyRetriever/`.

2. Copy the closest existing run script as a template.

3. Update the `sys.path.insert` line so it points **to the `experiments/` directory**
   (count directory levels from your script up to `experiments/`):
   ```python
   sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))
   ```

4. Set a unique `weave_project_name` in `ExperimentRunnerSettings`.

5. Run the script directly. Results (CSV predictions and logs) are saved under
   the `results_folder_path` you specify in `ExperimentRunnerSettings`.

---

## Key settings (`ExperimentRunnerSettings`)

| Field | Default | Description |
|---|---|---|
| `results_folder_path` | `None` | Where CSV results and logs are saved |
| `qa_data_path` | `None` | Path to the QA dataset CSV |
| `weave_project_name` | `None` | W&B entity/project for tracking |
| `debugging` | `False` | Verbose logging |
| `log_to_results_folder` | `False` | Write a log file next to the results |
| `execution_strategy` | `SEQUENTIAL` | `SEQUENTIAL`, `PARALLEL_EVALUATION`, or `NO_EVALUATION` |
| `number_of_workers` | `1` | Parallel workers (for `PARALLEL_EVALUATION`) |
| `skip_base_config` | `False` | Skip the baseline and run only parameter variants |
| `skip_weave` | `False` | Skip W&B/Weave initialisation — run fully offline without credentials |

---

## Package layout

```
experiments/
├── implementation/          # installed package — from implementation.* imports
│   ├── config/models/       # Pydantic config types (ExperimentConfig, PipelineConfig, …)
│   ├── core/                # standalone logger + re-exports of HubLink types
│   ├── data_management/     # FilePathManager, DatasetManager
│   ├── data_loader/         # CSVQALoader
│   ├── experimentation/     # ExperimentRunner, ExperimentConfigBuilder, evaluators
│   ├── models/              # QADataset, QAPair, PipeIOData, ParameterRange
│   ├── pipe/                # HubLink pipe wrappers (may import HubLink directly)
│   └── pipeline/            # RetrievalPipeline, PipelineFactory
├── experiment_runs/         # not installed — run experiments scripts and config JSON files
└── data/prompts             # shared prompt YAML files
```

`FilePathManager` anchors all relative paths to the `experiments/` root
(the directory containing `pyproject.toml`).
