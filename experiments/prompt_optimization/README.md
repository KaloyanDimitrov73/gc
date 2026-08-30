# DSPy question-processing prompt optimization

This experiment starts from HubLink's existing YAML question-processing prompt
and optimizes its instruction against retrieval quality without requiring gold
`components` or `keywords`.

The optimized program produces schema-independent semantic components and exact
lexical routing keywords. The existing HubLink retriever embeds those components,
optionally runs configured sparse channels when keywords are present, and returns
ranked hubs and paths. The optimization metric is `hit@10_triples`: the fraction
of `golden_triples` found in the first ten ranked retrieved triples. It reuses
the same pure matching function as HubLink's existing `HitAtKEvaluator`.
Retrieved source IDs remain in the result CSVs for diagnosis but do not affect
the score. Partial-answer and final-answer LLM calls are deliberately excluded.

## 1. Install

From `experiments/prompt_optimization/`, create the dedicated environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This environment is separate from `experiments/.venv`. `requirements.txt`
installs the local HubLink retrieval core and DSPy/LiteLLM versions compatible
with the current backend stack.
The API/Guardrails package is not installed because it is not part of this
experiment and its Faker constraint conflicts with ORKG's constraint.

## 2. Create the dataset splits

From `experiments/`:

```powershell
.\prompt_optimization\.venv\Scripts\python.exe `
  -m prompt_optimization.split_dataset
```

This creates `prompt_optimization/splits/{train,validation,test}.csv`. The split
is 60/20/20 by five folds, grouped by `based_on_template`, and stratified by
`retrieval_operation`. The source CSV is never modified.

## 3. Run optimization

The default pipeline is the KIT configuration. Set `KIT_TOOLBOX_API_KEY` in
`experiments/.env`, then run:

```powershell
.\prompt_optimization\.venv\Scripts\python.exe `
  -m prompt_optimization.run_optimization
```

For a cheap integration smoke test before a full compile:

```powershell
.\prompt_optimization\.venv\Scripts\python.exe `
  -m prompt_optimization.run_optimization `
  --train-limit 10 `
  --validation-limit 5 `
  --test-limit 5
```

COPRO defaults to five candidates per round and two iterative rounds. Adjust
the search budget with `--breadth` and `--depth`, for example:

```powershell
.\prompt_optimization\.venv\Scripts\python.exe `
  -m prompt_optimization.run_optimization `
  --breadth 10 `
  --depth 3
```

Use another existing HubLink pipeline configuration with:

```powershell
.\prompt_optimization\.venv\Scripts\python.exe `
  -m prompt_optimization.run_optimization `
  --pipeline-config path\to\pipeline_config.json
```

The DSPy model defaults to the query LLM in that configuration. It can be
overridden with `--dspy-model`, `--dspy-api-base`, and `--dspy-api-key-env`.
The seed prompt defaults to HubLink's existing
`question_processing_prompt.yaml`; use `--initial-prompt` to select another
compatible YAML prompt.

## Outputs

`prompt_optimization/results/` contains:

- `legacy_validation.csv`: current YAML prompt on the validation set;
- `unoptimized_validation.csv`: the YAML instruction seeded into DSPy, before
  compilation;
- `copro_candidate_validation.csv`: COPRO's best training-set candidate on the
  held-out validation set;
- `legacy_test.csv`, `unoptimized_test.csv`, `copro_candidate_test.csv`, and
  `optimized_test.csv` (the validation-selected program);
- `initial_instruction.txt`: YAML instruction used to seed DSPy;
- `optimized_question_processing.json`: compiled DSPy program;
- `optimized_instruction.txt`: human-readable optimized instruction;
- `summary.json`: aggregate comparison scores.

The original YAML prompt is not modified.

The legacy and unoptimized variants share the same instruction content, but
they are not byte-for-byte identical requests: legacy uses the existing
LangChain prompt/parser, while DSPy renders typed input and output fields with
its adapter. This separates prompt optimization from the effect of adopting
DSPy's structured calling convention.

User feedback is intentionally not part of this first experiment. Here the
offline gold retrieval evidence is the controlled optimization signal. Feedback
can later be added as a curated evaluation signal after its reliability,
per-user/session scope, and resistance to manipulation have been defined.

## Experimental controls and limitations

- Keep the pipeline config, model, embedding model, indices, and retrieval
  parameters fixed across all three variants.
- The prompt experiment ranks individual triples, while a normal pipeline run
  may package several triples into one retrieved context. Both use the same
  Hit@k matching formula, but the ranked item granularity can differ.
- If both `use_bm25_hybrid_search` and `use_splade_hybrid_search` are false,
  keywords cannot affect retrieval; only component extraction is optimized.
- Sparse routing requires a previously built matching sparse index.
- The fixed `retrieval_contract` is passed as an input on every DSPy call so an
  optimized instruction cannot silently remove the schema-independence
  requirement.
- This dataset covers one scholarly KG. It tests a schema-independent contract,
  but cross-schema generalization still requires an unchanged test on another KG.
- COPRO performs instruction-only optimization. It searches on the training set;
  the candidate replaces the original DSPy instruction only when it also scores
  higher on validation. Compilation evaluates roughly `breadth * depth` prompt
  batches and can still make many LLM and retrieval calls, so start with the
  limit flags.

To use the saved artifact, construct `QuestionProcessingRetrievalProgram` with a
HubLink adapter and call `program.load(path)`. The JSON stores DSPy's optimized
state; `optimized_instruction.txt` is the readable prompt-level result.
