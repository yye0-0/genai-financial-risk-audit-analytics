# Product validation evidence

## Scope

This change adds a clone-ready frozen-case walkthrough, editable analyst
summaries, original-draft provenance, feedback, review conflict detection,
deterministic narrative checks, and more explicit evaluation run accounting.
It also adds a Chinese PRD, quickstart, user-study templates and an iteration log.

The existing PaySim model metrics are retained. No new fraud model was trained,
no real provider was called, and no user interviews or enterprise deployment
were performed in this change.

## Automated tests

Command: `python -m pytest -q`

Result: **21 passed**, with one upstream Streamlit/Altair deprecation warning.
The original eight system tests remain included. Additional coverage checks:

- preservation of generated original, human edit, feedback and previous status;
- rejection of stale concurrent reviews without an extra audit event;
- migration of an existing database without losing analyst notes;
- mandatory rationale before escalation/closure;
- stable identity and score mapping in the frozen-case walkthrough;
- exclusion of gold/metadata at the provider boundary;
- rejection of schema-valid but control-inconsistent drafts;
- usage and cost accounting with fake providers;
- real-adapter usage extraction using a fake SDK client, with no network call;
- a fully failed real-provider-labelled test run not marked as a completed comparison;
- Streamlit startup, mock generation, human editing, feedback, saving and reload.

Validated runtime: Python 3.12.14, NumPy 1.26.4, pandas 2.2.2,
scikit-learn 1.5.1, SciPy 1.13.1, Pydantic 2.8.2, Streamlit 1.37.1,
PyArrow 17.0.0, pytest 9.2.0. CI additionally targets Python 3.11.
The Arrow dependency is explicitly constrained to remain compatible with the
project's NumPy 1.x environment.

## Frozen-case contract run

```bash
python -m src.evaluation --provider mock --output-dir artifacts/evaluations/product-validation
```

- Cases: 60; prompt variants: 4; attempted/succeeded requests: 240/240.
- Strict-schema-valid outputs: 240/240.
- Source case-file SHA-256:
  `5a346fbcdeb011862eec290ddc1173f15e7ed0baf9d39a7e2a845d3341265b8c`.
- `is_real_provider_run=false`; `is_real_llm_comparison=false`.
- Token usage and estimated cost remain unknown for the mock.

The deterministic mock copies authoritative fields and ignores prompt wording.
Its perfect control scores establish plumbing only, not LLM quality or prompt
superiority. Generated runs are excluded from git and can be recreated with
the command above; the published baseline artifacts remain unchanged.

## Remaining validation

No live API key was configured in the execution environment. A live/local model
smoke run and frozen-model prompt comparison remain pending. The fake SDK test
only verifies adapter integration logic.

The user-study materials are blank plans, not completed research. Task-time
savings, quality improvements, adoption and business benefits require actual
participants and task records. The frozen cases are outcome-stratified and
must not be used to infer production prevalence or queue workload.

Local SQLite records provide an application-level append history, not a
tamper-proof audit service. Reviewer names are demonstration inputs, not
authenticated identities. Role-based permissions and deployment controls
remain out of scope for this public-data PoC.
