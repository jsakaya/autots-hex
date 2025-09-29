# AutoTS Hexagonal Redesign with Darts Backend

## 1. Purpose & Vision
- Deliver a ground-up reimplementation of AutoTS using hexagonal (ports-and-adapters) architecture.
- Replace the bespoke model zoo with Darts as the standardized forecasting engine.
- Preserve signature AutoTS capabilities (transform search, genetic AutoML, ensembles, multi-series support) while improving maintainability, extensibility, and testability.

## 2. Scope & Non-Goals
### 2.1 In Scope
- Domain-driven redesign of preprocessing, template search, validation, scoring, and ensembling.
- Abstraction of data handling, model execution, transformations, metrics, persistence, and orchestration into explicit ports.
- Implementation of adapters for:
  - Darts models and transformers.
  - Pandas/numpy data ingestion and output.
  - Local file persistence for templates/results (with future DB-ready interfaces).
  - CLI/REST front-ends.
- Migration utilities to convert limited subsets of legacy AutoTS templates.

### 2.2 Out of Scope (Phase 1)
- Legacy model wrappers outside Darts.
- GUI or full orchestration UI.
- Distributed training across clusters (considered later via adapter swap).

## 3. Guiding Principles
- **Isolation**: Domain core remains unaware of pandas/Darts specifics.
- **Immutability**: Prefer value objects and pure functions to simplify testing.
- **Determinism**: RNG, timers, and external state behind ports for reproducible runs.
- **Extensibility**: New models/transforms require only new adapters or specs, not domain changes.
- **Observability**: Structured events and metrics at each port boundary.

## 4. Target Architecture Overview
```
+------------------+      Ports      +-----------------------+
|  Application     |<-------------->|  Adapters (Darts, CLI |
|  Services        |                |  Filesystem, etc.)    |
+------------------+                +-----------------------+
        | Domain Core (Entities, Value Objects, Domain Services)
        +------------------------------------------------------
```
- Packages:
  - `core/domain`: Entities, value objects, domain services, error types.
  - `core/application`: Use cases (commands/queries), orchestrators.
  - `ports`: Interface definitions for adapters.
  - `adapters/primary`: CLI, HTTP, template importer.
  - `adapters/secondary`: Darts, transformations, persistence, logging, random/time.
  - `config`: Dependency injection wiring, presets, feature flags.
  - `tests`: Unit/integration/end-to-end suites.

## 5. Domain Model & Services
### 5.1 Key Value Objects
- `SeriesId`, `Timestamp`, `Frequency`, `ForecastHorizon` (strong typing around primitives).
- `TimeSeriesSlice`: Immutable matrix + index metadata for wide-format windows.
- `DatasetProfile`: Data shape, frequency, missingness statistics.
- `TransformationSpec`: Ordered transformation identifiers + parameters.
- `ModelSpec`: Darts model identifier + hyperparameters.
- `Template`: Combination of `TransformationSpec`, `ModelSpec`, metadata (IDs, provenance).
- `ValidationFold`: Train/test index ranges, weighting, optional future regressor windows.
- `CandidateResult`: Transformation summary, runtime stats, forecast outputs, raw metrics.
- `MetricVector` and `ScoreBreakdown`: Weighted scores and supporting metrics.

### 5.2 Domain Services
- `DatasetService`: Converts raw input into canonical `TimeSeriesSlice`, handles long→wide, frequency inference, cleaning heuristics.
- `TransformationService`: Applies sequences of transformation ports, tracks handles for inverse operations.
- `ValidationService`: Generates folds per strategy (backwards, even, seasonal, similarity, mixed-length) using deterministic algorithms.
- `SearchService`: Coordinates template execution and interacts with `SearchStrategy` to request new candidates each generation.
- `ScoringService`: Aggregates metrics into composite scores, per-series/per-timestamp breakdowns.
- `EnsembleService`: Builds horizontal and mosaic ensembles using candidate histories.
- `TemplateRegistry`: Maintains catalog of generated/tried templates, avoids duplicates, persists state via repository port.

## 6. Ports (Interfaces)
- `DatasetPort`: Acquire datasets (pandas DataFrame, CSV ingestion, streaming) -> `TimeSeriesSlice`.
- `TransformationPort`: Fit/transform/inverse operations on slices; returns adapter-specific handles.
- `ModelPort`: Train and forecast Darts models; handles probabilistic intervals, exogenous inputs, update-fit scenarios.
- `MetricPort`: Compute metric set (SMAPE, MAE, RMSE, made, pinball, runtime) and per-series aggregates.
- `TemplateRepositoryPort`: Persist templates, results, population state, experiment metadata.
- `SearchStrategyPort`: Provide new templates given scored population (supports GA, random, hill-climb, etc.).
- `EnsembleBuilderPort`: Optional plug-in to allow alternative ensembling algorithms.
- `ExecutionPort`: Abstract over parallel execution pools/workers.
- `RandomPort` & `ClockPort`: RNG seeds/timing.
- `EventPort`: Emit domain events for logging/telemetry/UI.

## 7. Adapters
### 7.1 Darts Model Adapter
- Maps `ModelSpec` to Darts estimator classes, configures with hyperparameters.
- Converts `TimeSeriesSlice` to Darts `TimeSeries` objects, aligning covariates and handling multivariate data.
- Supports training, prediction, and optional incremental update (`fit_from_dataset`, `predict` with n steps).
- Produces point and quantile forecasts, translated back to domain value objects.

### 7.2 Transformation Adapter
- Wrap Darts transformers where possible.
- Re-implement AutoTS-specific transforms (differencing, holiday flags, anomaly removal, FFT, log/shift) using numpy/pandas but hidden behind transformation port.
- Maintain transform provenance for inverse application during prediction.

### 7.3 Persistence Adapter
- Initial implementation: file-backed repository storing JSON/Parquet for templates/results.
- Schema definition via Pydantic or dataclasses for serialization stability.
- Extendable to databases (Postgres, DuckDB) via adapter swap.

### 7.4 Interface Adapters
- CLI (Typer) to configure experiments, inspect results, export templates.
- Optional HTTP API (FastAPI) providing endpoints for dataset registration, experiment run, status, results retrieval.
- Logging adapter bridging domain events to `structlog`/`logging`.

## 8. Data Flow Walkthroughs
### 8.1 Fit Workflow
1. `ExperimentController` receives command with dataset reference, search config, validation config.
2. Loads dataset via `DatasetPort`, obtains `TimeSeriesSlice` + profile.
3. `ValidationService` generates base folds.
4. `TemplateRegistry` seeds initial population (imported templates + random via `SearchStrategyPort`).
5. For each template:
   - `TransformationService` resolves adapters, fits transforms on training fold.
   - `ModelPort` trains Darts model on transformed data.
   - Forecast produced and inversed via transformation handles.
   - `MetricPort` evaluates forecasts against actuals.
   - `ScoringService` calculates composite score.
   - Result persisted through `TemplateRepositoryPort` and published via `EventPort`.
6. `SearchService` aggregates results, requests new candidates from `SearchStrategyPort` each generation until max generations/timeout.
7. Post-search:
   - `ValidationService` reruns winning templates on additional folds if configured.
   - `EnsembleService` composes ensembles and executes them via same execution pipeline.
8. Final results persisted and exposed via CLI/API.

### 8.2 Predict Workflow
- Import selected template/template bundle from repository.
- Load new dataset segment (or reuse existing) through `DatasetPort`.
- Optionally refit transformation handles/model via `ModelPort` update-fit support.
- Produce forecasts for new horizon, apply inverse transforms, enforce constraints.
- Return `ForecastPackage` (point forecast, intervals, diagnostics).

## 9. Genetic Search Strategy
- Implement domain `GeneticSearchStrategy` with configuration:
  - Population size, tournament selection parameters.
  - Crossover rules for `ModelSpec` and `TransformationSpec`.
  - Mutation operators (parameter perturbation, transform insertion/removal).
  - Elitism and diversity constraints (e.g., max templates per model class).
- Provide fallback strategies (pure random, grid subsets) via strategy port.
- Maintain reproducibility with `RandomPort` seeds.

## 10. Ensemble Reimagining
- **Horizontal Ensembles**: Map per-series best candidates using `ScoreBreakdown` and generate ensemble templates referencing base templates.
- **Mosaic Ensembles**: Combine models for defined windows/crosshair profiles; integrate unpredictability scoring domain-side.
- Implement as domain services independent of Darts specifics—ensembles operate on candidate result metadata and delegate final forecast execution through ports.

## 11. Darts Integration Considerations
- Support for probabilistic output: Darts models often emit samples/quantiles; adapter must align with required intervals.
- Handling exogenous covariates: map AutoTS regressor utilities to Darts `future_covariates`/`past_covariates`.
- Batch/multiseries training: evaluate use of `RegressionModel`, `BlockRNNModel`, `TiDEModel`, etc., verifying scalability.
- GPU/CPU selection exposed via adapter configuration.
- Fallback path for models unavailable in target environment (raise domain-level capability errors).

## 12. Implementation Roadmap
### Phase 0 – Setup (Weeks 1–2)
- Establish repository structure, tooling (Poetry, pre-commit, black/ruff/mypy, pytest).
- Document coding standards, branching strategy, CI scaffold.

### Phase 1 – Domain & Ports (Weeks 3–6)
- Implement value objects, domain services stubs with in-memory adapters.
- Write unit tests ensuring orchestration without real model execution.
- Deliver working “no-op” experiment pipeline (mocked transforms/models).

### Phase 2 – Darts & Transformation Adapters (Weeks 7–12)
- Implement Darts model adapter covering core model families.
- Port high-priority transforms (scalers, differencing, log, seasonal diff, anomaly removal, FFT) to transformation port.
- Introduce metric adapter with full AutoTS metric suite.
- Validate single-generation AutoML on sample datasets.

### Phase 3 – Search & Ensembles (Weeks 13–18)
- Implement genetic strategy, template registry, duplicate detection.
- Add multi-generation orchestration, early-stopping/timeout controls.
- Recreate ensemble strategies with new domain models.
- Integration tests comparing against legacy behavior on benchmarks.

### Phase 4 – Persistence & Interfaces (Weeks 19–22)
- File-based template/result repository; JSON schema definitions.
- CLI for experiment management; optional REST API stub.
- Migration tool to import simplified legacy templates.

### Phase 5 – Hardening & Release (Weeks 23–26)
- Performance profiling, parallel execution via `ExecutionPort` (e.g., joblib/Ray adapter).
- Documentation (developer guide, API reference, tutorials, architecture whitepaper).
- Release packaging, Docker image, CI gates.

## 13. Testing Strategy
- **Unit Tests**: Domain services with fake adapters, verifying invariants (no duplicate templates, correct fold generation, scoring math).
- **Adapter Tests**: Darts integration with synthetic data, ensuring transforms+models produce expected outputs.
- **Property-Based Tests**: Random template specs to ensure serialization fidelity and GA invariants.
- **Regression Tests**: Compare new system metrics vs. legacy AutoTS on curated datasets; define acceptable tolerances.
- **Performance Tests**: Benchmark runtime per generation and ensemble overhead.
- **End-to-End Scenarios**: CLI-driven runs from dataset load through prediction export.

## 14. Tooling & DevOps
- Dependency management via Poetry/uv with lockfiles.
- Formatting/linting: black, ruff, isort, mypy (strict optional typing).
- CI: GitHub Actions or similar for lint, unit, integration tests; optional GPU runner for Darts deep models.
- Observability: structured logging, optional OpenTelemetry trace adapter for long experiments.
- Feature flags/config: Pydantic-based settings enabling/disabling model families.

## 15. Risks & Mitigation
| Risk | Impact | Mitigation |
|------|--------|------------|
| Darts feature gaps vs. legacy models | Medium | Prioritize Darts model coverage; design adapters to fallback/flag unsupported features early. |
| Transform parity | High | Audit transforms, stage delivery (core first, esoteric later), allow community plugins. |
| Performance regressions | High | Integrate profiling early; leverage parallel execution; tune default model lists. |
| Complexity of GA tuning | Medium | Encapsulate strategy parameters, provide diagnostics, allow alternative strategies via port. |
| Migration of user templates | Medium | Provide clear conversion tooling and documentation; accept reduced compatibility in v1 with communicated roadmap. |

## 16. Deliverables Checklist
- ✅ Architecture blueprint & domain model documentation.
- ✅ Port interface definitions with docstrings and diagrams.
- ✅ Implemented domain services with coverage.
- ✅ Darts adapters with supported model catalog.
- ✅ Transformation suite parity table (legacy vs new).
- ✅ Testing pyramid, regression harness.
- ✅ CLI/API tooling and persistence adapters.
- ✅ Migration guide + sample templates.
- ✅ DevOps pipeline and deployment artifacts.

## 17. Open Questions
- Which Darts models constitute MVP coverage (e.g., `ExponentialSmoothing`, `ARIMA`, `Prophet`, `TiDE`, `NBEATS`)?
- Do we require GPU support from day one, or can it be optional adapter configuration?
- How to prioritize rarely used AutoTS transforms/ensembles—community-driven backlog?
- Should the new system support streaming/online updates immediately or defer to v2?

## 18. Next Steps
1. Review and refine domain vocabulary with stakeholders to ensure shared understanding.
2. Finalize MVP feature list and success metrics (accuracy tolerance, runtime budget).
3. Kick off Phase 0 tasks (repo scaffolding, CI setup, coding standards).
4. Schedule architecture review once domain and ports packages are stubbed with documentation.
