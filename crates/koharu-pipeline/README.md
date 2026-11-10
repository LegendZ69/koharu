# koharu-pipeline

`koharu-pipeline` coordinates detection, OCR, translation, and inpainting over a
`koharu_scene` project. It owns model lifetime and scheduling; the caller owns
durability through the `Committer` trait.

The execution unit is one stage on one page; translation runs on a batch of
pages and still commits each page separately. A stage result is committed as
soon as it finishes, so the application can refresh that page immediately and
retain completed work if the user stops the run or a later model fails.

## Execution

```rust,ignore
let report = pipeline
    .execute(
        session.snapshot(),
        Request {
            operation: Operation::Full,
            scope: Scope::Project,
            stop: stop.clone(),
            progress: Some(progress),
        },
        &mut committer,
    )
    .await?;

match report.status {
    RunStatus::Completed => println!("all page stages finished"),
    RunStatus::Stopped => println!("completed commits were kept"),
}
```

`Committer::commit` receives a `StageOutput` and returns the committed
`Snapshot`. Pipeline outputs are optimistically rebased onto the latest
snapshot before the callback. Stage patches observe the exact hierarchy,
components, and assets they consume. OCR, translation, and inpainting branches
on one page therefore compose, while a changed input or overlapping write fails
conflict validation instead of publishing stale derived output. When this rebase
finds that an edit superseded a page's output, only that output is dropped: the
page reports `Skipped`, and the rest of its batch and of the run continue.

The application records all revisions from one invocation as one undo group.

## Configuration

`Pipeline::load` reads `PipelineConfig` itself. The pipeline subscribes to the
shared configuration and publishes a new immutable stage generation through
`ArcSwap` whenever that section changes. An execution keeps the generation it
started with, while later executions see the replacement. Models remain lazy.

The public pipeline constructor accepts provider configuration rather than a
prebuilt translator. Processor construction is a pipeline responsibility;
translation does not require a separate application-owned runtime path.

Translation is a normal pipeline processor. Its model selection, target
language, instructions, batching, and generation options live under
`[pipeline.translation]` and are captured in the same immutable runner
generation as detection, OCR, and inpainting. Provider connection settings are
owned by `koharu-translator` under `[providers]`.

The translation processor delegates provider execution and local-model
residency to `koharu_translator::Translator`, but the translator does not load
or watch workflow configuration. Each result replaces the `Translation`
component of a semantic `TextContent` entity; the selected target language is
stored as metadata on that value. The related analysis region and presentation
layer remain independently editable entities.

## Fixed workflow

The workflow is small and explicit:

```text
detection -> OCR -> translation
         \-> inpainting
```

There is no runtime graph or graph library. `Operation` selects a subset of
this workflow:

- `Full`
- `Through(Stage)`
- `Only(Stage)`

For each page, a stage starts only after its selected prerequisite commits.
Pages enter in project order and the scheduler always selects the oldest ready
page; there is no global wave barrier. On an accelerator, ready jobs share one
execution lane while loaded models remain resident across pages; remote
translation does not take the lane. Benchmarks on the target CUDA workload
showed that overlapping heterogeneous model pairs increased makespan by 2.5-4x
through kernel and memory-bandwidth contention. Serial model execution
therefore produces more pages per second than maximizing the activity
percentage reported by the GPU.

The readiness window still matters: completed detection immediately exposes
both of that page's branches, commits remain page-local, and the next best job
can start without waiting for an unrelated page to finish. CPU-only execution
retains independent per-model lanes because it does not use the accelerator
gate.

```text
ready event               accelerator lane
page 1 enters             detection page 1
detection commits         OCR page 1
page 1 image branch ready inpainting page 1
page 2 window enters      detection page 2
page 1 OCR commits        translation page 1
```

Page priority prevents an upstream model from racing arbitrarily far ahead,
while independent page branches use otherwise idle models. The number of
active pages is bounded by the selected stage count, so the scheduler keeps a
small rolling window instead of filling an unbounded upstream queue. A page
whose only remaining stage is translation is parked outside the window, so a
translation batch can grow larger than the window.

Translation takes its batch, in project order, from `batching`: `all` sends
every page of the run in one request, `halving` does the same but splits a
request that is too large into halves down to single pages, `custom` sends
`pages_per_request` pages per request, and `auto` fills a request up to the
model's estimated budget and halves it when it is still too large. A batch
starts once its pages finished OCR. With `halving` and `auto`, the pages a
multi-page response left incomplete are sent again in smaller requests.

Several requests may run at once, from one or several projects. They share
the accelerator lane and one admission per stage, so at most one job per stage
runs at a time across all runs. A page is processed by one run of its project
at a time: another run skips that page, also when it forms translation batches,
continues with later pages, and waits for the page once nothing else can start.

Stages return an empty patch when their page has no applicable input. A page
with no detected text skips OCR and translation work; an empty removal mask
skips inpainting inference. These are normal completed work items, not
preflight failures.

## Stop semantics

`StopToken` is cooperative scheduling control, not transactional cancellation.

- `stop()` prevents new work from starting.
- A job waiting for a stage admission or the accelerator lane leaves the queue
  at once, and a remote translation request in flight is abandoned.
- Accelerated work already in progress finishes its current step, such as a
  model load, a page's inference, or one local translation request.
- Its result is discarded if stop was requested before commit.
- Every earlier commit remains in the project.
- `execute` returns `Ok(Report { status: RunStatus::Stopped, .. })`.

Errors are reserved for invalid input/output, model load or inference failure,
and commit failure. An error also leaves earlier page-stage commits intact.

## Model residency

Models are loaded lazily and remain resident for reuse across pages. A new
configuration generation keeps the processors of stages whose settings did not
change, so runs on either generation share their models. Normal execution
never unloads a model based on recent usage or sampled memory.
If a stage reports an out-of-memory failure, the accelerator gate unloads every
other model of every live generation, allows the device to settle, and retries
that stage once. The requested model remains loaded when possible.

The resource monitor samples the selected accelerator ten times per second for
UI telemetry. On Windows, DXGI supplies the process-aware memory budget while
NVML supplies NVIDIA compute utilization. Telemetry does not control model
lifetime.

`ModelCell` serializes access to one model and prevents an active model from
being unloaded.

## Stages

`stages/mod.rs` defines `StageLifecycle`. Each stage owns the same small
lifecycle contract:

- identify its model;
- load lazily;
- unload during explicit memory-pressure recovery;
- state whether its work takes the accelerator lane.

Detection, OCR, and inpainting process exactly one page and produce a semantic
`Patch`. Translation processes a batch of pages and produces one `Patch` per
page.

`StageInput` carries one page ID plus optional entity and region filters for
that page. Only translation receives several inputs; no stage receives a
project. Stage implementations own their preprocessing, inference, and scene
mapping. There is no shared `common.rs`, stage registry indexing, or catch-all
processor enum. `Stages` has named fields and one exhaustive dispatch point.

## Progress

Progress events carry both page and stage:

- `Started { pages, stages }`
- `Loading { page, stage, model }`
- `Running { page, stage, model }`
- `Finished { page, stage, model, elapsed }`
- `Skipped { page, stage }`

`Finished` is emitted after the stage output commits. Progress callbacks are
isolated from panics and cannot crash execution.

## Source map

```text
src/
  accelerator.rs  serialized accelerator execution and OOM recovery
  admission.rs    stage, accelerator, and page admission shared by all runs
  config.rs       model selection and serialized settings
  error.rs        typed execution failures
  execution.rs    one request's scheduling, commits, progress, and report
  images.rs       decoded scene-asset reuse within each active page
  model_cell.rs   lazy model ownership and per-model serialization
  pipeline.rs     configuration generations and public execution entry point
  progress.rs     request-local progress events
  report.rs       run outcome, stage output, and Committer
  request.rs      operation, scope, and StopToken
  resources.rs    UI-facing accelerator and host-memory telemetry
  scheduler.rs    page window, stage readiness, and translation batches
  scope.rs        validated project/page/region/entity scope
  stage.rs        stable stage identifiers
  stage_runner.rs model loading, processing, progress, and retry classification
  stages/         detection, OCR, translation, and inpainting processors
```

## Validation

```powershell
cargo test -p koharu-pipeline --all-targets
cargo clippy -p koharu-pipeline --all-targets --no-deps -- -D warnings
cargo bench --profile dev -p koharu-ml --bench pipeline_overlap
```
