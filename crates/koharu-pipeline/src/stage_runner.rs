use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Result;
use koharu_scene::{EntityId, Patch};

use crate::{
    ErrorKind, PipelineConfig, PipelineError, Progress, ProgressSink, Stage, StopToken,
    admission::Admission,
    progress,
    stages::{StageInput, Stages},
};

pub(crate) struct StageRunner {
    stages: Arc<Stages>,
    admission: Arc<Admission>,
}

impl StageRunner {
    pub(crate) fn new(
        config: &PipelineConfig,
        translator: koharu_translator::Translator,
        device: &koharu_ml::Device,
        admission: Arc<Admission>,
        previous: Option<&Self>,
    ) -> Result<Self> {
        let stages = Arc::new(Stages::new(
            config,
            translator,
            device,
            previous.map(|runner| runner.stages.as_ref()),
        )?);
        admission.register(&stages);
        Ok(Self { stages, admission })
    }

    pub(crate) fn stages(&self) -> &Stages {
        &self.stages
    }

    pub(crate) fn admission(&self) -> &Arc<Admission> {
        &self.admission
    }

    #[tracing::instrument(skip_all)]
    pub(crate) async fn run(&self, job: StageJob) -> StageCompletion {
        let started = Instant::now();
        let pages = job.inputs.iter().map(StageInput::page).collect::<Vec<_>>();
        let model = self.stages.model(job.stage).to_owned();
        let outcome = self.run_with_recovery(&job, &model).await;
        StageCompletion {
            stage: job.stage,
            pages,
            model,
            elapsed: started.elapsed(),
            outcome,
        }
    }

    async fn run_with_recovery(
        &self,
        job: &StageJob,
        model: &str,
    ) -> std::result::Result<Vec<StageOutcome>, PipelineError> {
        if job.stop.stopped() {
            return Ok(vec![StageOutcome::Stopped; job.inputs.len()]);
        }
        let mut skipped = Vec::with_capacity(job.inputs.len());
        for input in &job.inputs {
            let skip = self.stages.skip(job.stage, input).map_err(|error| {
                self.stage_error(
                    job.stage,
                    model,
                    AttemptFailure {
                        kind: ErrorKind::Processing,
                        error,
                    },
                )
            })?;
            skipped.push(skip);
        }
        let inputs = job
            .inputs
            .iter()
            .zip(&skipped)
            .filter(|(_, skipped)| !**skipped)
            .map(|(input, _)| input.clone())
            .collect::<Vec<_>>();
        if inputs.is_empty() {
            return Ok(vec![StageOutcome::Skipped; job.inputs.len()]);
        }

        // Lock order: stage, then accelerator lane, then the stage's model cell. A stop ends
        // every wait at once; until then a queued acquire keeps its place.
        let Some(_stage) = until_stopped(&job.stop, self.admission.stage(job.stage)).await else {
            return Ok(vec![StageOutcome::Stopped; job.inputs.len()]);
        };
        let accelerated = self.stages.accelerated(job.stage);
        let lane = if accelerated {
            let Some(lane) = until_stopped(&job.stop, self.admission.accelerator().acquire()).await
            else {
                return Ok(vec![StageOutcome::Stopped; job.inputs.len()]);
            };
            Some(lane)
        } else {
            None
        };
        let first = if accelerated {
            self.load_and_process(job, &inputs, model).await
        } else {
            // Work off the lane leaves no device busy, so a stop abandons it in flight.
            let Some(first) =
                until_stopped(&job.stop, self.load_and_process(job, &inputs, model)).await
            else {
                return Ok(vec![StageOutcome::Stopped; job.inputs.len()]);
            };
            first
        };
        let failure = match first {
            Ok(outcomes) => return Ok(merge(&skipped, outcomes)),
            Err(failure)
                if accelerated && is_out_of_memory(&failure.error) && !job.stop.stopped() =>
            {
                failure
            }
            Err(failure) => return Err(self.stage_error(job.stage, model, failure)),
        };

        drop(lane);
        tracing::warn!(stage = %job.stage, pages = inputs.len(), error = %failure.error, "retrying stage after memory pressure");
        let Some(_lane) =
            until_stopped(&job.stop, self.admission.recover(job.stage, &self.stages)).await
        else {
            return Ok(vec![StageOutcome::Stopped; job.inputs.len()]);
        };
        match self.load_and_process(job, &inputs, model).await {
            Ok(outcomes) => Ok(merge(&skipped, outcomes)),
            Err(failure) => Err(self.stage_error(job.stage, model, failure)),
        }
    }

    async fn load_and_process(
        &self,
        job: &StageJob,
        inputs: &[StageInput],
        model: &str,
    ) -> std::result::Result<Vec<StageOutcome>, AttemptFailure> {
        for input in inputs {
            progress::emit(
                job.progress.as_ref(),
                Progress::Loading {
                    page: input.page(),
                    stage: job.stage,
                    model: model.to_owned(),
                },
            );
        }
        self.stages
            .load(job.stage)
            .await
            .map_err(|error| AttemptFailure {
                kind: ErrorKind::ModelLoad,
                error,
            })?;
        if job.stop.stopped() {
            return Ok(vec![StageOutcome::Stopped; inputs.len()]);
        }
        for input in inputs {
            progress::emit(
                job.progress.as_ref(),
                Progress::Running {
                    page: input.page(),
                    stage: job.stage,
                    model: model.to_owned(),
                },
            );
        }
        self.stages
            .process(job.stage, inputs.to_vec(), &job.stop)
            .await
            .and_then(|patches| {
                anyhow::ensure!(
                    patches.len() == inputs.len(),
                    "{} returned {} patches for {} pages",
                    job.stage,
                    patches.len(),
                    inputs.len()
                );
                Ok(patches
                    .into_iter()
                    .map(|patch| {
                        if patch.is_empty() {
                            StageOutcome::Skipped
                        } else {
                            StageOutcome::Patch(patch)
                        }
                    })
                    .collect())
            })
            .map_err(|error| AttemptFailure {
                kind: ErrorKind::Processing,
                error,
            })
    }

    fn stage_error(&self, stage: Stage, model: &str, failure: AttemptFailure) -> PipelineError {
        self.stages.unload(stage);
        let message = match failure.kind {
            ErrorKind::ModelLoad => format!("failed to load {model}"),
            _ => format!("{model} failed"),
        };
        PipelineError::new(failure.kind, Some(stage), failure.error.context(message))
    }
}

/// Awaits `wait`, or returns `None` once `stop` is requested first. `wait` stays pinned in
/// place until then, so a queued acquire keeps its turn.
pub(crate) async fn until_stopped<T>(stop: &StopToken, wait: impl Future<Output = T>) -> Option<T> {
    tokio::select! {
        biased;
        () = stop.wait() => None,
        value = wait => Some(value),
    }
}

/// Places the outcomes of the processed inputs between the skipped ones, in job order.
fn merge(skipped: &[bool], processed: Vec<StageOutcome>) -> Vec<StageOutcome> {
    let mut processed = processed.into_iter();
    skipped
        .iter()
        .map(|skipped| {
            if *skipped {
                StageOutcome::Skipped
            } else {
                processed
                    .next()
                    .expect("a stage returns one outcome per processed input")
            }
        })
        .collect()
}

struct AttemptFailure {
    kind: ErrorKind,
    error: anyhow::Error,
}

fn is_out_of_memory(error: &anyhow::Error) -> bool {
    // llama.cpp reports a KV cache it cannot allocate as a context it failed to create.
    error.chain().any(|source| {
        let message = source.to_string().to_ascii_lowercase();
        message.contains("out of memory")
            || message.contains("cuda_error_out_of_memory")
            || message.contains("not enough memory")
            || message.contains("failed to create llama.cpp context")
    })
}

pub(crate) struct StageJob {
    stage: Stage,
    inputs: Vec<StageInput>,
    stop: StopToken,
    progress: Option<ProgressSink>,
}

impl StageJob {
    pub(crate) fn new(
        stage: Stage,
        inputs: Vec<StageInput>,
        stop: StopToken,
        progress: Option<ProgressSink>,
    ) -> Self {
        Self {
            stage,
            inputs,
            stop,
            progress,
        }
    }
}

#[derive(Clone)]
pub(crate) enum StageOutcome {
    Patch(Patch),
    Skipped,
    Stopped,
}

pub(crate) struct StageCompletion {
    pub(crate) stage: Stage,
    pub(crate) pages: Vec<EntityId>,
    pub(crate) model: String,
    pub(crate) elapsed: Duration,
    /// One outcome per page, in page order.
    pub(crate) outcome: std::result::Result<Vec<StageOutcome>, PipelineError>,
}
