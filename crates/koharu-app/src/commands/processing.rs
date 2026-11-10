use std::{collections::HashMap, error::Error as _, fmt, path::PathBuf, sync::Arc};

use anyhow::{Context as _, Result};
use koharu_pipeline::{
    Committer, ErrorKind, InpaintingMask, Pipeline, PipelineError, Progress, Report, RunStatus,
    StageOutput, StopToken,
};
use koharu_scene::Snapshot;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::{Manager as _, WebviewWindow};
use tauri_runtime_cef::CefRuntime;
use tokio::{sync::Notify, task::JoinHandle};
use uuid::Uuid;

use super::{ChannelExt as _, Error, workspace::Workspace};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize, Type)]
#[serde(transparent)]
pub struct JobId(Uuid);

impl JobId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct Job {
    pub id: JobId,
    pub state: JobState,
    #[specta(type = f64)]
    pub completed: usize,
    #[specta(type = f64)]
    pub total: usize,
    pub page: Option<koharu_scene::EntityId>,
    pub stage: Option<koharu_pipeline::Stage>,
    pub model: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Running,
    Finished,
    Failed,
    Stopped,
}

/// One workspace's running jobs.
#[derive(Default)]
pub(crate) struct Processing {
    stops: Mutex<HashMap<JobId, StopToken>>,
    jobs: Mutex<HashMap<JobId, Job>>,
    finished: Notify,
}

impl Processing {
    pub(crate) fn jobs(&self) -> Vec<Job> {
        self.jobs.lock().values().cloned().collect()
    }

    pub(crate) fn is_running(&self) -> bool {
        !self.stops.lock().is_empty()
    }

    pub(crate) fn stop(&self, job: JobId) -> Result<()> {
        self.stops
            .lock()
            .get(&job)
            .with_context(|| format!("job {job} is not running"))?
            .stop();
        Ok(())
    }

    /// Stops every running job and returns their ids.
    pub(crate) fn stop_all(&self) -> Vec<JobId> {
        let stops = self.stops.lock();
        for stop in stops.values() {
            stop.stop();
        }
        stops.keys().copied().collect()
    }

    pub(crate) async fn wait(&self, jobs: &[JobId]) {
        loop {
            let finished = self.finished.notified();
            let running = {
                let stops = self.stops.lock();
                jobs.iter().any(|job| stops.contains_key(job))
            };
            if !running {
                return;
            }
            finished.await;
        }
    }
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn process(
    scope: koharu_pipeline::Scope,
    operation: koharu_pipeline::Operation,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<JobId, Error> {
    let (job, _) = start_job(&Workspace::of(&window)?, scope, operation, None).await?;
    Ok(job)
}

/// Starts a job on the workspace's open project and commits its results into that project
/// only. The returned handle resolves after the job no longer holds project snapshots.
pub(crate) async fn start_job(
    workspace: &Arc<Workspace>,
    scope: koharu_pipeline::Scope,
    operation: koharu_pipeline::Operation,
    inpainting_mask: Option<InpaintingMask>,
) -> Result<(JobId, JoinHandle<Result<Report, PipelineError>>)> {
    let pipeline = workspace.handle.state::<Pipeline>().inner().clone();
    let id = JobId::new();
    let stop = StopToken::default();
    let (snapshot, binding) = {
        let project = workspace.project.lock().await;
        let snapshot = project.as_ref().context("no project is open")?.snapshot();
        let job = Job {
            id,
            state: JobState::Running,
            completed: 0,
            total: 0,
            page: None,
            stage: None,
            model: None,
            error: None,
        };
        // Registering under the project lock means a project switch either stops this job or
        // hands it the next project.
        workspace.processing.stops.lock().insert(id, stop.clone());
        workspace.processing.jobs.lock().insert(id, job.clone());
        workspace.channels.job.publish(job);
        (snapshot, workspace.binding())
    };

    let task_workspace = Arc::clone(workspace);
    let outcome = tokio::spawn(async move {
        let _registration = Registration {
            workspace: Arc::clone(&task_workspace),
            job: id,
        };
        let progress = Arc::new(Mutex::new((0_usize, 0_usize)));
        let progress_workspace = Arc::clone(&task_workspace);
        let mut request = koharu_pipeline::Request {
            operation,
            scope,
            stop: stop.clone(),
            progress: None,
            inpainting_mask,
        };
        request.progress = Some(Arc::new(move |event| {
            let update = match event {
                Progress::Started { pages, stages } => {
                    let mut progress = progress.lock();
                    *progress = (0, pages.len().saturating_mul(stages.len()));
                    Some((0, progress.1, None, None, None))
                }
                Progress::Loading { page, stage, model } => {
                    let progress = progress.lock();
                    Some((progress.0, progress.1, Some(page), Some(stage), Some(model)))
                }
                Progress::Finished {
                    page, stage, model, ..
                } => {
                    let mut progress = progress.lock();
                    progress.0 = progress.0.saturating_add(1).min(progress.1);
                    Some((progress.0, progress.1, Some(page), Some(stage), Some(model)))
                }
                Progress::Skipped { page, stage } => {
                    let mut progress = progress.lock();
                    progress.0 = progress.0.saturating_add(1).min(progress.1);
                    Some((progress.0, progress.1, Some(page), Some(stage), None))
                }
                Progress::Running { .. } => None,
            };
            if let Some((completed, total, page, stage, model)) = update {
                let job = {
                    let mut jobs = progress_workspace.processing.jobs.lock();
                    jobs.get_mut(&id).map(|job| {
                        job.completed = completed;
                        job.total = total;
                        job.page = page;
                        job.stage = stage;
                        job.model = model;
                        job.clone()
                    })
                };
                if let Some(job) = job {
                    progress_workspace.channels.job.publish(job);
                }
            }
        }));

        struct PipelineCommitter {
            workspace: Arc<Workspace>,
            binding: Option<PathBuf>,
        }

        #[async_trait::async_trait]
        impl Committer for PipelineCommitter {
            async fn commit(&mut self, output: StageOutput) -> Result<Snapshot> {
                let (commit, page) = {
                    let mut project = self.workspace.project.lock().await;
                    let project = project
                        .as_mut()
                        .filter(|_| self.workspace.binding() == self.binding)
                        .ok_or(ProjectClosed)?;
                    let Some(commit) = project.commit_rebased(output.patch).await? else {
                        return Ok(project.snapshot());
                    };
                    project.record_commit(&commit);
                    let page = project.active_page();
                    (commit, page)
                };
                let snapshot = commit.snapshot.clone();
                self.workspace.synchronize(&commit, page).await?;
                Ok(snapshot)
            }
        }

        let mut committer = PipelineCommitter {
            workspace: Arc::clone(&task_workspace),
            binding,
        };
        let result = pipeline.execute(snapshot, request, &mut committer).await;
        // A failed run counts as stopped only when the switch or close that stopped it refused
        // its commit; any other failure is reported, also when a stop followed it.
        let stopped = match &result {
            Ok(report) => report.status == RunStatus::Stopped,
            Err(error) => {
                stop.stopped()
                    && error.kind == ErrorKind::Commit
                    && std::iter::successors(error.source(), |&cause| cause.source())
                        .any(|cause| cause.is::<ProjectClosed>())
            }
        };
        let error = match &result {
            Ok(_) => None,
            Err(error) if stopped => {
                tracing::debug!(stage = ?error.stage, %error, "a stopped job's commit was refused");
                None
            }
            Err(error) => {
                tracing::error!(stage = ?error.stage, %error, "processing failed");
                Some(format!("{error:#}"))
            }
        };
        task_workspace.processing.stops.lock().remove(&id);
        let job = task_workspace
            .processing
            .jobs
            .lock()
            .remove(&id)
            .map(|mut job| {
                job.state = if stopped {
                    JobState::Stopped
                } else if error.is_some() {
                    JobState::Failed
                } else {
                    JobState::Finished
                };
                job.error = error;
                job
            });
        if let Some(job) = job {
            task_workspace.channels.job.publish(job);
        }
        result
    });
    Ok((id, outcome))
}

/// The committer's refusal once the job's project was switched or closed. A switch or close
/// stops the job before the committer can refuse its commit, so this error ends a stopped job.
#[derive(Debug)]
struct ProjectClosed;

impl fmt::Display for ProjectClosed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("the job's project is no longer open")
    }
}

impl std::error::Error for ProjectClosed {}

/// Unregisters a job when its task ends, even by panic, so waiting for the job always returns.
struct Registration {
    workspace: Arc<Workspace>,
    job: JobId,
}

impl Drop for Registration {
    fn drop(&mut self) {
        let processing = &self.workspace.processing;
        processing.stops.lock().remove(&self.job);
        let job = processing.jobs.lock().remove(&self.job);
        if let Some(mut job) = job {
            job.state = JobState::Failed;
            job.error = Some("processing stopped unexpectedly".to_owned());
            self.workspace.channels.job.publish(job);
        }
        processing.finished.notify_waiters();
    }
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn stop_job(
    job: JobId,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<(), Error> {
    Ok(Workspace::of(&window)?.processing.stop(job)?)
}
