use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};

use anyhow::{Context as _, Result, ensure};
use futures::{StreamExt as _, stream::FuturesUnordered};
use koharu_scene::{EntityId, ProjectId, Snapshot};

use crate::{
    Committer, ErrorKind, PipelineError, Progress, ProgressSink, Report, Request, RunStatus, Stage,
    StageOutput, StopToken,
    admission::PageLease,
    images::ImageCache,
    progress,
    resources::ResourceMonitor,
    scheduler::{Plan, Remaining, Scheduler},
    scope::NormalizedScope,
    stage_runner::{StageCompletion, StageJob, StageOutcome, StageRunner, until_stopped},
    stages::StageInput,
};

pub(crate) struct Execution<'a> {
    runner: Arc<StageRunner>,
    resources: Arc<ResourceMonitor>,
    committer: &'a mut dyn Committer,
    stop: StopToken,
    progress: Option<ProgressSink>,
    scope: NormalizedScope,
    scheduler: Scheduler,
    scene: Snapshot,
    project: ProjectId,
    leases: BTreeMap<EntityId, PageLease>,
    images: BTreeMap<EntityId, Arc<ImageCache>>,
    busy_stages: BTreeSet<Stage>,
    completed: usize,
    failure: Option<PipelineError>,
    base: koharu_scene::Revision,
    started: Instant,
    inpainting_mask: Option<crate::InpaintingMask>,
}

impl<'a> Execution<'a> {
    pub(crate) async fn new(
        runner: Arc<StageRunner>,
        resources: Arc<ResourceMonitor>,
        snapshot: Snapshot,
        request: Request,
        committer: &'a mut dyn Committer,
    ) -> std::result::Result<Self, PipelineError> {
        let started = Instant::now();
        let base = snapshot.revision();
        let stages = request
            .operation
            .stages()
            .map_err(|error| PipelineError::new(ErrorKind::InvalidInput, None, error))?;
        let scope = NormalizedScope::new(&snapshot, &request.scope, &stages)
            .map_err(|error| PipelineError::new(ErrorKind::InvalidInput, None, error))?;
        let pages = scope.pages().to_vec();
        if let Some(mask) = request.inpainting_mask.as_ref()
            && (!pages.contains(&mask.page) || !stages.contains(&Stage::Inpainting))
        {
            return Err(PipelineError::new(
                ErrorKind::InvalidInput,
                Some(Stage::Inpainting),
                anyhow::anyhow!("the inpainting mask page is outside the inpainting scope"),
            ));
        }
        progress::emit(
            request.progress.as_ref(),
            Progress::Started {
                pages: pages.clone(),
                stages: stages.clone(),
            },
        );
        // Sizing translation batches can ask the provider, so a run without translation, which
        // forms no batch, does not. A stop during that lookup ends the run before it schedules
        // anything, so it does not wait for the provider's answer.
        let plan = if stages.contains(&Stage::Translation) {
            until_stopped(&request.stop, runner.stages().translation_plan())
                .await
                .unwrap_or(Plan::All)
        } else {
            Plan::All
        };

        Ok(Self {
            scheduler: Scheduler::new(&pages, &stages, plan),
            runner,
            resources,
            committer,
            stop: request.stop,
            progress: request.progress,
            scope,
            project: snapshot.project_id(),
            scene: snapshot,
            leases: BTreeMap::new(),
            images: BTreeMap::new(),
            busy_stages: BTreeSet::new(),
            completed: 0,
            failure: None,
            base,
            started,
            inpainting_mask: request.inpainting_mask,
        })
    }

    pub(crate) async fn run(mut self) -> std::result::Result<Report, PipelineError> {
        if self.stopped() {
            return Ok(self.report(RunStatus::Stopped));
        }

        self.resources.start();
        self.resources.wait_for_sample().await;

        let runner = self.runner.clone();
        let mut released = runner.admission().releases();
        let mut running = FuturesUnordered::new();
        loop {
            // A lease released after this point wakes the wait below.
            released.mark_unchanged();
            while let Some(job) = self.take_ready_job() {
                running.push(runner.run(job));
            }

            let leased = !self.stopped() && self.failure.is_none() && self.scheduler.blocked();
            let completion = if running.is_empty() {
                if !leased {
                    break;
                }
                tokio::select! {
                    _ = released.changed() => {}
                    () = self.stop.wait() => {}
                }
                continue;
            } else if leased {
                tokio::select! {
                    completion = running.next() => completion,
                    _ = released.changed() => continue,
                }
            } else {
                running.next().await
            };
            let Some(completion) = completion else {
                break;
            };
            self.busy_stages.remove(&completion.stage);
            if self.stopped() || self.failure.is_some() {
                continue;
            }
            if let Err(error) = self.apply_completion(completion).await {
                self.failure = Some(error);
            }
        }

        self.finalize()
    }

    fn take_ready_job(&mut self) -> Option<StageJob> {
        if self.stopped() || self.failure.is_some() {
            return None;
        }
        let admission = self.runner.admission();
        let (stage, pages) = self.scheduler.start_next(
            &self.busy_stages,
            |page| match admission.try_lease(self.project, page) {
                Some(lease) => {
                    self.leases.insert(page, lease);
                    true
                }
                None => false,
            },
            |page| admission.leased(self.project, page),
            |page| {
                let input = StageInput::new(
                    self.scene.clone(),
                    page,
                    self.scope.entities(),
                    self.scope.region(page),
                    Arc::default(),
                    None,
                );
                // A page the scene cannot describe fails in the translation job itself.
                self.runner
                    .stages()
                    .translation_weight(&input)
                    .unwrap_or_default()
            },
        )?;
        self.busy_stages.insert(stage);
        let inputs = pages
            .into_iter()
            .map(|page| {
                let images = if stage == Stage::Translation {
                    // Translation scales its own copy of each image, so the batch does not keep
                    // the pages' full-resolution decodes alive.
                    Arc::default()
                } else {
                    self.images.entry(page).or_default().clone()
                };
                StageInput::new(
                    self.scene.clone(),
                    page,
                    self.scope.entities(),
                    self.scope.region(page),
                    images,
                    self.inpainting_mask
                        .as_ref()
                        .filter(|mask| stage == Stage::Inpainting && mask.page == page)
                        .cloned(),
                )
            })
            .collect();
        Some(StageJob::new(
            stage,
            inputs,
            self.stop.clone(),
            self.progress.clone(),
        ))
    }

    async fn apply_completion(
        &mut self,
        completion: StageCompletion,
    ) -> std::result::Result<(), PipelineError> {
        let StageCompletion {
            stage,
            pages,
            model,
            elapsed,
            outcome,
        } = completion;
        for (page, outcome) in pages.into_iter().zip(outcome?) {
            if self.stopped() {
                break;
            }
            match outcome {
                StageOutcome::Stopped => {}
                StageOutcome::Skipped => {
                    self.mark_complete(page, stage);
                    progress::emit(self.progress.as_ref(), Progress::Skipped { page, stage });
                }
                StageOutcome::Patch(patch) => {
                    let event = match self.commit_patch(page, stage, patch).await? {
                        Commit::Committed => Progress::Finished {
                            page,
                            stage,
                            model: model.clone(),
                            elapsed,
                        },
                        Commit::Superseded => Progress::Skipped { page, stage },
                        Commit::Stopped => return Ok(()),
                    };
                    self.mark_complete(page, stage);
                    progress::emit(self.progress.as_ref(), event);
                }
            }
        }
        Ok(())
    }

    async fn commit_patch(
        &mut self,
        page: EntityId,
        stage: Stage,
        patch: koharu_scene::Patch,
    ) -> std::result::Result<Commit, PipelineError> {
        let rebased = patch.rebase_on(&self.scene).and_then(|patch| {
            patch.validate_on(&self.scene)?;
            Ok(patch.with_label(format!("Pipeline {stage} for page {page}")))
        });
        // An edit that changed what the output observed supersedes this page's output only, as
        // it does when the committer's own rebase finds the edit.
        if let Err(
            error @ (koharu_scene::Error::PatchConflict(_)
            | koharu_scene::Error::EntityNotFound(_)
            | koharu_scene::Error::RelationNotFound(_)),
        ) = &rebased
        {
            tracing::debug!(%page, %stage, %error, "stage output was superseded by an edit");
            return Ok(Commit::Superseded);
        }
        let patch = rebased
            .context("failed to rebase stage output onto the latest scene")
            .map_err(|error| PipelineError::new(ErrorKind::InvalidOutput, Some(stage), error))?;
        if self.stopped() {
            return Ok(Commit::Stopped);
        }

        let next = self
            .committer
            .commit(StageOutput { page, stage, patch })
            .await
            .with_context(|| format!("failed to commit {stage} output for page {page}"))
            .map_err(|error| PipelineError::new(ErrorKind::Commit, Some(stage), error))?;
        validate_commit(&self.scene, &next)
            .map_err(|error| PipelineError::new(ErrorKind::Commit, Some(stage), error))?;
        self.scene = next;
        Ok(Commit::Committed)
    }

    fn mark_complete(&mut self, page: EntityId, stage: Stage) {
        match self.scheduler.complete_stage(page, stage) {
            Remaining::Local => {}
            Remaining::Translation => {
                self.images.remove(&page);
            }
            Remaining::Nothing => {
                self.images.remove(&page);
                self.leases.remove(&page);
            }
        }
        self.completed += 1;
    }

    fn stopped(&self) -> bool {
        self.stop.stopped()
    }

    fn finalize(mut self) -> std::result::Result<Report, PipelineError> {
        if let Some(error) = self.failure.take() {
            return Err(error);
        }
        if !self.stopped() && self.completed != self.scheduler.total() {
            return Err(PipelineError::new(
                ErrorKind::InvalidOutput,
                None,
                anyhow::anyhow!(
                    "pipeline scheduler stopped after {} of {} work items",
                    self.completed,
                    self.scheduler.total()
                ),
            ));
        }
        let status = if self.stopped() {
            RunStatus::Stopped
        } else {
            RunStatus::Completed
        };
        Ok(self.report(status))
    }

    fn report(&self, status: RunStatus) -> Report {
        Report {
            status,
            base: self.base,
            final_revision: self.scene.revision(),
            completed: self.completed,
            total: self.scheduler.total(),
            elapsed: self.started.elapsed(),
        }
    }
}

/// What became of one page's stage output.
enum Commit {
    Committed,
    /// A document edit changed what the output observed, so it was dropped.
    Superseded,
    /// The run stopped first, so the output was discarded.
    Stopped,
}

fn validate_commit(previous: &Snapshot, next: &Snapshot) -> Result<()> {
    ensure!(
        previous.project_id() == next.project_id(),
        "committer returned a snapshot from another project"
    );
    ensure!(
        next.revision() > previous.revision(),
        "committer did not advance the scene revision"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{sync::Mutex, time::Duration};

    use koharu_scene::{At, Authored, PageDraft, Session, SourceText, Translation};

    use super::*;
    use crate::{Operation, PipelineConfig, admission::Admission};

    /// Commits to a session as the application does, and records the pages it committed.
    struct SessionCommitter {
        session: Session,
        pages: Vec<EntityId>,
    }

    #[async_trait::async_trait]
    impl Committer for SessionCommitter {
        async fn commit(&mut self, output: StageOutput) -> Result<Snapshot> {
            self.pages.push(output.page);
            self.session.commit(output.patch).await?;
            Ok(self.session.snapshot())
        }
    }

    fn source(text: &str) -> SourceText {
        SourceText {
            text: Authored::user(text.to_owned()),
            language: None,
        }
    }

    #[tokio::test]
    async fn an_edit_that_supersedes_one_page_of_a_batch_skips_only_that_page() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let (mut pages, mut texts) = (Vec::new(), Vec::new());
        for name in ["one", "two"] {
            let page = setup
                .add_page(PageDraft::new(name, 1.0, 1.0), At::End)
                .unwrap();
            let text = setup.add_text_content(page, At::End).unwrap();
            setup.set(text, &source(name)).unwrap();
            pages.push(page);
            texts.push(text);
        }
        session.commit(setup.finish().unwrap()).await.unwrap();
        // The batch translates both pages from the scene it started on.
        let base = session.snapshot();
        let translated = Translation {
            text: Authored::user("translated".to_owned()),
            language: None,
        };
        let outcomes = texts
            .iter()
            .map(|&text| {
                let mut edit = base.edit();
                edit.observe::<SourceText>(text).unwrap();
                edit.set(text, &translated).unwrap();
                StageOutcome::Patch(edit.finish().unwrap())
            })
            .collect();
        // Meanwhile the user corrects the first page's source text.
        let edit = base
            .patch(|edit| edit.set(texts[0], &source("edited")))
            .unwrap();
        session.commit(edit).await.unwrap();

        let device = koharu_ml::Device::cpu();
        let resources = ResourceMonitor::new(&device);
        let translator = koharu_translator::Translator::from_config(
            device.clone(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        let runner = StageRunner::new(
            &PipelineConfig::default(),
            translator,
            &device,
            Arc::new(Admission::new(&device, resources.clone())),
            None,
        )
        .unwrap();
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        let snapshot = session.snapshot();
        let mut committer = SessionCommitter {
            session,
            pages: Vec::new(),
        };
        let mut execution = Execution::new(
            Arc::new(runner),
            resources,
            snapshot,
            Request {
                operation: Operation::Only {
                    stage: Stage::Translation,
                },
                progress: Some(Arc::new(move |event| sink.lock().unwrap().push(event))),
                ..Request::default()
            },
            &mut committer,
        )
        .await
        .unwrap();
        let completion = |pages, outcomes| StageCompletion {
            stage: Stage::Translation,
            pages,
            model: "test".to_owned(),
            elapsed: Duration::ZERO,
            outcome: Ok(outcomes),
        };

        execution
            .apply_completion(completion(pages.clone(), outcomes))
            .await
            .unwrap();
        assert_eq!(execution.completed, 2);
        // Output that cannot rebase for another reason still fails the run.
        let foreign = Session::memory()
            .await
            .unwrap()
            .snapshot()
            .patch(|edit| {
                edit.add_page(PageDraft::new("other", 1.0, 1.0), At::End)?;
                Ok(())
            })
            .unwrap();
        let foreign = completion(vec![EntityId::new()], vec![StageOutcome::Patch(foreign)]);
        let error = execution.apply_completion(foreign).await.unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidOutput);
        drop(execution);

        assert_eq!(committer.pages, [pages[1]]);
        let finished = events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|event| match event {
                Progress::Skipped { page, .. } => Some((*page, false)),
                Progress::Finished { page, .. } => Some((*page, true)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(finished, [(pages[0], false), (pages[1], true)]);
        let scene = committer.session.snapshot();
        assert_eq!(
            scene.component::<SourceText>(texts[0]).unwrap(),
            Some(source("edited"))
        );
        assert!(scene.component::<Translation>(texts[0]).unwrap().is_none());
        assert!(scene.component::<Translation>(texts[1]).unwrap().is_some());
    }
}
