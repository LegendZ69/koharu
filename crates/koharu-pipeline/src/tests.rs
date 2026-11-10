use super::*;

#[test]
fn configuration_ignores_unknown_fields() {
    let config = toml::from_str::<PipelineConfig>("legacy_limit = 1").unwrap();
    assert_eq!(config, PipelineConfig::default());
}

#[tokio::test]
async fn stop_is_a_successful_partial_result() {
    let pipeline = pipeline(Default::default());
    let stop = StopToken::default();
    stop.stop();
    let request = Request {
        stop,
        ..Request::default()
    };
    let mut committer = RejectCommitter;

    let report = pipeline
        .execute(
            koharu_scene::Session::memory().await.unwrap().snapshot(),
            request,
            &mut committer,
        )
        .await
        .unwrap();

    assert_eq!(report.status, RunStatus::Stopped);
    assert_eq!(report.completed, 0);
}

#[tokio::test]
async fn stop_after_a_page_keeps_completed_progress() {
    let translation = TranslationConfig {
        model: koharu_translator::ModelSelection {
            provider: koharu_translator::Provider::OpenAi,
            model: Some("gpt-5.6-luna".to_owned()),
            quantization: None,
            vision: true,
            reasoning: true,
        },
        ..Default::default()
    };
    let pipeline = pipeline(translation);
    let mut session = koharu_scene::Session::memory().await.unwrap();
    let patch = session
        .snapshot()
        .patch(|edit| {
            edit.add_page(
                koharu_scene::PageDraft::new("one", 1.0, 1.0),
                koharu_scene::At::End,
            )?;
            edit.add_page(
                koharu_scene::PageDraft::new("two", 1.0, 1.0),
                koharu_scene::At::End,
            )?;
            Ok(())
        })
        .unwrap();
    session.commit(patch).await.unwrap();
    let stop = StopToken::default();
    let progress_stop = stop.clone();
    let request = Request {
        operation: Operation::Only {
            stage: Stage::Translation,
        },
        stop,
        progress: Some(std::sync::Arc::new(move |event| {
            if matches!(event, Progress::Skipped { .. }) {
                progress_stop.stop();
            }
        })),
        ..Request::default()
    };
    let mut committer = RejectCommitter;

    let report = pipeline
        .execute(session.snapshot(), request, &mut committer)
        .await
        .unwrap();

    assert_eq!(report.status, RunStatus::Stopped);
    assert_eq!(report.completed, 1);
    assert_eq!(report.total, 2);
}

#[tokio::test]
async fn a_page_leased_by_another_run_is_processed_after_its_release() {
    let device = koharu_ml::Device::cpu();
    let resources = resources::ResourceMonitor::new(&device);
    let admission = std::sync::Arc::new(admission::Admission::new(&device, resources.clone()));
    let config = PipelineConfig {
        translation: TranslationConfig {
            model: koharu_translator::ModelSelection {
                provider: koharu_translator::Provider::OpenAi,
                model: Some("gpt-5.6-luna".to_owned()),
                quantization: None,
                vision: true,
                reasoning: true,
            },
            ..Default::default()
        },
        ..PipelineConfig::default()
    };
    let translator = koharu_translator::Translator::from_config(
        device.clone(),
        koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
    )
    .unwrap();
    let runner = std::sync::Arc::new(
        stage_runner::StageRunner::new(&config, translator, &device, admission.clone(), None)
            .unwrap(),
    );
    let mut session = koharu_scene::Session::memory().await.unwrap();
    let mut first = None;
    let patch = session
        .snapshot()
        .patch(|edit| {
            first = Some(edit.add_page(
                koharu_scene::PageDraft::new("one", 1.0, 1.0),
                koharu_scene::At::End,
            )?);
            edit.add_page(
                koharu_scene::PageDraft::new("two", 1.0, 1.0),
                koharu_scene::At::End,
            )?;
            Ok(())
        })
        .unwrap();
    session.commit(patch).await.unwrap();
    let held = admission
        .try_lease(session.project_id(), first.unwrap())
        .unwrap();
    let release = tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        drop(held);
    });
    let mut committer = RejectCommitter;

    let report = execution::Execution::new(
        runner,
        resources,
        session.snapshot(),
        Request {
            operation: Operation::Only {
                stage: Stage::Translation,
            },
            ..Request::default()
        },
        &mut committer,
    )
    .await
    .unwrap()
    .run()
    .await
    .unwrap();
    release.await.unwrap();

    assert_eq!(report.status, RunStatus::Completed);
    assert_eq!(report.completed, 2);
}

/// The application spawns every execution, so its future must be `Send`; this only has to compile.
#[tokio::test]
async fn an_execution_can_run_on_a_spawned_task() {
    fn send<T: Send>(_: T) {}
    let pipeline = pipeline(Default::default());
    let mut committer = RejectCommitter;
    let snapshot = koharu_scene::Session::memory().await.unwrap().snapshot();
    send(pipeline.execute(snapshot, Request::default(), &mut committer));
}

#[tokio::test]
async fn stop_ends_a_job_waiting_for_its_stage() {
    let (admission, runner) = runner(&koharu_ml::Device::cpu());
    let held = admission.stage(Stage::Translation).await;
    assert_stop_ends_the_wait(&runner).await;
    drop(held);
}

#[tokio::test]
async fn stop_ends_a_job_waiting_for_the_accelerator_lane() {
    // Only an accelerator has a lane, and local translation takes it.
    let (admission, runner) = runner(&koharu_ml::Device::cuda(0));
    let held = admission.accelerator().acquire().await;
    assert_stop_ends_the_wait(&runner).await;
    drop(held);
}

fn runner(
    device: &koharu_ml::Device,
) -> (
    std::sync::Arc<admission::Admission>,
    stage_runner::StageRunner,
) {
    let admission = std::sync::Arc::new(admission::Admission::new(
        device,
        resources::ResourceMonitor::new(device),
    ));
    let translator = koharu_translator::Translator::from_config(
        device.clone(),
        koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
    )
    .unwrap();
    let runner = stage_runner::StageRunner::new(
        &PipelineConfig::default(),
        translator,
        device,
        admission.clone(),
        None,
    )
    .unwrap();
    (admission, runner)
}

/// Starts a translation job that waits for admission the caller holds, and checks that a stop
/// ends it at once without processing.
async fn assert_stop_ends_the_wait(runner: &stage_runner::StageRunner) {
    let scene = koharu_scene::Session::memory().await.unwrap().snapshot();
    let page = koharu_scene::EntityId::new();
    let stop = StopToken::default();
    let input = stages::StageInput::new(scene, page, None, None, Default::default(), None);
    let job = stage_runner::StageJob::new(Stage::Translation, vec![input], stop.clone(), None);
    let run = runner.run(job);
    tokio::pin!(run);
    let wait = std::time::Duration::from_millis(50);
    assert!(tokio::time::timeout(wait, &mut run).await.is_err());

    stop.stop();
    let completion = tokio::time::timeout(std::time::Duration::from_secs(5), run)
        .await
        .expect("a stop ends the wait");
    assert!(matches!(
        completion.outcome.as_deref(),
        Ok([stage_runner::StageOutcome::Stopped])
    ));
}

struct RejectCommitter;

fn pipeline(translation: TranslationConfig) -> Pipeline {
    let config = PipelineConfig {
        translation,
        ..PipelineConfig::default()
    };
    Pipeline::from_config(
        koharu_config::Config::memory(config),
        koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        koharu_ml::Device::cpu(),
    )
    .unwrap()
}

#[async_trait::async_trait]
impl Committer for RejectCommitter {
    async fn commit(&mut self, _output: StageOutput) -> anyhow::Result<koharu_scene::Snapshot> {
        anyhow::bail!("stopped execution must not commit")
    }
}

#[test]
fn operations_expand_to_the_supported_workflows() {
    assert_eq!(
        Operation::Through {
            stage: Stage::Translation,
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Ocr, Stage::Translation],
    );
    assert_eq!(
        Operation::Through {
            stage: Stage::Inpainting,
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Inpainting],
    );
    assert_eq!(
        Operation::Only {
            stage: Stage::Translation,
        }
        .stages()
        .unwrap(),
        vec![Stage::Translation],
    );
    assert_eq!(
        Operation::Stages {
            stages: vec![Stage::Translation, Stage::Detection, Stage::Translation],
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Translation],
    );
}
