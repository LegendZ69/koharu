use std::sync::Arc;

use anyhow::{Context as _, Result};
use arc_swap::ArcSwap;
use koharu_config::Config;
use koharu_scene::Snapshot;

use crate::{
    Committer, PipelineConfig, PipelineError, Report, Request, ResourceSnapshot,
    admission::Admission, execution::Execution, resources::ResourceMonitor,
    stage_runner::StageRunner,
};

#[derive(Clone)]
pub struct Pipeline {
    current: Arc<ArcSwap<StageRunner>>,
    resources: Arc<ResourceMonitor>,
}

impl Pipeline {
    pub fn load(device: koharu_ml::Device) -> Result<Self> {
        Self::from_config(
            PipelineConfig::load()?,
            koharu_translator::ProvidersConfig::load()?,
            device,
        )
    }

    #[tracing::instrument(skip_all)]
    pub fn from_config(
        config: Config<PipelineConfig>,
        providers: Config<koharu_translator::ProvidersConfig>,
        device: koharu_ml::Device,
    ) -> Result<Self> {
        let translator = koharu_translator::Translator::from_config(device.clone(), providers)?;
        let resources = ResourceMonitor::new(&device);
        let admission = Arc::new(Admission::new(&device, resources.clone()));
        let runner = {
            let value = config.read()?;
            StageRunner::new(&value, translator.clone(), &device, admission.clone(), None)?
        };
        let current = Arc::new(ArcSwap::from_pointee(runner));
        let watched = current.clone();
        let _watcher = tokio::runtime::Handle::try_current()
            .context("pipeline requires a Tokio runtime")?
            .spawn(async move {
                let mut changes = config.subscribe();
                while changes.changed().await.is_ok() {
                    let previous = watched.load_full();
                    let runner = config.read().and_then(|value| {
                        StageRunner::new(
                            &value,
                            translator.clone(),
                            &device,
                            admission.clone(),
                            Some(&previous),
                        )
                    });
                    match runner {
                        Ok(runner) => watched.store(Arc::new(runner)),
                        Err(error) => tracing::error!(%error, "failed to reload pipeline"),
                    }
                }
            });
        Ok(Self { current, resources })
    }

    pub fn subscribe_resources(&self) -> tokio::sync::watch::Receiver<ResourceSnapshot> {
        self.resources.start();
        self.resources.subscribe()
    }

    /// Several requests may run at once, also on one project. At most one job per stage runs at
    /// a time across all of them, and a page is processed by one run of its project at a time.
    #[tracing::instrument(skip_all)]
    pub async fn execute(
        &self,
        snapshot: Snapshot,
        request: Request,
        committer: &mut dyn Committer,
    ) -> std::result::Result<Report, PipelineError> {
        Execution::new(
            self.current.load_full(),
            self.resources.clone(),
            snapshot,
            request,
            committer,
        )
        .await?
        .run()
        .await
    }
}
