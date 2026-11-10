mod detection;
mod inpainting;
mod ocr;
mod translation;

use std::{collections::BTreeSet, sync::Arc};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use koharu_scene::{Edit, EntityId, Generation, Patch, ProducerId, Snapshot};

pub use detection::KoharuLayoutRFDetrSeg2XLConfig;
pub use inpainting::{Flux2KleinConfig, RoremMixedConfig};

use crate::{
    Bounds, ImageCache, InpaintingMask, PipelineConfig, Stage, StopToken, scheduler::Plan,
};

#[derive(Clone)]
pub(crate) struct StageInput {
    scene: koharu_scene::Snapshot,
    page: EntityId,
    entities: Option<Arc<BTreeSet<EntityId>>>,
    region: Option<Bounds>,
    images: Arc<ImageCache>,
    inpainting_mask: Option<InpaintingMask>,
}

impl StageInput {
    pub(crate) fn new(
        scene: Snapshot,
        page: EntityId,
        entities: Option<Arc<BTreeSet<EntityId>>>,
        region: Option<Bounds>,
        images: Arc<ImageCache>,
        inpainting_mask: Option<InpaintingMask>,
    ) -> Self {
        Self {
            scene,
            page,
            entities,
            region,
            images,
            inpainting_mask,
        }
    }

    pub(crate) fn page(&self) -> EntityId {
        self.page
    }

    fn contains_entity(&self, entity: EntityId) -> Result<bool> {
        crate::scope::contains_entity(
            &self.scene,
            self.page,
            self.entities.as_deref(),
            self.region,
            entity,
        )
    }
}

/// The lifecycle every stage processor shares. Processing stays per kind: detection, OCR, and
/// inpainting process one page, and translation processes a batch of pages.
#[async_trait]
trait StageLifecycle: Send + Sync {
    fn model(&self) -> &'static str;
    fn skip(&self, _input: &StageInput) -> Result<bool> {
        Ok(false)
    }
    fn unload(&self) -> bool;
    async fn load(&self) -> Result<()>;
    /// Whether the stage's work takes the accelerator lane.
    fn accelerated(&self) -> bool {
        true
    }
}

pub(crate) struct Stages {
    detection: Arc<detection::Processor>,
    ocr: Arc<ocr::Processor>,
    translation: Arc<translation::Processor>,
    inpainting: Arc<inpainting::Processor>,
    /// The local-model slot that every generation's translation processor shares.
    translator: koharu_translator::Translator,
}

impl Stages {
    /// Builds one runner generation. A processor whose stage configuration is unchanged carries
    /// over from `previous`, so a preference save neither duplicates nor unloads its resident
    /// model, and runs on either generation share it.
    pub(crate) fn new(
        config: &PipelineConfig,
        translator: koharu_translator::Translator,
        device: &koharu_ml::Device,
        previous: Option<&Self>,
    ) -> Result<Self> {
        Ok(Self {
            detection: carry_over(
                previous.map(|stages| &stages.detection),
                detection::Processor::new(config.detection()?, device.clone()),
                |processor| &processor.config,
            ),
            ocr: carry_over(
                previous.map(|stages| &stages.ocr),
                ocr::Processor::new(config.ocr.clone(), device.clone()),
                |processor| &processor.config,
            ),
            translation: carry_over(
                previous.map(|stages| &stages.translation),
                translation::Processor::new(config.translation.clone(), translator.clone()),
                |processor| &processor.config,
            ),
            inpainting: carry_over(
                previous.map(|stages| &stages.inpainting),
                inpainting::Processor::new(config.inpainting()?, device.clone())?,
                |processor| &processor.config,
            ),
            translator,
        })
    }

    fn processor(&self, stage: Stage) -> &dyn StageLifecycle {
        match stage {
            Stage::Detection => &*self.detection,
            Stage::Ocr => &*self.ocr,
            Stage::Translation => &*self.translation,
            Stage::Inpainting => &*self.inpainting,
        }
    }

    pub(crate) fn model(&self, stage: Stage) -> &'static str {
        self.processor(stage).model()
    }

    pub(crate) fn skip(&self, stage: Stage, input: &StageInput) -> Result<bool> {
        self.processor(stage).skip(input)
    }

    pub(crate) fn accelerated(&self, stage: Stage) -> bool {
        self.processor(stage).accelerated()
    }

    pub(crate) async fn load(&self, stage: Stage) -> Result<()> {
        self.processor(stage).load().await
    }

    /// Returns one patch per input, in input order. Only translation accepts more than one page.
    pub(crate) async fn process(
        &self,
        stage: Stage,
        inputs: Vec<StageInput>,
        stop: &StopToken,
    ) -> Result<Vec<Patch>> {
        match stage {
            Stage::Detection => Ok(vec![self.detection.process(single(inputs)?).await?]),
            Stage::Ocr => Ok(vec![self.ocr.process(single(inputs)?).await?]),
            Stage::Translation => self.translation.process(inputs, stop).await,
            Stage::Inpainting => Ok(vec![self.inpainting.process(single(inputs)?).await?]),
        }
    }

    pub(crate) fn unload(&self, stage: Stage) -> bool {
        self.processor(stage).unload()
    }

    /// Unloads this generation's models except the one `requester` runs `stage` with. A model
    /// in use stays loaded. Returns whether any model was unloaded.
    pub(crate) fn unload_except(&self, stage: Stage, requester: &Self) -> bool {
        let mut unloaded = false;
        for other in Stage::ALL {
            if other == stage && self.shares_model(stage, requester) {
                continue;
            }
            if self.unload(other) {
                unloaded = true;
                tracing::debug!(
                    stage = %other,
                    "unloaded model while recovering from memory pressure"
                );
            }
        }
        unloaded
    }

    fn shares_model(&self, stage: Stage, other: &Self) -> bool {
        match stage {
            Stage::Detection => Arc::ptr_eq(&self.detection, &other.detection),
            Stage::Ocr => Arc::ptr_eq(&self.ocr, &other.ocr),
            // Every generation's translator shares one local-model slot, which keeps the
            // previous model until a newly selected one has loaded.
            Stage::Translation => self.translator.loaded(&other.translation.config.model),
            Stage::Inpainting => Arc::ptr_eq(&self.inpainting, &other.inpainting),
        }
    }

    pub(crate) async fn translation_plan(&self) -> Plan {
        self.translation.plan().await
    }

    pub(crate) fn translation_weight(&self, input: &StageInput) -> Result<usize> {
        self.translation.weight(input)
    }
}

fn carry_over<P, C: PartialEq>(
    previous: Option<&Arc<P>>,
    next: P,
    config: impl Fn(&P) -> &C,
) -> Arc<P> {
    match previous {
        Some(previous) if config(previous) == config(&next) => Arc::clone(previous),
        _ => Arc::new(next),
    }
}

fn single(inputs: Vec<StageInput>) -> Result<StageInput> {
    let [input] = <[StageInput; 1]>::try_from(inputs)
        .map_err(|inputs| anyhow!("a page-local stage received {} pages", inputs.len()))?;
    Ok(input)
}

fn generation(producer: &str, model: &str) -> Result<Generation> {
    let mut generation = Generation::new(ProducerId::new(producer)?);
    generation.model = Some(model.to_owned());
    Ok(generation)
}

fn finish(edit: Edit) -> Result<Patch> {
    edit.finish().map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(bytes: &'static [u8]) -> koharu_scene::AssetInput {
        koharu_scene::AssetInput::new(
            bytes,
            "image/png",
            koharu_scene::AssetMetadata {
                width: Some(1),
                height: Some(1),
                attributes: std::collections::BTreeMap::new(),
            },
        )
    }

    #[tokio::test]
    async fn registry_contains_every_stage() {
        let translator = koharu_translator::Translator::from_config(
            koharu_ml::Device::cpu(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        let stages = Stages::new(
            &PipelineConfig::default(),
            translator,
            &koharu_ml::Device::cpu(),
            None,
        )
        .unwrap();

        assert_eq!(
            Stage::ALL.map(|stage| stages.model(stage)),
            [
                "koharu-layout-rfdetr-seg-2xl",
                "paddleocr-vl-1.6",
                "local",
                "lama",
            ]
        );
    }

    #[tokio::test]
    async fn a_generation_keeps_processors_whose_configuration_is_unchanged() {
        let translator = koharu_translator::Translator::from_config(
            koharu_ml::Device::cpu(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        let device = koharu_ml::Device::cpu();
        let previous = Stages::new(
            &PipelineConfig::default(),
            translator.clone(),
            &device,
            None,
        )
        .unwrap();
        let config = PipelineConfig {
            ocr: crate::OcrModel::MangaOcr,
            ..PipelineConfig::default()
        };
        let next = Stages::new(&config, translator, &device, Some(&previous)).unwrap();

        assert!(Arc::ptr_eq(&previous.detection, &next.detection));
        assert!(!Arc::ptr_eq(&previous.ocr, &next.ocr));
        assert!(Arc::ptr_eq(&previous.translation, &next.translation));
        assert!(Arc::ptr_eq(&previous.inpainting, &next.inpainting));
    }

    #[tokio::test]
    async fn translation_and_inpainting_compose_without_weakening_text_guards() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(
                koharu_scene::PageDraft::new("page", 1.0, 1.0),
                koharu_scene::At::End,
            )
            .unwrap();
        let text = setup.add_text_content(page, koharu_scene::At::End).unwrap();
        setup
            .set(
                text,
                &koharu_scene::SourceText {
                    text: koharu_scene::Authored::user("before".to_owned()),
                    language: None,
                },
            )
            .unwrap();
        setup
            .set_asset(
                page,
                &koharu_scene::AssetRole::new("source").unwrap(),
                asset(b"source"),
            )
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let base = session.snapshot();

        let mut text_edit = base.edit();
        text_edit.observe::<koharu_scene::SourceText>(text).unwrap();
        text_edit
            .observe::<koharu_scene::Translation>(text)
            .unwrap();
        text_edit
            .set(
                text,
                &koharu_scene::Translation {
                    text: koharu_scene::Authored::user("after".to_owned()),
                    language: None,
                },
            )
            .unwrap();
        let text_patch = text_edit.finish().unwrap();

        let mut image_edit = base.edit();
        image_edit.observe_assets(page).unwrap();
        let cleanup = image_edit
            .add_entity(page, koharu_scene::At::Start)
            .unwrap();
        image_edit
            .set(
                cleanup,
                &koharu_scene::RasterLayer {
                    origin: koharu_scene::Origin::User,
                    name: "Cleanup".to_owned(),
                    kind: koharu_scene::RasterLayerKind::Cleanup,
                },
            )
            .unwrap();
        image_edit
            .set_asset(
                cleanup,
                &koharu_scene::AssetRole::new("source").unwrap(),
                asset(b"clean"),
            )
            .unwrap();
        let image_patch = image_edit.finish().unwrap();

        let image_first = base.preview([&image_patch]).unwrap();
        assert!(text_patch.rebase_on(&image_first).is_ok());
        let text_first = base.preview([&text_patch]).unwrap();
        assert!(image_patch.rebase_on(&text_first).is_ok());

        let changed_source = base
            .patch(|edit| {
                edit.set(
                    text,
                    &koharu_scene::SourceText {
                        text: koharu_scene::Authored::user("changed".to_owned()),
                        language: None,
                    },
                )
            })
            .unwrap();
        let changed_source = base.preview([&changed_source]).unwrap();
        assert!(text_patch.rebase_on(&changed_source).is_err());
    }
}
