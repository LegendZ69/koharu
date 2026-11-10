use std::{collections::BTreeSet, future::Future};

use anyhow::{Context as _, Result, ensure};
use async_trait::async_trait;
use futures::{StreamExt as _, TryStreamExt as _, stream};
use koharu_scene::{
    AssetRole, Authored, EntityId, LanguageTag, Origin, Patch, SourceText, Translation,
};
use koharu_translator::{
    PageImage, Provider, Translated, TranslationPage, TranslationRequest, Translator,
};

use crate::{StopToken, TranslationBatching, TranslationConfig, scheduler::Plan};

use super::{StageInput, StageLifecycle, finish, generation};

const PRODUCER: &str = "dev.koharu.pipeline.translation";

pub(super) struct Processor {
    pub(super) config: TranslationConfig,
    translator: Translator,
}

impl Processor {
    pub(super) fn new(config: TranslationConfig, translator: Translator) -> Self {
        Self { config, translator }
    }

    pub(super) async fn plan(&self) -> Plan {
        match self.config.batching {
            TranslationBatching::All | TranslationBatching::Halving => Plan::All,
            TranslationBatching::Custom => {
                Plan::Pages(self.config.pages_per_request.max(1) as usize)
            }
            TranslationBatching::Auto => {
                Plan::Budget(self.translator.batch_budget(&self.config.model).await)
            }
        }
    }

    /// Estimated source tokens the page adds to a request; a page without text adds none.
    pub(super) fn weight(&self, input: &StageInput) -> Result<usize> {
        let sources = targets(input)?
            .into_iter()
            .map(|(_, source)| source)
            .collect::<Vec<_>>();
        Ok(if sources.is_empty() {
            0
        } else {
            Translator::page_weight(&sources, self.vision())
        })
    }

    fn vision(&self) -> bool {
        Translator::supports_vision(&self.config.model, &self.config.generation)
    }

    /// Returns one patch per input, in input order. A page without text, or one left unsent
    /// because the run stopped, gets an empty patch.
    pub(super) async fn process(
        &self,
        inputs: Vec<StageInput>,
        stop: &StopToken,
    ) -> Result<Vec<Patch>> {
        let mut pages = Vec::with_capacity(inputs.len());
        for input in &inputs {
            pages.push(targets(input)?);
        }
        let sent = (0..pages.len())
            .filter(|&index| !pages[index].is_empty())
            .collect::<Vec<_>>();
        let counts = sent
            .iter()
            .map(|&index| pages[index].len())
            .collect::<Vec<_>>();
        let sending = match self.config.batching {
            TranslationBatching::All | TranslationBatching::Custom => Sending::Whole,
            TranslationBatching::Halving | TranslationBatching::Auto => Sending::Adaptive {
                split: Translator::splits_batches(&self.config.model),
            },
        };
        let batch = self.prepare(&inputs, &pages, &sent).await?;
        let translated = translate_batch(&counts, sending, stop, |group: &[usize]| {
            self.request(group.iter().map(|&index| batch[index].clone()).collect())
        })
        .await?;

        let mut results = inputs.iter().map(|_| None).collect::<Vec<_>>();
        for (index, translation) in sent.into_iter().zip(translated) {
            results[index] = translation;
        }
        inputs
            .iter()
            .zip(pages)
            .zip(results)
            .map(|((input, targets), translation)| self.patch(input, targets, translation))
            .collect()
    }

    /// The request page of each input index in `sent`, in that order. Each image is decoded,
    /// scaled down, and put in the form the provider reads once per job off the async thread,
    /// and only that form is kept, so every request of the job reuses it.
    async fn prepare(
        &self,
        inputs: &[StageInput],
        pages: &[Vec<(EntityId, String)>],
        sent: &[usize],
    ) -> Result<Vec<TranslationPage>> {
        let vision = self.vision();
        let provider = self.config.model.provider;
        let prepared = sent
            .iter()
            .map(|&index| async move {
                Ok(TranslationPage {
                    segments: pages[index]
                        .iter()
                        .map(|(_, source)| source.clone())
                        .collect(),
                    image: if vision {
                        source_image(&inputs[index], provider).await?
                    } else {
                        None
                    },
                })
            })
            .collect::<Vec<_>>();
        // Collected first: a stream that keeps the borrowing closure makes the job's future not
        // `Send`. One page per pool thread at a time keeps few full-resolution decodes alive.
        stream::iter(prepared)
            .buffered(rayon::current_num_threads())
            .try_collect()
            .await
    }

    /// Sends pages of one job as one provider request.
    async fn request(&self, pages: Vec<TranslationPage>) -> Result<Translated> {
        let mut request = TranslationRequest::pages(pages, self.config.target_language);
        if let Some(instructions) = self.config.instructions.as_deref() {
            request = request.with_instructions(instructions);
        }
        self.translator
            .translate(&self.config.model, self.config.generation, request)
            .await
    }

    fn patch(
        &self,
        input: &StageInput,
        targets: Vec<(EntityId, String)>,
        translation: Option<PageTranslation>,
    ) -> Result<Patch> {
        let Some(PageTranslation { provider, segments }) = translation else {
            return finish(input.scene.edit());
        };
        let language = LanguageTag::new(self.config.target_language.tag())?;
        let generated = generation(PRODUCER, provider)?;
        let mut edit = input.scene.edit_as(generated.clone());
        for (entity, _) in &targets {
            edit.observe::<SourceText>(*entity)?;
            edit.observe::<Translation>(*entity)?;
        }
        for ((entity, source), text) in targets.into_iter().zip(segments) {
            if input
                .scene
                .component::<Translation>(entity)?
                .is_some_and(|value| matches!(value.text.origin, Origin::User))
            {
                continue;
            }
            let text = if source.trim() == "\u{2026}" {
                "\u{2026}".to_owned()
            } else {
                text
            };
            edit.set(
                entity,
                &Translation {
                    text: Authored::generated(text, generated.clone()),
                    language: Some(language.clone()),
                },
            )?;
        }
        finish(edit)
    }
}

#[async_trait]
impl StageLifecycle for Processor {
    fn model(&self) -> &'static str {
        Translator::model(&self.config.model)
    }

    fn unload(&self) -> bool {
        self.translator.unload()
    }

    async fn load(&self) -> Result<()> {
        self.translator.load_model(&self.config.model).await
    }

    fn accelerated(&self) -> bool {
        // A hosted provider does not run on this device.
        self.config.model.provider == Provider::Local
    }
}

/// The page's in-scope text content with a non-blank source, in reading order.
fn targets(input: &StageInput) -> Result<Vec<(EntityId, String)>> {
    let mut targets = Vec::new();
    if let Some(group) = input.scene.page(input.page)?.text_group()? {
        for layer in group.text_layers()? {
            if !input.contains_entity(layer.id())? {
                continue;
            }
            let content = layer.content()?;
            let Some(source) = content.source()? else {
                continue;
            };
            if !source.text.value.trim().is_empty() {
                targets.push((content.id(), source.text.value));
            }
        }
    }
    Ok(targets)
}

/// The page's source image as a request to `provider` sends it. Neither the full-resolution
/// decode nor the pixels a hosted provider does not read outlive the decoding thread.
async fn source_image(input: &StageInput, provider: Provider) -> Result<Option<PageImage>> {
    let Some(asset) = input.scene.asset(input.page, &AssetRole::new("source")?)? else {
        return Ok(None);
    };
    let bytes = input.scene.read_blob(asset.blob).await?;
    let page = input.page;
    tokio_rayon::spawn(move || {
        let image = image::load_from_memory(&bytes)
            .with_context(|| format!("failed to decode source image for entity {page}"))?;
        Ok(Some(PageImage::new(image, provider)?))
    })
    .await
}

struct PageTranslation {
    provider: &'static str,
    segments: Vec<String>,
}

/// How one translation job sends its pages.
#[derive(Clone, Copy, Debug)]
enum Sending {
    /// The scheduler's batch in one request.
    Whole,
    /// The whole batch first; then smaller requests for the pages a multi-page response left
    /// incomplete and, with `split`, for a multi-page request that failed as too large.
    Adaptive { split: bool },
}

/// Sends pages that have `counts[page]` segments each. `request` translates a group of page
/// indexes, in page order, as one provider request. Returns one translation per page, or `None`
/// for a page left unsent because `stop` was requested.
///
/// Every smaller request covers fewer pages than the one it replaces, and a single page is
/// never split, so the number of requests is bounded.
async fn translate_batch<F, Fut>(
    counts: &[usize],
    sending: Sending,
    stop: &StopToken,
    mut request: F,
) -> Result<Vec<Option<PageTranslation>>>
where
    F: FnMut(&[usize]) -> Fut,
    Fut: Future<Output = Result<Translated>>,
{
    let mut translated = counts.iter().map(|_| None).collect::<Vec<_>>();
    // Last first, so the halves of a group go out in page order.
    let mut pending = vec![(0..counts.len()).collect::<Vec<_>>()];
    while let Some(group) = pending.pop() {
        if group.is_empty() {
            continue;
        }
        if stop.stopped() {
            break;
        }
        let several = group.len() > 1;
        let response = match request(&group).await {
            Ok(response) => response,
            Err(error)
                if several
                    && matches!(sending, Sending::Adaptive { split: true })
                    && Translator::is_oversize(&error) =>
            {
                tracing::debug!(
                    pages = group.len(),
                    "splitting a translation request that was too large: {error:#}"
                );
                pending.extend(halves(group).into_iter().rev());
                continue;
            }
            Err(error) => return Err(error),
        };
        let expected = group.iter().map(|&page| counts[page]).sum::<usize>();
        ensure!(
            response.segments.len() == expected,
            "{} returned {} translations for {expected} segments",
            response.provider,
            response.segments.len()
        );
        let missing = response.missing.into_iter().collect::<BTreeSet<_>>();
        let mut segments = response.segments.into_iter();
        let mut start = 0;
        let mut incomplete = Vec::new();
        for page in group {
            let end = start + counts[page];
            let page_segments = segments.by_ref().take(counts[page]).collect();
            if several
                && matches!(sending, Sending::Adaptive { .. })
                && missing.range(start..end).next().is_some()
            {
                incomplete.push(page);
            } else {
                translated[page] = Some(PageTranslation {
                    provider: response.provider,
                    segments: page_segments,
                });
            }
            start = end;
        }
        // A single page accepts its response as is: missing segments keep their source text.
        match incomplete.len() {
            0 => {}
            1 => pending.push(incomplete),
            _ => pending.extend(halves(incomplete).into_iter().rev()),
        }
    }
    Ok(translated)
}

fn halves(mut group: Vec<usize>) -> [Vec<usize>; 2] {
    let second = group.split_off(group.len().div_ceil(2));
    [group, second]
}

#[cfg(test)]
mod tests {
    use std::future::ready;

    use super::*;

    fn answer(counts: &[usize], group: &[usize], missing: &[usize]) -> Result<Translated> {
        let mut segments = Vec::new();
        for &page in group {
            segments.extend((0..counts[page]).map(|segment| format!("{page}.{segment}")));
        }
        Ok(Translated {
            provider: "test",
            segments,
            missing: missing.to_vec(),
        })
    }

    fn oversize() -> anyhow::Error {
        anyhow::anyhow!("null reference from llama.cpp")
            .context("failed to create llama.cpp context")
    }

    /// Sends `counts` to a fake provider and returns the groups it received with the result.
    async fn send(
        counts: &[usize],
        sending: Sending,
        mut provider: impl FnMut(&[usize]) -> Result<Translated>,
    ) -> (Vec<Vec<usize>>, Result<Vec<Option<Vec<String>>>>) {
        let mut groups = Vec::new();
        let result = translate_batch(
            counts,
            sending,
            &StopToken::default(),
            |group: &[usize]| {
                groups.push(group.to_vec());
                ready(provider(group))
            },
        )
        .await
        .map(|pages| {
            pages
                .into_iter()
                .map(|page| page.map(|page| page.segments))
                .collect()
        });
        (groups, result)
    }

    fn texts(pages: &[&[&str]]) -> Vec<Option<Vec<String>>> {
        pages
            .iter()
            .map(|page| Some(page.iter().map(|text| (*text).to_owned()).collect()))
            .collect()
    }

    #[tokio::test]
    async fn a_batch_that_succeeds_is_one_request_split_back_per_page() {
        let counts = [1, 2];
        for sending in [Sending::Whole, Sending::Adaptive { split: true }] {
            let (groups, result) =
                send(&counts, sending, |group| answer(&counts, group, &[])).await;
            assert_eq!(groups, [vec![0, 1]]);
            assert_eq!(result.unwrap(), texts(&[&["0.0"], &["1.0", "1.1"]]));
        }
    }

    #[tokio::test]
    async fn an_oversized_batch_splits_into_halves_in_page_order() {
        let counts = [1, 1, 1];
        let (groups, result) = send(&counts, Sending::Adaptive { split: true }, |group| {
            if group.len() > 1 {
                Err(oversize())
            } else {
                answer(&counts, group, &[])
            }
        })
        .await;
        assert_eq!(
            groups,
            [vec![0, 1, 2], vec![0, 1], vec![0], vec![1], vec![2]]
        );
        assert_eq!(result.unwrap(), texts(&[&["0.0"], &["1.0"], &["2.0"]]));
    }

    #[tokio::test]
    async fn other_failures_fixed_batches_and_hosted_mt_do_not_split() {
        let counts = [1, 1];
        for (sending, error) in [
            (Sending::Adaptive { split: true }, anyhow::anyhow!("quota")),
            (Sending::Adaptive { split: false }, oversize()),
            (Sending::Whole, oversize()),
        ] {
            let mut error = Some(error);
            let (groups, result) = send(&counts, sending, |_| Err(error.take().unwrap())).await;
            assert_eq!(groups, [vec![0, 1]]);
            assert!(result.is_err());
        }
    }

    #[tokio::test]
    async fn only_pages_with_missing_segments_are_translated_again() {
        let counts = [1, 2, 1, 1];
        let mut responses = vec![
            answer(&counts, &[3], &[0]),
            answer(&counts, &[1], &[]),
            answer(&counts, &[0, 1, 2, 3], &[1, 4]),
        ];
        let (groups, result) = send(&counts, Sending::Adaptive { split: true }, |_| {
            responses.pop().unwrap()
        })
        .await;
        assert_eq!(groups, [vec![0, 1, 2, 3], vec![1], vec![3]]);
        // The last single-page response is accepted with its missing segment padded.
        assert_eq!(
            result.unwrap(),
            texts(&[&["0.0"], &["1.0", "1.1"], &["2.0"], &["3.0"]])
        );

        let (groups, result) = send(&counts[..2], Sending::Whole, |group| {
            answer(&counts, group, &[0])
        })
        .await;
        assert_eq!(groups, [vec![0, 1]]);
        assert_eq!(result.unwrap(), texts(&[&["0.0"], &["1.0", "1.1"]]));
    }

    #[tokio::test]
    async fn splitting_ends_at_single_pages() {
        let counts = [1, 1, 1, 1];
        let (groups, result) = send(&counts, Sending::Adaptive { split: true }, |_| {
            Err(oversize())
        })
        .await;
        assert_eq!(groups, [vec![0, 1, 2, 3], vec![0, 1], vec![0]]);
        assert!(result.is_err());

        let (groups, result) = send(&counts, Sending::Adaptive { split: true }, |group| {
            let missing = (0..group.len()).collect::<Vec<_>>();
            answer(&counts, group, &missing)
        })
        .await;
        assert_eq!(
            groups,
            [
                vec![0, 1, 2, 3],
                vec![0, 1],
                vec![0],
                vec![1],
                vec![2, 3],
                vec![2],
                vec![3],
            ]
        );
        assert_eq!(
            result.unwrap(),
            texts(&[&["0.0"], &["1.0"], &["2.0"], &["3.0"]])
        );
    }

    #[tokio::test]
    async fn a_stop_leaves_the_remaining_pages_unsent() {
        let counts = [1, 1];
        let stop = StopToken::default();
        let mut groups = Vec::new();
        let translated = translate_batch(
            &counts,
            Sending::Adaptive { split: true },
            &stop,
            |group: &[usize]| {
                groups.push(group.to_vec());
                stop.stop();
                ready(Err(oversize()))
            },
        )
        .await
        .unwrap();
        assert_eq!(groups, [vec![0, 1]]);
        assert!(translated.iter().all(Option::is_none));
    }
}
