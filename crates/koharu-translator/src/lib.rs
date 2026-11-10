//! Translation through local and hosted providers.

mod backend;
mod batch;
mod error;
mod language;
mod local;
mod model;
mod prompt;
mod provider;
mod remote;

use std::{sync::Arc, time::Duration};

use koharu_ml::Device;

use backend::Translations;
use batch::Batching;
use error::{Error, Result};
use local::LocalTranslator;

pub use backend::{PageImage, Translated, TranslationContext, TranslationPage, TranslationRequest};
pub use language::Language;
pub use model::{GenerationConfig, Model, ModelSelection, Quantization};
pub(crate) use model::{ModelGeneration, QuantizationDefinition, display_name};
pub use provider::{Provider, ProviderConfig, ProvidersConfig};

#[derive(Clone)]
pub struct Translator {
    providers: koharu_config::Config<ProvidersConfig>,
    local: Arc<tokio::sync::Mutex<Option<LoadedLocal>>>,
    client: reqwest::Client,
    device: Device,
}

struct LoadedLocal {
    model: Option<String>,
    quantization: Option<String>,
    translator: Arc<LocalTranslator>,
}

impl LoadedLocal {
    fn matches(&self, selection: &ModelSelection) -> bool {
        self.model == selection.model && self.quantization == selection.quantization
    }
}

impl Translator {
    pub fn from_config(
        device: Device,
        providers: koharu_config::Config<ProvidersConfig>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            providers,
            local: Arc::new(tokio::sync::Mutex::new(None)),
            client: koharu_runtime::http_client()?,
            device,
        })
    }

    #[must_use]
    pub fn model(selection: &ModelSelection) -> &'static str {
        selection.provider.into()
    }

    /// Whether `translate` keeps the request's page images for this selection and generation.
    #[must_use]
    pub fn supports_vision(selection: &ModelSelection, generation: &GenerationConfig) -> bool {
        generation.for_model(selection).vision.unwrap_or(false)
            && (selection.provider != Provider::Local || local::supports_vision(selection))
    }

    /// Whether a multi-page request that failed as oversized can be split into smaller requests.
    /// Language models can; hosted machine translation fails immediately.
    #[must_use]
    pub fn splits_batches(selection: &ModelSelection) -> bool {
        batching(selection).splits()
    }

    /// Estimated source tokens one request should carry, in the unit of [`Self::page_weight`].
    pub async fn batch_budget(&self, selection: &ModelSelection) -> usize {
        if selection.provider == Provider::Local {
            return local::batching(selection).budget();
        }
        // Cloned so that no read guard lives across the provider query. A configuration that
        // cannot be read fails the request itself.
        match self.providers.read().map(|providers| providers.clone()) {
            Ok(providers) => remote::reported_batching(&self.client, &providers, selection).await,
            Err(_) => remote::batching(selection.provider),
        }
        .budget()
    }

    /// Estimated source tokens a page adds to a request.
    #[must_use]
    pub fn page_weight(segments: &[String], with_image: bool) -> usize {
        batch::page_weight(segments, with_image)
    }

    /// Whether a `translate` failure means the request was too large for the provider or model.
    #[must_use]
    pub fn is_oversize(error: &anyhow::Error) -> bool {
        error
            .chain()
            .any(|cause| match cause.downcast_ref::<Error>() {
                Some(error) => error.is_oversize(),
                None => local::is_context_overflow(&cause.to_string()),
            })
    }

    #[must_use]
    pub fn loaded(&self, selection: &ModelSelection) -> bool {
        if selection.provider != Provider::Local {
            return true;
        }
        self.local
            .try_lock()
            .map(|loaded| {
                loaded
                    .as_ref()
                    .is_some_and(|loaded| loaded.matches(selection))
            })
            .unwrap_or(true)
    }

    pub fn unload(&self) -> bool {
        self.local
            .try_lock()
            .map(|mut loaded| loaded.take().is_some())
            .unwrap_or(false)
    }

    #[tracing::instrument(skip_all)]
    pub async fn load_model(&self, selection: &ModelSelection) -> anyhow::Result<()> {
        if selection.provider == Provider::Local {
            self.local(selection).await?;
        }
        Ok(())
    }

    pub async fn translate(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        mut request: TranslationRequest,
    ) -> anyhow::Result<Translated> {
        let provider = selection.provider;
        let provider_id: &'static str = provider.into();
        if request.segments.is_empty() {
            return Ok(Translated {
                provider: provider_id,
                segments: request.segments,
                missing: Vec::new(),
            });
        }

        let generation = generation.for_model(selection);

        if !Self::supports_vision(selection, &generation) {
            request.remove_images();
        }

        let expected = request.segments.len();
        let Translations { segments, missing } = if provider == Provider::Local {
            self.local(selection)
                .await?
                .translate(request, generation)
                .await?
        } else {
            let providers = self.providers.read()?.clone();
            remote::translate(&self.client, &providers, selection, &generation, &request).await?
        };
        if segments.len() != expected {
            return Err(Error::SegmentCount {
                provider: provider_id,
                expected,
                actual: segments.len(),
            }
            .into());
        }
        Ok(Translated {
            provider: provider_id,
            segments,
            missing,
        })
    }

    #[tracing::instrument(skip_all)]
    pub async fn models() -> anyhow::Result<Vec<Model>> {
        static CLIENT: tokio::sync::OnceCell<reqwest::Client> = tokio::sync::OnceCell::const_new();

        let providers = ProvidersConfig::load()?;
        let providers = providers.read()?.clone();
        let client = CLIENT
            .get_or_try_init(|| async {
                reqwest::Client::builder()
                    .timeout(Duration::from_secs(5))
                    .build()
            })
            .await?;
        let mut models = local::models();
        models.extend(remote::models(client, &providers).await);
        Ok(models)
    }

    async fn local(&self, selection: &ModelSelection) -> Result<Arc<LocalTranslator>> {
        let mut loaded = self.local.lock().await;
        if loaded
            .as_ref()
            .is_none_or(|loaded| !loaded.matches(selection))
        {
            *loaded = Some(LoadedLocal {
                model: selection.model.clone(),
                quantization: selection.quantization.clone(),
                translator: Arc::new(LocalTranslator::load(self.device.clone(), selection).await?),
            });
        }
        Ok(Arc::clone(
            &loaded
                .as_ref()
                .expect("local translator was loaded")
                .translator,
        ))
    }
}

fn batching(selection: &ModelSelection) -> Batching {
    match selection.provider {
        Provider::Local => local::batching(selection),
        provider => remote::batching(provider),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local_selection(model: &str) -> ModelSelection {
        ModelSelection {
            provider: Provider::Local,
            model: Some(model.to_owned()),
            quantization: None,
            vision: true,
            reasoning: true,
        }
    }

    #[test]
    fn local_vision_requires_capability_and_generation_setting() {
        assert!(Translator::supports_vision(
            &local_selection("gemma4-e2b-it"),
            &GenerationConfig {
                vision: Some(true),
                ..GenerationConfig::default()
            }
        ));
        assert!(!Translator::supports_vision(
            &local_selection("gemma4-e2b-it"),
            &GenerationConfig {
                vision: Some(false),
                ..GenerationConfig::default()
            }
        ));
        assert!(!Translator::supports_vision(
            &local_selection("lfm2.5-1.2b-instruct"),
            &GenerationConfig {
                vision: Some(true),
                ..GenerationConfig::default()
            }
        ));
    }

    #[test]
    fn vision_follows_the_selected_model_capability() {
        let generation = GenerationConfig {
            vision: Some(true),
            ..GenerationConfig::default()
        };
        let remote = |vision| ModelSelection {
            provider: Provider::OpenAi,
            model: Some("gpt-5.6-luna".to_owned()),
            quantization: None,
            vision,
            reasoning: false,
        };
        assert!(Translator::supports_vision(&remote(true), &generation));
        assert!(!Translator::supports_vision(&remote(false), &generation));
        assert!(!Translator::supports_vision(
            &ModelSelection {
                vision: false,
                ..local_selection("gemma4-e2b-it")
            },
            &generation
        ));
    }

    #[tokio::test]
    async fn local_requests_are_sized_by_the_selected_model_context() {
        assert_eq!(
            batching(&local_selection("lfm2.5-1.2b-instruct")),
            Batching::prompt(32_768)
        );
        assert_eq!(
            batching(&local_selection("gemma4-e4b-it")),
            Batching::prompt(131_072)
        );
        let translator = Translator::from_config(
            Device::cpu(),
            koharu_config::Config::memory(ProvidersConfig::default()),
        )
        .unwrap();
        assert_eq!(
            translator
                .batch_budget(&local_selection(local::DEFAULT_MODEL))
                .await,
            8_192
        );
    }

    #[test]
    fn classifies_failures_that_fewer_pages_can_avoid() {
        fn api(status: u16, message: &str) -> anyhow::Error {
            Error::Api {
                provider: "test",
                status,
                message: message.to_owned(),
            }
            .into()
        }
        let oversize: Vec<anyhow::Error> = vec![
            prompt::translations("test", r#"{"translations":["#, &["one".to_owned()])
                .unwrap_err()
                .into(),
            Error::Truncated { provider: "local" }.into(),
            Error::ContextExceeded {
                provider: "local",
                required: 40_000,
                context: 32_768,
            }
            .into(),
            api(413, "Request Entity Too Large"),
            api(400, r#"{"error":{"code":"context_length_exceeded"}}"#),
            api(400, "This model's maximum context length is 8192 tokens."),
            api(400, "prompt is too long: 215000 tokens > 200000 maximum"),
            api(422, "Input is too long for requested model."),
            // Output budgets a multi-page request scaled past the model's limit.
            api(
                400,
                "max_tokens: 128001 > 128000, which is the maximum allowed number of output \
                 tokens for claude-opus-4-6",
            ),
            api(
                400,
                "max_completion_tokens is too large: 20000. This model supports at most 16384 \
                 completion tokens, whereas you provided 20000.",
            ),
            api(
                400,
                "Unable to submit request because it has a maxOutputTokens value of 16384 but the \
                 supported range is from 1 (inclusive) to 8193 (exclusive). Update the value and \
                 try again.",
            ),
            Error::Other(
                anyhow::anyhow!("null reference from llama.cpp")
                    .context("failed to create llama.cpp context"),
            )
            .into(),
            Error::Other(
                anyhow::anyhow!("Decode Error 1: NoKvCacheSlot")
                    .context("failed to decode prompt batch"),
            )
            .into(),
            Error::Other(anyhow::anyhow!(
                "n_ctx is too small: need at least 9000 positions for prompt and generation"
            ))
            .into(),
            anyhow::Error::from(Error::Truncated { provider: "local" })
                .context("translation failed"),
        ];
        for error in &oversize {
            assert!(Translator::is_oversize(error), "{error:#}");
        }

        let other: Vec<anyhow::Error> = vec![
            Error::QuotaExceeded { provider: "test" }.into(),
            Error::UnsupportedLanguage {
                provider: "test",
                language: Language::Filipino,
            }
            .into(),
            Error::UnsupportedSourceLanguage {
                provider: "test",
                language: Language::Filipino,
            }
            .into(),
            Error::SegmentCount {
                provider: "test",
                expected: 2,
                actual: 1,
            }
            .into(),
            api(400, "Incorrect API key provided."),
            api(401, "maximum context length"),
            api(500, "context length exceeded"),
            Error::Other(anyhow::anyhow!("connection reset").context("openai request failed"))
                .into(),
        ];
        for error in &other {
            assert!(!Translator::is_oversize(error), "{error:#}");
        }
    }
}
