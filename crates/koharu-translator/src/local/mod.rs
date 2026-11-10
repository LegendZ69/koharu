use std::sync::Arc;

use anyhow::Context;
use koharu_ml::llm::{
    ChatMessage, ChatTemplateOptions, FinishReason, Generation, Input, Llm, LoadOptions,
    MtmdOptions, media_marker,
};

mod catalog;

pub use catalog::LocalConfig;
use catalog::LocalModelDescriptor;
pub(crate) use catalog::{DEFAULT_MODEL, DEFAULT_QUANTIZATION};

use crate::{
    Device, Error, GenerationConfig, Model, ModelSelection, Provider, Quantization, Result,
    TranslationRequest,
    backend::Translations,
    batch::{self, Batching},
    prompt,
};

#[derive(Debug)]
pub struct LocalTranslator {
    descriptor: LocalModelDescriptor,
    llm: Arc<Llm>,
}

impl LocalTranslator {
    pub async fn load(device: Device, selection: &ModelSelection) -> Result<Self> {
        let model = selection
            .model
            .as_deref()
            .context("local translation requires a selected model")?;
        let descriptor = catalog::MODELS
            .iter()
            .copied()
            .find(|descriptor| descriptor.id == model)
            .with_context(|| format!("unknown local translator '{model}'"))?;
        let resolved = descriptor.resolve(selection).await?;
        let options = LoadOptions {
            mtmd: resolved.projector.map(MtmdOptions::new),
            ..LoadOptions::default()
        };
        let llm = Llm::load_with_options(device, resolved.model, options)
            .await
            .context("failed to load local translation model")?;
        if llm.capabilities().vision != descriptor.projector.is_some() {
            return Err(anyhow::anyhow!(
                "local translator vision capability does not match its catalog"
            )
            .into());
        }
        Ok(Self {
            descriptor,
            llm: Arc::new(llm),
        })
    }

    pub(crate) async fn translate(
        &self,
        request: TranslationRequest,
        generation: GenerationConfig,
    ) -> Result<Translations> {
        let expected = request.segments.len();
        if expected == 0 {
            return Ok(Translations::complete(Vec::new()));
        }
        if !self
            .descriptor
            .target_languages
            .contains(request.target_language)
        {
            return Err(Error::UnsupportedLanguage {
                provider: "local",
                language: request.target_language,
            });
        }

        let images = request.images.clone();
        let prompt = self.render_prompt(
            &request,
            generation.reasoning.unwrap_or(false) && self.descriptor.reasoning,
        )?;
        let schema = prompt::output_schema(expected);
        let llm = Arc::clone(&self.llm);
        let context = self.descriptor.context_length;
        let mut generation = self.descriptor.generation.options(generation);
        generation.max_tokens = output_budget(
            batch::output_tokens(generation.max_tokens, request.page_count),
            batch::prompt_tokens(&prompt, images.len()),
            batch::page_weight(&request.segments, false),
            context,
        )?;
        let output = tokio_rayon::spawn(move || {
            let mut input = Input::new(&prompt);
            for image in &images {
                input = input.with_image(image.image()?);
            }
            llm.inference_with_json_schema(&input, &generation, &schema)
        })
        .await?;
        response(&output, context, &request.segments)
    }

    fn render_prompt(&self, request: &TranslationRequest, reasoning: bool) -> Result<String> {
        let (system, payload) = prompt::prompts(request)?;
        // MTMD pairs every marker with the next attached bitmap, so each image needs its own.
        let payload = format!("{}\n", media_marker()).repeat(request.images.len()) + &payload;
        Ok(self
            .llm
            .render_chat_prompt_with_options(
                &[ChatMessage::system(system), ChatMessage::user(payload)],
                ChatTemplateOptions {
                    add_generation_prompt: true,
                    enable_thinking: reasoning,
                },
            )
            .context("failed to render local translation prompt")?)
    }
}

/// The output budget of a local request: `requested`, clamped to what the model's trained
/// `context` leaves after the `prompt`, because the request's llama.cpp context is sized to hold
/// both. A translation takes about as many tokens as its `source`, so a request that leaves less
/// room than that cannot complete and fails as too large.
fn output_budget(requested: usize, prompt: usize, source: usize, context: usize) -> Result<usize> {
    let available = context.saturating_sub(prompt + 1);
    if available < source {
        return Err(Error::ContextExceeded {
            provider: "local",
            required: prompt + source + 1,
            context,
        });
    }
    Ok(requested.min(available))
}

/// Reads a local response. Its output budget came from an estimated prompt, so a response the
/// exact counts place beyond the trained `context` fails closed, as does JSON cut off at the
/// token limit; a response that ends exactly at the limit can still be complete JSON.
fn response(output: &Generation, context: usize, segments: &[String]) -> Result<Translations> {
    let used = output.prompt_tokens + output.generated_tokens;
    if used > context {
        return Err(Error::ContextExceeded {
            provider: "local",
            required: used,
            context,
        });
    }
    match prompt::translations("local", &output.text, segments) {
        Err(Error::InvalidOutput { .. }) if output.finish_reason == FinishReason::Length => {
            Err(Error::Truncated { provider: "local" })
        }
        translations => translations,
    }
}

pub(crate) fn models() -> Vec<Model> {
    catalog::MODELS
        .iter()
        .map(|descriptor| Model {
            provider: Provider::Local,
            model: Some(descriptor.id.to_owned()),
            name: descriptor.name.to_owned(),
            quantizations: descriptor
                .quantizations
                .iter()
                .map(|quantization| Quantization {
                    id: quantization.id.to_owned(),
                    name: quantization.name.to_owned(),
                    downloaded: std::iter::once(quantization.filename)
                        .chain(descriptor.projector)
                        .all(|filename| descriptor.file(filename).exists()),
                })
                .collect(),
            vision: descriptor.projector.is_some(),
            reasoning: descriptor.reasoning,
        })
        .collect()
}

/// Whether a local failure means the prompt and output budget did not fit the llama.cpp context
/// or its KV cache, as koharu-ml and koharu-llama report it.
pub(crate) fn is_context_overflow(message: &str) -> bool {
    [
        "failed to create llama.cpp context",
        "NoKvCacheSlot",
        "n_ctx is too small",
    ]
    .iter()
    .any(|failure| message.contains(failure))
}

pub(crate) fn supports_vision(selection: &ModelSelection) -> bool {
    selection.model.as_deref().is_some_and(|model| {
        catalog::MODELS
            .iter()
            .find(|descriptor| descriptor.id == model)
            .is_some_and(|descriptor| descriptor.projector.is_some())
    })
}

/// Local requests are sized by the selected model's trained context.
pub(crate) fn batching(selection: &ModelSelection) -> Batching {
    let descriptor = catalog::MODELS
        .iter()
        .find(|descriptor| Some(descriptor.id) == selection.model.as_deref());
    // A model outside the catalog cannot load, so the smallest budget serves it.
    Batching::prompt(descriptor.map_or(0, |descriptor| descriptor.context_length))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn output_budget_keeps_the_prompt_and_output_within_the_trained_context() {
        assert_eq!(output_budget(4_000, 8_000, 2_000, 262_144).unwrap(), 4_000);
        // The source reserve only decides whether the request fits; it never raises a budget.
        assert_eq!(output_budget(500, 8_000, 2_000, 262_144).unwrap(), 500);

        // A request's context holds the prompt, the output budget, and one more position.
        let output = output_budget(32_768, 10_000, 2_000, 32_768).unwrap();
        assert_eq!(10_000 + output + 1, 32_768);
        assert_eq!(
            output_budget(32_768, 10_000, 22_767, 32_768).unwrap(),
            22_767
        );

        assert!(matches!(
            output_budget(32_768, 10_000, 22_768, 32_768),
            Err(Error::ContextExceeded {
                required: 32_769,
                context: 32_768,
                ..
            })
        ));
        assert!(matches!(
            output_budget(1_000, 40_000, 1, 32_768),
            Err(Error::ContextExceeded {
                required: 40_002,
                ..
            })
        ));
    }

    #[test]
    fn a_response_fails_closed_when_cut_off_or_beyond_the_trained_context() {
        let segments = ["こんにちは".to_owned()];
        let read = |text: &str, prompt_tokens, finish_reason| {
            let output = Generation {
                text: text.to_owned(),
                prompt_tokens,
                generated_tokens: 20,
                prompt_duration: Duration::ZERO,
                generation_duration: Duration::ZERO,
                finish_reason,
            };
            response(&output, 1_000, &segments)
        };
        let complete = r#"{"translations":[{"id":0,"text":"Hello"}]}"#;
        let cut = r#"{"translations":[{"id":0,"text":"Hel"#;

        // JSON that closes exactly at the token limit is complete.
        let translated = read(complete, 980, FinishReason::Length).unwrap();
        assert_eq!(translated.segments, ["Hello"]);
        assert!(matches!(
            read(cut, 100, FinishReason::Length),
            Err(Error::Truncated { .. })
        ));
        assert!(matches!(
            read(cut, 100, FinishReason::StopToken),
            Err(Error::InvalidOutput { .. })
        ));
        assert!(matches!(
            read(complete, 981, FinishReason::StopToken),
            Err(Error::ContextExceeded {
                required: 1_001,
                context: 1_000,
                ..
            })
        ));
    }
}
