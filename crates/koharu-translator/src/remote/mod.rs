mod caiyun;
mod claude;
mod deepl;
mod deepseek;
mod gemini;
mod google_cloud;
mod grok;
mod lm_studio;
mod minimax;
mod openai;
mod openai_compatible;
mod openrouter;

use anyhow::Context;
use futures::{FutureExt, future::BoxFuture, future::join_all};
use reqwest::{Client, RequestBuilder, StatusCode};
use serde::de::DeserializeOwned;

pub use caiyun::CaiyunConfig;
pub use claude::ClaudeConfig;
pub use deepl::DeepLConfig;
pub use deepseek::DeepSeekConfig;
pub use gemini::GeminiConfig;
pub use google_cloud::GoogleCloudConfig;
pub use grok::GrokConfig;
pub use lm_studio::LmStudioConfig;
pub use minimax::MiniMaxConfig;
pub use openai::OpenAiConfig;
pub use openai_compatible::OpenAiCompatibleConfig;
pub use openrouter::OpenRouterConfig;

use crate::{
    Error, GenerationConfig, Model, ModelSelection, Provider, ProvidersConfig, Result,
    TranslationRequest, backend::Translations, batch::Batching,
};

pub(crate) async fn translate(
    client: &Client,
    providers: &ProvidersConfig,
    selection: &ModelSelection,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<Translations> {
    let generation = &generation.for_pages(request.page_count);
    let model = || {
        selection
            .model
            .as_deref()
            .with_context(|| format!("{} requires a selected model", selection.provider))
    };
    match selection.provider {
        Provider::OpenAi => {
            openai::translate(client, &providers.openai, model()?, generation, request).await
        }
        Provider::Gemini => {
            gemini::translate(client, &providers.gemini, model()?, generation, request).await
        }
        Provider::Claude => {
            claude::translate(client, &providers.claude, model()?, generation, request).await
        }
        Provider::Grok => {
            grok::translate(client, &providers.grok, model()?, generation, request).await
        }
        Provider::MiniMax => {
            minimax::translate(client, &providers.minimax, model()?, generation, request).await
        }
        Provider::DeepSeek => {
            deepseek::translate(client, &providers.deepseek, model()?, generation, request).await
        }
        Provider::OpenAiCompatible => {
            openai_compatible::translate(
                client,
                &providers.openai_compatible,
                model()?,
                generation,
                request,
            )
            .await
        }
        Provider::OpenRouter => {
            openrouter::translate(client, &providers.openrouter, model()?, generation, request)
                .await
        }
        Provider::LmStudio => {
            lm_studio::translate(client, &providers.lm_studio, model()?, generation, request).await
        }
        Provider::DeepL => deepl::translate(client, &providers.deepl, request)
            .await
            .map(Translations::complete),
        Provider::GoogleCloudTranslation => {
            google_cloud::translate(client, &providers.google_cloud_translation, request)
                .await
                .map(Translations::complete)
        }
        Provider::Caiyun => caiyun::translate(client, &providers.caiyun, request)
            .await
            .map(Translations::complete),
        Provider::Local => unreachable!("local translation has its own backend"),
    }
}

pub(crate) fn batching(provider: Provider) -> Batching {
    match provider {
        Provider::OpenAi => openai::BATCHING,
        Provider::Gemini => gemini::BATCHING,
        Provider::Claude => claude::BATCHING,
        Provider::Grok => grok::BATCHING,
        Provider::MiniMax => minimax::BATCHING,
        Provider::DeepSeek => deepseek::BATCHING,
        Provider::OpenAiCompatible => openai_compatible::BATCHING,
        Provider::OpenRouter => openrouter::BATCHING,
        Provider::LmStudio => lm_studio::BATCHING,
        Provider::DeepL => deepl::BATCHING,
        Provider::GoogleCloudTranslation => google_cloud::BATCHING,
        Provider::Caiyun => caiyun::BATCHING,
        Provider::Local => unreachable!("local translation has its own backend"),
    }
}

/// [`batching`] for `selection`, sized by the context its server reports where it reports one.
pub(crate) async fn reported_batching(
    client: &Client,
    providers: &ProvidersConfig,
    selection: &ModelSelection,
) -> Batching {
    match (selection.provider, selection.model.as_deref()) {
        (Provider::LmStudio, Some(model)) => {
            lm_studio::batching(client, &providers.lm_studio, model).await
        }
        (provider, _) => batching(provider),
    }
}

pub(crate) async fn models(client: &Client, providers: &ProvidersConfig) -> Vec<Model> {
    let mut models = Vec::new();
    let pending: Vec<BoxFuture<'_, Result<Vec<Model>>>> = vec![
        openai::models(client).boxed(),
        gemini::models(client).boxed(),
        claude::models(client).boxed(),
        grok::models(client).boxed(),
        minimax::models(client).boxed(),
        deepseek::models(client).boxed(),
        openai_compatible::models(client, &providers.openai_compatible).boxed(),
        openrouter::models(client).boxed(),
        lm_studio::models(client, &providers.lm_studio).boxed(),
        deepl::models().boxed(),
        google_cloud::models().boxed(),
        caiyun::models().boxed(),
    ];
    for result in join_all(pending).await {
        append_models(&mut models, result);
    }
    models.sort_by(|left, right| {
        left.provider
            .to_string()
            .cmp(&right.provider.to_string())
            .then_with(|| left.model.cmp(&right.model))
    });
    models.dedup_by(|left, right| left.provider == right.provider && left.model == right.model);
    models
}

fn append_models(models: &mut Vec<Model>, result: Result<Vec<Model>>) {
    match result {
        Ok(mut provider_models) => models.append(&mut provider_models),
        Err(error) => tracing::warn!(%error, "failed to list translation models"),
    }
}

pub(super) async fn send_json<T: DeserializeOwned>(
    provider: &'static str,
    request: RequestBuilder,
) -> Result<T> {
    send_json_sized(provider, request, |_| false).await
}

/// Sends a request to a provider whose rate limiter also answers HTTP 429 when one request alone
/// is larger than a limit. `too_large` recognizes that answer from the response body: a smaller
/// request can succeed, so it is a size failure rather than a quota failure.
pub(super) async fn send_json_sized<T: DeserializeOwned>(
    provider: &'static str,
    request: RequestBuilder,
    too_large: fn(&str) -> bool,
) -> Result<T> {
    let response = request
        .send()
        .await
        .with_context(|| format!("{provider} request failed"))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .with_context(|| format!("failed to read {provider} response"))?;

    if !status.is_success() {
        return Err(failure(provider, status, &text, too_large));
    }

    serde_json::from_str(&text)
        .with_context(|| format!("failed to decode {provider} response"))
        .map_err(Into::into)
}

fn failure(
    provider: &'static str,
    status: StatusCode,
    text: &str,
    too_large: fn(&str) -> bool,
) -> Error {
    if status == StatusCode::TOO_MANY_REQUESTS && too_large(text) {
        return Error::RequestTooLarge {
            provider,
            message: text.chars().take(4096).collect(),
        };
    }
    let lower = text.to_ascii_lowercase();
    if status == StatusCode::TOO_MANY_REQUESTS
        || lower.contains("insufficient_quota")
        || lower.contains("resource_exhausted")
        || lower.contains("rate limit exceeded")
        || lower.contains("credit balance is too low")
    {
        return Error::QuotaExceeded { provider };
    }
    Error::Api {
        provider,
        status: status.as_u16(),
        message: text.chars().take(4096).collect(),
    }
}

/// Whether an HTTP failure reports a request beyond a context, token, or size limit. Providers
/// answer 413 for an oversized body, and 400 or 422 with their own wording otherwise.
pub(crate) fn exceeds_size_limit(status: u16, message: &str) -> bool {
    const LIMITS: &[&str] = &[
        "context length",
        "context_length",
        "context size",
        "context_size",
        "context window",
        "maximum context",
        "maximum prompt length",
        "too many tokens",
        "token limit",
        "prompt is too long",
        "input is too long",
        "too large",
        "too many images",
        "payload size",
        "exceeds the maximum",
        "max_tokens",
        "maxoutputtokens",
    ];
    match status {
        413 => true,
        400 | 422 => {
            let message = message.to_ascii_lowercase();
            LIMITS.iter().any(|limit| message.contains(limit))
        }
        _ => false,
    }
}
