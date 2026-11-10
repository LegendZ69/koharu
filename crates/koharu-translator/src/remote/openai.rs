// https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
// https://developers.openai.com/api/reference/resources/models/methods/list

use anyhow::Context;
use koharu_secrets::ExposeSecret;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{send_json, send_json_sized};
use crate::{
    GenerationConfig, Model, PageImage, Provider, Result, TranslationRequest,
    backend::Translations, batch::Batching, display_name, prompt,
};

const URL: &str = "https://api.openai.com/v1/chat/completions";
const MODELS_URL: &str = "https://api.openai.com/v1/models";
/// Every supported chat family has at least this context window.
pub(super) const BATCHING: Batching = Batching::prompt(128_000);

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct OpenAiConfig {}

pub(super) async fn models(client: &Client) -> Result<Vec<Model>> {
    let Some(api_key) = koharu_secrets::get("openai")? else {
        return Ok(Vec::new());
    };
    let response: ModelsResponse = send_json(
        "openai",
        client.get(MODELS_URL).bearer_auth(api_key.expose_secret()),
    )
    .await?;
    Ok(response
        .data
        .into_iter()
        .filter(|model| supports_translation(&model.id))
        .map(|model| Model {
            provider: Provider::OpenAi,
            name: display_name(&model.id),
            model: Some(model.id),
            quantizations: Vec::new(),
            vision: true,
            reasoning: true,
        })
        .collect())
}

fn supports_translation(id: &str) -> bool {
    let base = id
        .strip_prefix("ft:")
        .and_then(|id| id.split(':').next())
        .unwrap_or(id)
        .to_ascii_lowercase();
    let supported_family = base.starts_with("gpt-5")
        || base.starts_with("gpt-4.1")
        || base.starts_with("gpt-4o")
        || base == "o3"
        || base.starts_with("o3-");
    let incompatible = [
        "-audio",
        "-codex",
        "-deep-research",
        "-image",
        "-pro",
        "-realtime",
        "-search",
        "-transcribe",
        "-tts",
    ]
    .iter()
    .any(|marker| base.contains(marker));
    supported_family && !incompatible
}

pub(super) async fn translate(
    client: &Client,
    _config: &OpenAiConfig,
    model: &str,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<Translations> {
    let api_key = koharu_secrets::get("openai")?.context("openai API key is not configured")?;
    let (system, user) = prompt::prompts(request)?;
    let user_content = MessageContent::user(user, &request.images)?;
    let body = ChatRequest {
        model,
        messages: [
            Message {
                role: "system",
                content: MessageContent::Text(system),
            },
            Message {
                role: "user",
                content: user_content,
            },
        ],
        temperature: generation.temperature,
        top_p: generation.top_p,
        max_completion_tokens: generation.max_tokens,
        frequency_penalty: generation.frequency_penalty,
        presence_penalty: generation.presence_penalty,
        reasoning_effort: generation
            .reasoning
            .map(|enabled| if enabled { "medium" } else { "none" }),
        response_format: ResponseFormat {
            kind: "json_schema",
            json_schema: JsonSchema {
                name: "manga_translation",
                strict: true,
                schema: prompt::output_schema(request.segments.len()),
            },
        },
    };
    let response: ChatResponse = send_json_sized(
        "openai",
        client
            .post(URL)
            .bearer_auth(api_key.expose_secret())
            .json(&body),
        request_too_large,
    )
    .await?;
    let text = response
        .choices
        .into_iter()
        .next()
        .context("OpenAI returned no choices")?
        .message
        .content
        .context("OpenAI returned no message content")?;
    prompt::translations("openai", &text, &request.segments)
}

/// OpenAI answers HTTP 429 "Request too large for … on tokens per min" (or "input-images per
/// min") when one request alone exceeds a per-minute limit, which waiting cannot fix. Ordinary
/// rate limits say "Rate limit reached for …" and quota failures "You exceeded your current quota".
fn request_too_large(body: &str) -> bool {
    body.to_ascii_lowercase().contains("request too large for")
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: [Message<'a>; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
    response_format: ResponseFormat,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    json_schema: JsonSchema,
}

#[derive(Serialize)]
struct JsonSchema {
    name: &'static str,
    strict: bool,
    schema: serde_json::Value,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'static str,
    content: MessageContent<'a>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum MessageContent<'a> {
    Text(String),
    Parts(Vec<ContentPart<'a>>),
}

impl<'a> MessageContent<'a> {
    fn user(text: String, images: &'a [PageImage]) -> anyhow::Result<Self> {
        if images.is_empty() {
            return Ok(Self::Text(text));
        }
        let mut parts = vec![ContentPart::Text { text }];
        for image in images {
            parts.push(ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: image.data_url()?,
                },
            });
        }
        Ok(Self::Parts(parts))
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentPart<'a> {
    Text { text: String },
    ImageUrl { image_url: ImageUrl<'a> },
}

#[derive(Serialize)]
struct ImageUrl<'a> {
    url: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ListedModel>,
}

#[derive(Deserialize)]
struct ListedModel {
    id: String,
}

#[cfg(test)]
mod tests {
    use reqwest::StatusCode;

    use super::*;
    use crate::{Translator, remote::failure};

    #[test]
    fn filters_models_to_the_chat_translation_contract() {
        assert!(supports_translation("gpt-5.6-luna"));
        assert!(supports_translation("gpt-4o-mini"));
        assert!(supports_translation("ft:gpt-4o-mini:org:name:id"));
        assert!(!supports_translation("gpt-5.5-pro"));
        assert!(!supports_translation("gpt-5.3-codex"));
        assert!(!supports_translation("gpt-4o-audio-preview"));
        assert!(!supports_translation("text-embedding-3-large"));
    }

    #[test]
    fn serializes_openai_request_contract() {
        let body = ChatRequest {
            model: "gpt-5.6-luna",
            messages: [
                Message {
                    role: "system",
                    content: MessageContent::Text("system".to_owned()),
                },
                Message {
                    role: "user",
                    content: MessageContent::Text("user".to_owned()),
                },
            ],
            temperature: None,
            top_p: None,
            max_completion_tokens: Some(1024),
            frequency_penalty: None,
            presence_penalty: None,
            reasoning_effort: Some("none"),
            response_format: ResponseFormat {
                kind: "json_schema",
                json_schema: JsonSchema {
                    name: "manga_translation",
                    strict: true,
                    schema: prompt::output_schema(2),
                },
            },
        };
        let value = serde_json::to_value(body).unwrap();

        assert_eq!(value["max_completion_tokens"], 1024);
        assert!(value.get("max_tokens").is_none());
        assert_eq!(value["reasoning_effort"], "none");
        assert_eq!(value["response_format"]["type"], "json_schema");
        assert_eq!(value["response_format"]["json_schema"]["strict"], true);
    }

    #[test]
    fn serializes_text_then_every_page_image_in_order() {
        let request = TranslationRequest::encoded_pages([
            image::DynamicImage::new_rgb8(1, 1),
            image::DynamicImage::new_rgb8(2, 2),
        ]);
        let images = &request.images;
        let content = MessageContent::user("translate".to_owned(), images).unwrap();
        let value = serde_json::to_value(content).unwrap();

        assert_eq!(
            value[0],
            serde_json::json!({ "type": "text", "text": "translate" })
        );
        assert_eq!(value[1]["type"], "image_url");
        assert_eq!(value[1]["image_url"]["url"], images[0].data_url().unwrap());
        assert_eq!(value[2]["image_url"]["url"], images[1].data_url().unwrap());

        let text = MessageContent::user("translate".to_owned(), &[]).unwrap();
        assert_eq!(serde_json::to_value(text).unwrap(), "translate");
    }

    #[test]
    fn only_a_request_over_a_per_minute_limit_is_too_large() {
        let too_large = |message: &str, recognize: fn(&str) -> bool| {
            // The wording arrives inside OpenAI's JSON error envelope.
            let body = serde_json::json!({ "error": { "message": message } }).to_string();
            let error = failure("openai", StatusCode::TOO_MANY_REQUESTS, &body, recognize);
            Translator::is_oversize(&error.into())
        };
        for message in [
            "Request too large for gpt-4o on tokens per min (TPM): Limit 30000, Requested 49634.",
            "Request too large for gpt-4o on input-images per min: Limit 10, Requested 26.",
        ] {
            assert!(too_large(message, request_too_large), "{message}");
            // A provider that does not recognize the wording keeps every 429 a quota failure.
            assert!(!too_large(message, |_| false), "{message}");
        }
        for message in [
            "Rate limit reached for gpt-4o on tokens per min (TPM): Used 28000, Requested 5000.",
            "You exceeded your current quota, please check your plan and billing details.",
        ] {
            assert!(!too_large(message, request_too_large), "{message}");
        }
    }
}
