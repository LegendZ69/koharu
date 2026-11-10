// Ported from:
// https://github.com/koharu-rs/koharu/blob/f4ce03999ed1ae2faaec938dd52c2f41a87d03d9/crates/koharu-llm/src/providers/caiyun.rs

use anyhow::{Context, anyhow};
use koharu_secrets::ExposeSecret;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::send_json;
use crate::{Error, Language, Model, Provider, Result, TranslationRequest, batch::Batching};

const URL: &str = "https://api.interpreter.caiyunai.com/v1/translator";
/// Requests are chunked below, so this window only sizes page batches.
pub(super) const BATCHING: Batching = Batching::chunked(128_000);
/// A conservative number of sources per request.
const MAX_SOURCES: usize = 50;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct CaiyunConfig {}

pub(super) async fn models() -> Result<Vec<Model>> {
    Ok(if koharu_secrets::get("caiyun")?.is_some() {
        vec![Model::service(Provider::Caiyun, "Caiyun")]
    } else {
        Vec::new()
    })
}

pub(super) async fn translate(
    client: &Client,
    _config: &CaiyunConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let api_key = koharu_secrets::get("caiyun")?.context("caiyun API key is not configured")?;
    let target = target(request.target_language).ok_or(Error::UnsupportedLanguage {
        provider: "caiyun",
        language: request.target_language,
    })?;
    let trans_type = format!("auto2{target}");
    let mut translations = Vec::with_capacity(request.segments.len());
    for source in request.segments.chunks(MAX_SOURCES) {
        let response: Response = send_json(
            "caiyun",
            client
                .post(URL)
                .header(
                    "X-Authorization",
                    format!("token {}", api_key.expose_secret()),
                )
                .json(&Request {
                    source,
                    trans_type: &trans_type,
                    request_id: "koharu-translator",
                    detect: true,
                    media: "text",
                }),
        )
        .await?;
        if response.rc != 0 {
            return Err(anyhow!(
                "Caiyun returned rc={}: {}",
                response.rc,
                response
                    .msg
                    .or(response.message)
                    .or(response.error)
                    .unwrap_or_else(|| "unknown error".to_owned())
            )
            .into());
        }
        let chunk = match response.target.context("Caiyun returned no target")? {
            Target::One(text) => vec![text],
            Target::Many(texts) => texts,
        };
        if chunk.len() != source.len() {
            return Err(Error::SegmentCount {
                provider: "caiyun",
                expected: source.len(),
                actual: chunk.len(),
            });
        }
        translations.extend(chunk);
    }
    Ok(translations)
}

#[derive(Serialize)]
struct Request<'a> {
    source: &'a [String],
    trans_type: &'a str,
    request_id: &'static str,
    detect: bool,
    media: &'static str,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Target {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    rc: i64,
    target: Option<Target>,
    msg: Option<String>,
    message: Option<String>,
    error: Option<String>,
}

fn target(language: Language) -> Option<&'static str> {
    use Language::*;
    Some(match language {
        ChineseSimplified => "zh",
        English => "en",
        French => "fr",
        Portuguese => "pt",
        Spanish => "es",
        Japanese => "ja",
        Turkish => "tr",
        Russian => "ru",
        Arabic => "ar",
        Korean => "ko",
        Thai => "th",
        Italian => "it",
        German => "de",
        Vietnamese => "vi",
        Indonesian => "id",
        ChineseTraditional => "zh-Hant",
        Polish => "pl",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_languages_are_not_substituted() {
        assert_eq!(target(Language::BrazilianPortuguese), None);
        assert_eq!(target(Language::Japanese), Some("ja"));
    }
}
