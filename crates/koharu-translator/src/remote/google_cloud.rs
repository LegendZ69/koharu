// Ported from:
// https://github.com/koharu-rs/koharu/blob/f4ce03999ed1ae2faaec938dd52c2f41a87d03d9/crates/koharu-llm/src/providers/google_translate.rs

use anyhow::Context;
use koharu_secrets::ExposeSecret;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use url::Url;

use super::send_json;
use crate::{Error, Model, Provider, Result, TranslationRequest, batch::Batching};

const URL: &str = "https://translation.googleapis.com/language/translate/v2";
/// Requests are chunked below, so this window only sizes page batches.
pub(super) const BATCHING: Batching = Batching::chunked(128_000);
/// Cloud Translation v2 accepts at most 128 `q` strings per request.
const MAX_QUERIES: usize = 128;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct GoogleCloudConfig {}

pub(super) async fn models() -> Result<Vec<Model>> {
    Ok(
        if koharu_secrets::get("google-cloud-translation")?.is_some() {
            vec![Model::service(
                Provider::GoogleCloudTranslation,
                "Google Cloud Translation",
            )]
        } else {
            Vec::new()
        },
    )
}

pub(super) async fn translate(
    client: &Client,
    _config: &GoogleCloudConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let api_key = koharu_secrets::get("google-cloud-translation")?
        .context("google-cloud-translation API key is not configured")?;
    let mut url = Url::parse(URL).expect("Google API URL is valid");
    url.query_pairs_mut()
        .append_pair("key", api_key.expose_secret());
    let mut translations = Vec::with_capacity(request.segments.len());
    for body in requests(request) {
        let response: Response = send_json(
            "google-cloud-translation",
            client.post(url.clone()).json(&body),
        )
        .await?;
        let chunk = response.data.translations;
        if chunk.len() != body.q.len() {
            return Err(Error::SegmentCount {
                provider: "google-cloud-translation",
                expected: body.q.len(),
                actual: chunk.len(),
            });
        }
        translations.extend(
            chunk
                .into_iter()
                .map(|translation| translation.translated_text),
        );
    }
    Ok(translations)
}

/// One request body per [`MAX_QUERIES`] segments.
fn requests(request: &TranslationRequest) -> impl Iterator<Item = Request<'_>> {
    request.segments.chunks(MAX_QUERIES).map(|q| Request {
        q,
        target: request.target_language.tag(),
        source: request.source_language.map(|language| language.tag()),
        format: "text",
    })
}

#[derive(Serialize)]
struct Request<'a> {
    q: &'a [String],
    target: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<&'a str>,
    format: &'static str,
}

#[derive(Deserialize)]
struct Response {
    data: Data,
}

#[derive(Deserialize)]
struct Data {
    translations: Vec<Translation>,
}

#[derive(Deserialize)]
struct Translation {
    #[serde(rename = "translatedText")]
    translated_text: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Language;

    #[test]
    fn splits_segments_into_requests_of_at_most_128() {
        let segments = (0..300).map(|index| index.to_string());
        let request = TranslationRequest::new(segments, Language::English);
        let bodies: Vec<_> = requests(&request).collect();

        let sizes: Vec<_> = bodies.iter().map(|body| body.q.len()).collect();
        assert_eq!(sizes, [128, 128, 44]);
        let value = serde_json::to_value(&bodies[2]).unwrap();
        assert_eq!(value["q"][0], "256");
        assert_eq!(value["target"], "en-US");
        assert_eq!(value["format"], "text");
        assert!(value.get("source").is_none());
    }
}
