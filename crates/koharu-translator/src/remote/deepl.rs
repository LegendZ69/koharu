// Ported from:
// https://github.com/koharu-rs/koharu/blob/f4ce03999ed1ae2faaec938dd52c2f41a87d03d9/crates/koharu-llm/src/providers/deepl.rs

use anyhow::Context;
use koharu_secrets::ExposeSecret;
use reqwest::Client;
use serde::Deserialize;
use url::Url;

use super::send_json;
use crate::{Error, Language, Model, Provider, Result, TranslationRequest, batch::Batching};

/// Requests are chunked below, so this window only sizes page batches.
pub(super) const BATCHING: Batching = Batching::chunked(128_000);
/// DeepL accepts at most 50 `text` values per request.
const MAX_TEXTS: usize = 50;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct DeepLConfig {
    pub base_url: Option<Url>,
}

pub(super) async fn models() -> Result<Vec<Model>> {
    Ok(if koharu_secrets::get("deepl")?.is_some() {
        vec![Model::service(Provider::DeepL, "DeepL")]
    } else {
        Vec::new()
    })
}

pub(super) async fn translate(
    client: &Client,
    config: &DeepLConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let api_key = koharu_secrets::get("deepl")?.context("deepl API key is not configured")?;
    let target = target(request.target_language).ok_or(Error::UnsupportedLanguage {
        provider: "deepl",
        language: request.target_language,
    })?;
    let source = request
        .source_language
        .map(|language| {
            source(language).ok_or(Error::UnsupportedSourceLanguage {
                provider: "deepl",
                language,
            })
        })
        .transpose()?;
    let root = config.base_url.as_ref().map_or_else(
        || {
            if api_key.expose_secret().trim_end().ends_with(":fx") {
                "https://api-free.deepl.com".to_owned()
            } else {
                "https://api.deepl.com".to_owned()
            }
        },
        |url| url.as_str().trim_end_matches('/').to_owned(),
    );
    // DeepL accepts source-language context but does not translate it.
    let context = request
        .context
        .iter()
        .map(|entry| entry.source.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let mut translations = Vec::with_capacity(request.segments.len());
    for (texts, form) in forms(&request.segments, target, source, &context) {
        let response: Response = send_json(
            "deepl",
            client
                .post(format!("{root}/v2/translate"))
                .header(
                    "Authorization",
                    format!("DeepL-Auth-Key {}", api_key.expose_secret()),
                )
                .form(&form),
        )
        .await?;
        if response.translations.len() != texts.len() {
            return Err(Error::SegmentCount {
                provider: "deepl",
                expected: texts.len(),
                actual: response.translations.len(),
            });
        }
        translations.extend(
            response
                .translations
                .into_iter()
                .map(|translation| translation.text),
        );
    }
    Ok(translations)
}

/// One form per request, each with at most [`MAX_TEXTS`] texts and the shared options.
fn forms<'a>(
    segments: &'a [String],
    target: &'a str,
    source: Option<&'a str>,
    context: &'a str,
) -> impl Iterator<Item = (&'a [String], Vec<(&'static str, &'a str)>)> {
    segments.chunks(MAX_TEXTS).map(move |texts| {
        let mut form = texts
            .iter()
            .map(|text| ("text", text.as_str()))
            .collect::<Vec<_>>();
        form.push(("target_lang", target));
        if let Some(source) = source {
            form.push(("source_lang", source));
        }
        if !context.is_empty() {
            form.push(("context", context));
        }
        (texts, form)
    })
}

#[derive(Deserialize)]
struct Response {
    translations: Vec<Translation>,
}

#[derive(Deserialize)]
struct Translation {
    text: String,
}

fn target(language: Language) -> Option<&'static str> {
    use Language::*;
    Some(match language {
        ChineseSimplified => "ZH-HANS",
        ChineseTraditional => "ZH-HANT",
        English => "EN-US",
        French => "FR",
        Portuguese => "PT-PT",
        BrazilianPortuguese => "PT-BR",
        Spanish => "ES",
        Japanese => "JA",
        Turkish => "TR",
        Russian => "RU",
        Arabic => "AR",
        Korean => "KO",
        Thai => "TH",
        Italian => "IT",
        German => "DE",
        Vietnamese => "VI",
        Malay => "MS",
        Indonesian => "ID",
        Hindi => "HI",
        Polish => "PL",
        Czech => "CS",
        Dutch => "NL",
        Gujarati => "GU",
        Urdu => "UR",
        Telugu => "TE",
        Marathi => "MR",
        Hebrew => "HE",
        Bengali => "BN",
        Bulgarian => "BG",
        Tamil => "TA",
        Ukrainian => "UK",
        Kazakh => "KK",
        Belarusian => "BE",
        Hungarian => "HU",
        Filipino | Khmer | Burmese | Persian | Tibetan | Mongolian | Uyghur | Cantonese => {
            return None;
        }
    })
}

fn source(language: Language) -> Option<&'static str> {
    use Language::*;
    Some(match language {
        ChineseSimplified | ChineseTraditional => "ZH",
        English => "EN",
        Portuguese | BrazilianPortuguese => "PT",
        language => target(language)?.split('-').next()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_languages_are_not_substituted() {
        assert_eq!(target(Language::Filipino), None);
        assert_eq!(target(Language::English), Some("EN-US"));
    }

    #[test]
    fn splits_texts_into_requests_of_at_most_fifty() {
        let segments: Vec<_> = (0..120).map(|index| index.to_string()).collect();
        let forms: Vec<_> = forms(&segments, "EN-US", Some("JA"), "").collect();

        let sizes: Vec<_> = forms.iter().map(|(texts, _)| texts.len()).collect();
        assert_eq!(sizes, [50, 50, 20]);
        let (texts, form) = &forms[2];
        assert_eq!(texts[0], "100");
        assert_eq!(form.len(), 22);
        assert_eq!(form[0], ("text", "100"));
        assert_eq!(form[19], ("text", "119"));
        assert_eq!(form[20], ("target_lang", "EN-US"));
        assert_eq!(form[21], ("source_lang", "JA"));
    }
}
