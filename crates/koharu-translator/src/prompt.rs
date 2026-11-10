use std::fmt::Write;

use anyhow::Context;
use indoc::indoc;
use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::{Value, json};

use crate::{Error, Language, TranslationContext, TranslationRequest, backend::Translations};

pub(crate) fn prompts(request: &TranslationRequest) -> anyhow::Result<(String, String)> {
    let paged = request.page_count > 1;
    let input = TranslationInput {
        source_language: request.source_language,
        target_language: request.target_language,
        context: &request.context,
        segments: request
            .segments
            .iter()
            .enumerate()
            .map(|(id, text)| TranslationInputSegment {
                id,
                page: request.segment_pages.get(id).copied().filter(|_| paged),
                text,
            })
            .collect(),
    };
    let user = serde_json::to_string(&input).context("failed to serialize translation input")?;
    Ok((translation_system_prompt(request), user))
}

/// Parses a model response. A missing ID keeps its source text and is reported in `missing`.
pub(crate) fn translations(
    provider: &'static str,
    text: &str,
    source_segments: &[String],
) -> crate::Result<Translations> {
    let output =
        serde_json::from_str::<TranslationOutput>(text).map_err(|source| Error::InvalidOutput {
            provider,
            segments: source_segments.len(),
            response: snippet(text),
            source,
        })?;
    let mut segments = source_segments.to_vec();
    let mut translated = vec![false; source_segments.len()];

    for translation in output.translations {
        if translation.id < segments.len() && !translated[translation.id] {
            segments[translation.id] = translation.text;
            translated[translation.id] = true;
        }
    }

    let missing = translated
        .iter()
        .enumerate()
        .filter_map(|(id, translated)| (!translated).then_some(id))
        .collect();
    Ok(Translations { segments, missing })
}

/// Keeps a model response readable in a log line without truncating so hard
/// that the shape of the failure is lost.
///
/// Responses are usually pretty-printed, so line breaks and other control
/// characters are escaped rather than passed through: the snippet is formatted
/// into an error context, and one failure should stay one line.
pub(crate) fn snippet(text: &str) -> String {
    const LIMIT: usize = 2000;
    let trimmed = text.trim();
    let (visible, elided) = match trimmed.char_indices().nth(LIMIT) {
        Some((end, _)) => (&trimmed[..end], true),
        None => (trimmed, false),
    };

    let mut output = String::with_capacity(visible.len());
    for character in visible.chars() {
        match character {
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            control if control.is_control() => {
                let _ = write!(output, "\\u{{{:04x}}}", control as u32);
            }
            character => output.push(character),
        }
    }
    if elided {
        let _ = write!(output, "… ({} bytes total)", trimmed.len());
    }
    output
}

pub(crate) fn output_schema(expected: usize) -> Value {
    json!({
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "minItems": expected,
                "maxItems": expected,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": expected.saturating_sub(1),
                            "description": "The ID copied from the corresponding input segment."
                        },
                        "text": {
                            "type": "string",
                            "description": "The translation of the input segment with this ID."
                        }
                    },
                    "required": ["id", "text"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["translations"],
        "additionalProperties": false
    })
}

fn translation_system_prompt(request: &TranslationRequest) -> String {
    let source = request
        .source_language
        .map(|language| language.to_string())
        .unwrap_or_else(|| "the detected source language".to_owned());
    let mut prompt = format!(
        indoc! {"
            You are a professional manga translator.

            Translation requirements:
            - Translate every input segment from {source} into natural {target}.
            - Preserve meaning, character voice, emotional tone, relationship nuance, emphasis, and sound effects.
            - Localize idioms and sound effects naturally while keeping wording concise enough for speech bubbles.
            - Use surrounding segments only for disambiguation and continuity; never merge or split segments.
            - Write every translated `text` value only in {target}; do not include source text, notes, explanations, or alternatives.
            - Never preserve or repeat original-language text; translate names, terms, and sound effects using natural {target} conventions.

            Output requirements:
            - Each input segment has a numeric `id`.
            - Return only a JSON object whose `translations` array contains one object with `id` and translated `text` for every input segment.
            - Copy every input ID exactly once; order does not matter.
            - Never merge, split, omit, duplicate, or add segments.
        "},
        source = source,
        target = request.target_language,
    )
    .trim_end()
    .to_owned();

    if request.page_count > 1 {
        let pages = format!(
            indoc! {"
                Page requirements:
                The input segments come from {pages} manga pages in reading order; each segment's `page` is its zero-based page index.
                Use surrounding pages only as context; never move text between pages.
            "},
            pages = request.page_count,
        );
        prompt.push_str("\n\n");
        prompt.push_str(pages.trim_end());
    }

    if !request.context.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Context requirements:
            Use the supplied context only to preserve terminology, character voice, and dialogue continuity.
            Do not translate or return the context entries.
        "}.trim_end());
    }

    match request.images.len() {
        0 => {}
        1 => {
            prompt.push_str("\n\n");
            prompt.push_str(indoc! {"
                Image requirements:
                Use the attached original page image as visual context for speaker identity, tone, layout, and ambiguous OCR.
                Translate only the supplied segments; do not add text seen in the image that is absent from the input segments.
            "}.trim_end());
        }
        _ => {
            prompt.push_str("\n\n");
            prompt.push_str(indoc! {"
                Image requirements:
                Use the attached original page images, one per page in page order, as visual context for speaker identity, tone, layout, and ambiguous OCR.
                Translate only the supplied segments; do not add text seen in the images that is absent from the input segments.
            "}.trim_end());
        }
    }

    if let Some(instructions) = request
        .instructions
        .as_deref()
        .map(str::trim)
        .filter(|instructions| !instructions.is_empty())
    {
        prompt.push_str("\n\nAdditional instructions:\n");
        prompt.push_str(instructions);
    }
    prompt
}

#[derive(Serialize)]
struct TranslationInput<'a> {
    source_language: Option<Language>,
    target_language: Language,
    context: &'a [TranslationContext],
    segments: Vec<TranslationInputSegment<'a>>,
}

#[derive(Serialize)]
struct TranslationInputSegment<'a> {
    id: usize,
    /// Set only in a multi-page request, so a single-page payload keeps its shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    page: Option<usize>,
    text: &'a str,
}

#[derive(Debug, Deserialize)]
struct TranslationOutput {
    translations: Vec<TranslationOutputSegment>,
}

#[derive(Debug, Deserialize)]
struct TranslationOutputSegment {
    #[serde(deserialize_with = "deserialize_segment_id")]
    id: usize,
    text: String,
}

fn deserialize_segment_id<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum SegmentId {
        Number(usize),
        String(String),
    }

    match SegmentId::deserialize(deserializer)? {
        SegmentId::Number(id) => Ok(id),
        SegmentId::String(id) => id.trim().parse().map_err(de::Error::custom),
    }
}

#[cfg(test)]
mod tests {
    use image::DynamicImage;

    use super::*;
    use crate::{PageImage, Provider, TranslationPage};

    #[test]
    fn snippet_keeps_a_response_on_one_line() {
        let flattened = snippet("{\n  \"translations\": [\n\t\"hello\"\n  ]\n}");

        assert!(!flattened.contains('\n'), "{flattened}");
        assert!(!flattened.contains('\t'), "{flattened}");
        assert_eq!(flattened, r#"{\n  "translations": [\n\t"hello"\n  ]\n}"#);
    }

    #[test]
    fn snippet_escapes_other_control_characters() {
        assert_eq!(snippet("before\u{7}after"), r"before\u{0007}after");
    }

    #[test]
    fn snippet_reports_the_length_it_elided() {
        let flattened = snippet(&"x".repeat(2_500));

        assert!(flattened.starts_with(&"x".repeat(2_000)));
        assert!(
            flattened.ends_with("\u{2026} (2500 bytes total)"),
            "{flattened}"
        );
    }

    #[test]
    fn parses_plain_json() {
        let source = ["one".to_owned(), "two".to_owned()];
        let response = r#"{"translations":[{"id":0,"text":"hello"},{"id":1,"text":"world"}]}"#;

        assert_eq!(
            translations("test", response, &source).unwrap().segments,
            ["hello", "world"]
        );
    }

    #[test]
    fn rejects_wrapped_and_malformed_json() {
        let source = ["one".to_owned(), "two".to_owned()];
        for response in [
            "```json\n{\"translations\":[{\"id\":0,\"text\":\"hello\"},{\"id\":1,\"text\":\"world\"}]}\n```",
            r#"{translations: [{id: 0, text: 'hello'}, {id: 1, text: 'world'},],}"#,
            r#"Here is the result: {"translations": [{"id": 0, "text": "hello"}, {"id": 1, "text": "world"},]}"#,
            "{\"translations\":[{\"id\":0,\"text\":\"hello\"},{\"id\":1,\"text\":\"world\"",
        ] {
            assert!(
                translations("test", response, &source).is_err(),
                "{response}"
            );
        }
    }

    #[test]
    fn parse_error_preserves_the_root_failure_and_response() {
        let source = ["one".to_owned()];
        let response = "{\n  \"translations\": [{\"id\": 0}]\n}";
        let error = translations("test", response, &source).expect_err("missing text should fail");
        let error = format!("{:#}", anyhow::Error::from(error));

        assert!(error.contains("missing field `text`"), "{error}");
        assert!(
            error.contains(r#"response was: {\n  "translations": [{"id": 0}]\n}"#),
            "{error}"
        );
    }

    #[test]
    fn restores_input_order_from_ids() {
        let source = ["one".to_owned(), "two".to_owned()];
        let response = r#"{"translations":[{"id":1,"text":"world"},{"id":0,"text":"hello"}]}"#;
        assert_eq!(
            translations("test", response, &source).unwrap().segments,
            ["hello", "world"]
        );
    }

    #[test]
    fn reports_missing_ids_and_ignores_duplicate_and_out_of_range_ids() {
        let source = ["one".to_owned(), "two".to_owned()];
        let short = r#"{"translations":[{"id":1,"text":"world"}]}"#;
        let translated = translations("test", short, &source).unwrap();
        assert_eq!(translated.segments, ["one", "world"]);
        assert_eq!(translated.missing, [0]);

        let response = concat!(
            r#"{"translations":["#,
            r#"{"id":0,"text":"hello"},"#,
            r#"{"id":0,"text":"duplicate"},"#,
            r#"{"id":9,"text":"extra"}"#,
            "]}"
        );
        let translated = translations("test", response, &source).unwrap();
        assert_eq!(translated.segments, ["hello", "two"]);
        assert_eq!(translated.missing, [1]);
    }

    #[test]
    fn single_page_payload_has_no_page_fields() {
        let (system, user) = prompts(&TranslationRequest::new(["new"], Language::English)).unwrap();
        assert_eq!(
            user,
            r#"{"source_language":null,"target_language":"en-US","context":[],"segments":[{"id":0,"text":"new"}]}"#
        );
        assert!(!system.contains("Page requirements"));
    }

    #[test]
    fn multi_page_payload_gives_each_segment_its_page() {
        let request = TranslationRequest::pages(
            [
                TranslationPage {
                    segments: vec!["a".to_owned(), "b".to_owned()],
                    image: None,
                },
                TranslationPage {
                    segments: vec!["c".to_owned()],
                    image: None,
                },
            ],
            Language::English,
        );
        let (system, user) = prompts(&request).unwrap();
        let input: Value = serde_json::from_str(&user).unwrap();
        assert_eq!(
            input["segments"],
            json!([
                { "id": 0, "page": 0, "text": "a" },
                { "id": 1, "page": 0, "text": "b" },
                { "id": 2, "page": 1, "text": "c" }
            ])
        );
        assert!(
            system.contains("2 manga pages in reading order"),
            "{system}"
        );
    }

    #[test]
    fn prompt_payload_contains_ordered_context() {
        let request = TranslationRequest::new(["new"], Language::English)
            .with_context([TranslationContext::new("old", "previous")]);
        let (_, user) = prompts(&request).unwrap();
        let input: serde_json::Value = serde_json::from_str(&user).unwrap();
        assert_eq!(input["context"][0]["source"], "old");
        assert_eq!(input["context"][0]["translation"], "previous");
        assert_eq!(input["segments"][0]["id"], 0);
        assert_eq!(input["segments"][0]["text"], "new");
    }

    #[test]
    fn system_prompt_encodes_invariants_and_custom_instructions() {
        let request = TranslationRequest::new(["hello"], Language::Korean)
            .with_source_language(Language::Japanese)
            .with_instructions("Use informal speech.");
        let prompt = translation_system_prompt(&request);
        assert!(prompt.contains("from Japanese into natural Korean"));
        assert!(prompt.contains("Copy every input ID exactly once"));
        assert!(prompt.contains("Use informal speech."));
    }

    #[test]
    fn schema_requires_the_expected_number_of_id_text_pairs() {
        let schema = output_schema(3);
        let translations = &schema["properties"]["translations"];
        assert_eq!(translations["minItems"], 3);
        assert_eq!(translations["maxItems"], 3);
        assert_eq!(translations["items"]["properties"]["id"]["minimum"], 0);
        assert_eq!(translations["items"]["properties"]["id"]["maximum"], 2);
        assert_eq!(translations["items"]["additionalProperties"], false);
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn empty_custom_instructions_are_ignored() {
        let request = TranslationRequest::new(["hello"], Language::English).with_instructions("  ");
        assert!(!translation_system_prompt(&request).contains("Additional instructions"));
    }

    #[test]
    fn context_is_reference_only() {
        let request = TranslationRequest::new(["Where is she?"], Language::Japanese)
            .with_context([TranslationContext::new("I saw Alice.", "アリスを見た。")]);
        let prompt = translation_system_prompt(&request);
        assert!(prompt.contains("dialogue continuity"));
        assert!(prompt.contains("Do not translate or return the context"));
    }

    fn page(text: &str, image: bool) -> TranslationPage {
        TranslationPage {
            segments: vec![text.to_owned()],
            image: image
                .then(|| PageImage::new(DynamicImage::new_rgb8(1, 1), Provider::Local).unwrap()),
        }
    }

    #[test]
    fn image_context_does_not_expand_the_translation_scope() {
        let request = TranslationRequest::pages([page("text", true)], Language::English);
        let prompt = translation_system_prompt(&request);
        assert!(prompt.contains("attached original page image as visual context"));
        assert!(prompt.contains("Translate only the supplied segments"));
    }

    #[test]
    fn several_page_images_are_attached_one_per_page() {
        let request =
            TranslationRequest::pages([page("a", true), page("b", true)], Language::English);
        let prompt = translation_system_prompt(&request);
        assert!(
            prompt.contains("attached original page images, one per page in page order"),
            "{prompt}"
        );
    }
}
