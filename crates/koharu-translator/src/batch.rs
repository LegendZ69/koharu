//! Sizing facts a caller uses to group pages into translation requests.

use std::ops::RangeInclusive;

/// Output budget cap for one multi-page request, in tokens.
pub(crate) const MAX_BATCH_OUTPUT_TOKENS: usize = 32_768;
/// JSON framing of one input segment, such as `{"id":12,"page":3,"text":""}`, in tokens.
const SEGMENT_TOKENS: usize = 8;
/// Tokens reserved for one attached page image.
const IMAGE_TOKENS: usize = 1_000;
/// Blocks whose characters tokenizers usually encode as about one token each.
const DENSE: &[RangeInclusive<char>] = &[
    '\u{1100}'..='\u{11FF}',   // Hangul Jamo
    '\u{2E80}'..='\u{9FFF}',   // CJK radicals and punctuation, kana, compatibility Jamo, Han
    '\u{A960}'..='\u{A97F}',   // Hangul Jamo Extended-A
    '\u{AC00}'..='\u{D7FF}',   // Hangul syllables and Jamo Extended-B
    '\u{F900}'..='\u{FAFF}',   // CJK compatibility ideographs
    '\u{FE30}'..='\u{FE4F}',   // CJK compatibility forms
    '\u{FF00}'..='\u{FFEF}',   // halfwidth and fullwidth forms
    '\u{20000}'..='\u{3FFFF}', // supplementary ideographic planes
];

/// How a provider's requests grow with the pages they carry. Each provider module declares its
/// own value; the local one follows the selected model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Batching {
    /// Context window of the provider's models, in tokens.
    context_window: usize,
    /// Whether a request is one model prompt, so a multi-page request that is too large can be
    /// split into smaller requests.
    splits: bool,
}

impl Batching {
    /// A language model that receives a whole request as one prompt.
    pub(crate) const fn prompt(context_window: usize) -> Self {
        Self {
            context_window,
            splits: true,
        }
    }

    /// A machine-translation service whose module chunks requests to the service's limits. A
    /// failed request is not split.
    pub(crate) const fn chunked(context_window: usize) -> Self {
        Self {
            context_window,
            splits: false,
        }
    }

    pub(crate) const fn splits(self) -> bool {
        self.splits
    }

    /// Estimated source tokens per request. The rest of the window is left for instructions,
    /// output, and reasoning.
    pub(crate) fn budget(self) -> usize {
        (self.context_window * 3 / 10).clamp(1_024, 8_192)
    }
}

/// A request for several pages may generate the per-page budget for each page, up to
/// [`MAX_BATCH_OUTPUT_TOKENS`], and never less than one page's budget.
pub(crate) fn output_tokens(per_page: usize, pages: usize) -> usize {
    per_page
        .saturating_mul(pages)
        .min(MAX_BATCH_OUTPUT_TOKENS)
        .max(per_page)
}

pub(crate) fn page_weight(segments: &[String], with_image: bool) -> usize {
    let text = segments
        .iter()
        .map(|segment| estimate_tokens(segment) + SEGMENT_TOKENS)
        .sum::<usize>();
    text + if with_image { IMAGE_TOKENS } else { 0 }
}

/// Estimated tokens of a rendered prompt with `images` attached, a quarter above the estimate
/// because tokenizers can split JSON punctuation and some scripts more finely than it assumes.
pub(crate) fn prompt_tokens(prompt: &str, images: usize) -> usize {
    let tokens = estimate_tokens(prompt) + images * IMAGE_TOKENS;
    tokens + tokens / 4
}

/// Approximates a tokenizer without depending on one: Han, kana, Hangul, and fullwidth
/// characters take about one token each, and other text about one token per four characters.
fn estimate_tokens(text: &str) -> usize {
    let (dense, other) = text.chars().fold((0, 0), |(dense, other), character| {
        if DENSE.iter().any(|block| block.contains(&character)) {
            (dense + 1, other)
        } else {
            (dense, other + 1)
        }
    });
    dense + usize::div_ceil(other, 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_dense_scripts_per_character_and_other_text_per_four() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("hello"), 2);
        assert_eq!(estimate_tokens("こんにちは"), 5);
        assert_eq!(estimate_tokens("안녕"), 2);
        assert_eq!(estimate_tokens("漢字！"), 3);
        assert_eq!(estimate_tokens("OK、行こう"), 5);
    }

    #[test]
    fn page_weight_adds_segment_framing_and_an_image_reserve() {
        let segments = ["hello".to_owned(), "こんにちは".to_owned()];
        assert_eq!(page_weight(&segments, false), 23);
        assert_eq!(page_weight(&segments, true), 1_023);
        assert_eq!(page_weight(&[], false), 0);
    }

    #[test]
    fn prompt_estimate_counts_images_and_errs_high() {
        assert_eq!(prompt_tokens("こんにちは", 0), 6);
        assert_eq!(prompt_tokens("こんにちは", 2), 2_506);
    }

    #[test]
    fn budget_is_a_bounded_share_of_the_context_window() {
        assert_eq!(Batching::prompt(1_000).budget(), 1_024);
        assert_eq!(Batching::prompt(8_192).budget(), 2_457);
        assert_eq!(Batching::chunked(1_000_000).budget(), 8_192);
    }
}
