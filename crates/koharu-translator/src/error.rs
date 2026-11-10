use thiserror::Error;

use crate::{Language, local, remote};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{provider} does not support target language {language}")]
    UnsupportedLanguage {
        provider: &'static str,
        language: Language,
    },
    #[error("{provider} does not support source language {language}")]
    UnsupportedSourceLanguage {
        provider: &'static str,
        language: Language,
    },
    #[error("{provider} returned {actual} segments; expected {expected}")]
    SegmentCount {
        provider: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error(
        "{provider} returned invalid translation JSON for {segments} segments; response was: {response}"
    )]
    InvalidOutput {
        provider: &'static str,
        segments: usize,
        response: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("{provider} reached its output token limit before completing the translation")]
    Truncated { provider: &'static str },
    #[error("{provider} request needs {required} context tokens, more than the model's {context}")]
    ContextExceeded {
        provider: &'static str,
        required: usize,
        context: usize,
    },
    #[error("{provider} rejected a request that alone exceeds its rate limit: {message}")]
    RequestTooLarge {
        provider: &'static str,
        message: String,
    },
    #[error("{provider} quota or rate limit was exceeded")]
    QuotaExceeded { provider: &'static str },
    #[error("{provider} API request failed with HTTP {status}: {message}")]
    Api {
        provider: &'static str,
        status: u16,
        message: String,
    },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl Error {
    /// Whether the request was too large for the provider or model, so a request with fewer
    /// pages can succeed.
    pub(crate) fn is_oversize(&self) -> bool {
        match self {
            Self::InvalidOutput { .. }
            | Self::Truncated { .. }
            | Self::ContextExceeded { .. }
            | Self::RequestTooLarge { .. } => true,
            Self::Api {
                status, message, ..
            } => remote::exceeds_size_limit(*status, message),
            // Transparent: the wrapped causes follow this one in an error chain.
            Self::Other(error) => local::is_context_overflow(&error.to_string()),
            Self::UnsupportedLanguage { .. }
            | Self::UnsupportedSourceLanguage { .. }
            | Self::SegmentCount { .. }
            | Self::QuotaExceeded { .. } => false,
        }
    }
}
