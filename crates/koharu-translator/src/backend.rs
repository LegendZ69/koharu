use std::{io::Cursor, sync::Arc};

use anyhow::{Context, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, ImageEncoder, RgbImage, codecs::jpeg::JpegEncoder};

use crate::{Language, Provider};

const MAX_IMAGE_DIMENSION: u32 = 2048;
const JPEG_QUALITY: u8 = 88;
const DATA_URL_PREFIX: &str = "data:image/jpeg;base64,";

/// One page of a translation request: its segments in reading order and its image.
#[derive(Debug, Clone, PartialEq)]
pub struct TranslationPage {
    pub segments: Vec<String>,
    pub image: Option<PageImage>,
}

/// A page image in the form its provider reads, at the size a request sends. Clones share it, so
/// every request of a job carries the page without copying or encoding it again.
#[derive(Debug, Clone, PartialEq)]
pub struct PageImage(Arc<ImageForm>);

#[derive(Debug, PartialEq)]
enum ImageForm {
    /// The pixels a local model reads.
    Pixels(DynamicImage),
    /// The JPEG data URL a hosted provider sends; its base64 payload follows [`DATA_URL_PREFIX`].
    Jpeg(String),
}

impl PageImage {
    /// Scales `image` down to the largest dimension a request sends; an image that already fits
    /// is kept as is. For a hosted `provider` only the JPEG encoding is kept, so a batch does not
    /// hold every page's pixels. This is CPU-bound work for the caller to run off the async
    /// thread.
    pub fn new(image: DynamicImage, provider: Provider) -> anyhow::Result<Self> {
        let image = downscale(&image)?.unwrap_or(image);
        let form = if provider == Provider::Local {
            ImageForm::Pixels(image)
        } else {
            ImageForm::Jpeg(data_url(&image)?)
        };
        Ok(Self(Arc::new(form)))
    }

    pub(crate) fn image(&self) -> anyhow::Result<&DynamicImage> {
        match &*self.0 {
            ImageForm::Pixels(image) => Ok(image),
            ImageForm::Jpeg(_) => bail!("translation image was prepared for a hosted provider"),
        }
    }

    pub(crate) fn data_url(&self) -> anyhow::Result<&str> {
        match &*self.0 {
            ImageForm::Jpeg(data_url) => Ok(data_url),
            ImageForm::Pixels(_) => bail!("translation image was prepared for a local model"),
        }
    }

    /// The base64 JPEG payload of [`Self::data_url`].
    pub(crate) fn base64(&self) -> anyhow::Result<&str> {
        Ok(&self.data_url()?[DATA_URL_PREFIX.len()..])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationRequest {
    /// Segments in page order, then reading order within a page.
    pub segments: Vec<String>,
    /// Zero-based page index of each segment.
    pub segment_pages: Vec<usize>,
    pub page_count: usize,
    /// Page images in page order; a page without an image contributes none.
    pub images: Vec<PageImage>,
    pub source_language: Option<Language>,
    pub target_language: Language,
    pub instructions: Option<String>,
    pub context: Vec<TranslationContext>,
}

impl TranslationRequest {
    /// A request for one page without an image.
    #[must_use]
    pub fn new(
        segments: impl IntoIterator<Item = impl Into<String>>,
        target_language: Language,
    ) -> Self {
        Self::pages(
            [TranslationPage {
                segments: segments.into_iter().map(Into::into).collect(),
                image: None,
            }],
            target_language,
        )
    }

    /// A request for pages in reading order. They need not be adjacent in their project.
    #[must_use]
    pub fn pages(
        pages: impl IntoIterator<Item = TranslationPage>,
        target_language: Language,
    ) -> Self {
        let mut request = Self {
            segments: Vec::new(),
            segment_pages: Vec::new(),
            page_count: 0,
            images: Vec::new(),
            source_language: None,
            target_language,
            instructions: None,
            context: Vec::new(),
        };
        for TranslationPage { segments, image } in pages {
            request
                .segment_pages
                .extend(std::iter::repeat_n(request.page_count, segments.len()));
            request.segments.extend(segments);
            request.images.extend(image);
            request.page_count += 1;
        }
        request
    }

    /// A request for one page per image, encoded as hosted providers send it.
    #[cfg(test)]
    pub(crate) fn encoded_pages(images: impl IntoIterator<Item = DynamicImage>) -> Self {
        Self::pages(
            images.into_iter().map(|image| TranslationPage {
                segments: vec!["text".to_owned()],
                image: Some(PageImage::new(image, Provider::OpenAi).unwrap()),
            }),
            Language::English,
        )
    }

    #[cfg(test)]
    #[must_use]
    pub fn with_source_language(mut self, language: Language) -> Self {
        self.source_language = Some(language);
        self
    }

    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    #[cfg(test)]
    #[must_use]
    pub fn with_context(mut self, context: impl IntoIterator<Item = TranslationContext>) -> Self {
        self.context = context.into_iter().collect();
        self
    }

    pub(crate) fn remove_images(&mut self) {
        self.images.clear();
    }
}

fn downscale(image: &DynamicImage) -> anyhow::Result<Option<DynamicImage>> {
    let (width, height) = (image.width(), image.height());
    let longest = width.max(height);
    if longest <= MAX_IMAGE_DIMENSION {
        return Ok(None);
    }
    let resized_width =
        (u64::from(width) * u64::from(MAX_IMAGE_DIMENSION) / u64::from(longest)).max(1) as u32;
    let resized_height =
        (u64::from(height) * u64::from(MAX_IMAGE_DIMENSION) / u64::from(longest)).max(1) as u32;
    let source = image.to_rgb8();
    let mut resized = RgbImage::new(resized_width, resized_height);
    Resizer::new()
        .resize(
            &source,
            &mut resized,
            &ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                .use_alpha(false),
        )
        .context("failed to resize translation image")?;
    Ok(Some(DynamicImage::ImageRgb8(resized)))
}

/// Translations of every request segment, in request order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Translated {
    pub provider: &'static str,
    pub segments: Vec<String>,
    /// Segment indexes a language model left out of its response; they hold the source text.
    pub missing: Vec<usize>,
}

/// What one provider call returns; [`crate::Translator`] adds the provider.
#[derive(Debug)]
pub(crate) struct Translations {
    pub(crate) segments: Vec<String>,
    pub(crate) missing: Vec<usize>,
}

impl Translations {
    /// The result of a provider that returns every segment or fails.
    pub(crate) fn complete(segments: Vec<String>) -> Self {
        Self {
            segments,
            missing: Vec::new(),
        }
    }
}

fn data_url(image: &DynamicImage) -> anyhow::Result<String> {
    let rgb = image.to_rgb8();
    let mut bytes = Cursor::new(Vec::new());
    JpegEncoder::new_with_quality(&mut bytes, JPEG_QUALITY)
        .write_image(
            rgb.as_raw(),
            rgb.width(),
            rgb.height(),
            image::ExtendedColorType::Rgb8,
        )
        .context("failed to encode translation image")?;
    Ok([DATA_URL_PREFIX, &STANDARD.encode(bytes.into_inner())].concat())
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TranslationContext {
    pub source: String,
    pub translation: String,
}

impl TranslationContext {
    #[cfg(test)]
    #[must_use]
    pub fn new(source: impl Into<String>, translation: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            translation: translation.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hosted_page_keeps_only_the_jpeg_of_its_scaled_pixels() {
        let local = PageImage::new(DynamicImage::new_luma8(4_096, 8), Provider::Local).unwrap();
        let fitted = local.image().unwrap();
        assert_eq!((fitted.width(), fitted.height()), (MAX_IMAGE_DIMENSION, 4));

        let hosted = PageImage::new(DynamicImage::new_luma8(4_096, 8), Provider::Claude).unwrap();
        assert!(hosted.image().is_err());
        assert_eq!(hosted.data_url().unwrap(), data_url(fitted).unwrap());
        assert_eq!(
            hosted.data_url().unwrap().strip_prefix(DATA_URL_PREFIX),
            Some(hosted.base64().unwrap())
        );

        // Every request carrying the page, such as half of a split batch, shares its encoding.
        let page = TranslationPage {
            segments: vec!["a".to_owned()],
            image: Some(hosted),
        };
        let whole = TranslationRequest::pages([page.clone(), page.clone()], Language::English);
        let half = TranslationRequest::pages([page], Language::English);
        assert!(std::ptr::eq(
            half.images[0].data_url().unwrap(),
            whole.images[1].data_url().unwrap()
        ));
    }
}
