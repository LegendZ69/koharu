//! Native page preparation shared by browser presentation and export.

use std::{collections::HashSet, future::Future, sync::Arc};

use anyhow::{Context as _, Result, bail};
use koharu_rasterizer::{Rasterizer, ResourceId};
use koharu_renderer::{Frame as RenderedFrame, LayerKind, Renderer};
use koharu_scene::{Commit, EntityId, Geometry, Point, ProjectId, Revision, Snapshot};
use parking_lot::{Mutex as SyncMutex, RwLock};
use serde::{Deserialize, Serialize};
use specta::Type;
use tokio::sync::{Mutex, Notify, OnceCell};

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize, Type)]
pub struct Frame {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub angle_degrees: f32,
}

impl Frame {
    fn is_valid(self) -> bool {
        [self.x, self.y, self.width, self.height, self.angle_degrees]
            .into_iter()
            .all(f32::is_finite)
            && self.width > 0.0
            && self.height > 0.0
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Type)]
pub struct TransformFrame {
    pub element: EntityId,
    pub frame: Frame,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct CanvasState {
    pub page: Option<EntityId>,
    pub revision: Option<Revision>,
    #[specta(type = f64)]
    pub generation: u64,
    pub size: [u32; 2],
    pub element_frames: Vec<TransformFrame>,
}

#[derive(Default)]
struct PresentationState {
    generation: u64,
    frame: Option<RenderedFrame>,
    requested_presentation: u64,
    requested_prefetch: u64,
    claimed: Option<(ProjectId, Revision)>,
}

/// Where a presentation request sits in the order of its project's state.
#[derive(Clone, Copy)]
enum Claim {
    /// A project the window just opened; it supersedes everything presented before.
    Project(ProjectId, Revision),
    /// State observed after its revision was committed, such as a page selection.
    Selection(ProjectId, Revision),
    /// The commit that created a revision, which is the earliest state at that revision.
    Commit(ProjectId, Revision),
    Clear,
}

#[derive(Clone, Copy)]
struct Ticket {
    presentation: u64,
    prefetch: Option<u64>,
}

impl Ticket {
    fn is_current(self, state: &PresentationState) -> bool {
        state.requested_presentation == self.presentation
            && self
                .prefetch
                .is_none_or(|prefetch| state.requested_prefetch == prefetch)
    }
}

const MAX_CACHED_PAGE_FRAMES: usize = 8;

#[derive(Default)]
struct PageFrameCache {
    clock: u64,
    entries: Vec<CachedPageFrame>,
}

struct CachedPageFrame {
    page: EntityId,
    revision: Revision,
    frame: RenderedFrame,
    last_used: u64,
}

/// Rendering resources shared by every window.
pub struct Desktop {
    renderer: Renderer,
    rasterizer: OnceCell<Arc<Rasterizer>>,
}

impl Desktop {
    pub fn new() -> Result<Self> {
        Ok(Self {
            renderer: Renderer::new()?,
            rasterizer: OnceCell::new(),
        })
    }

    #[must_use]
    pub fn renderer(&self) -> Renderer {
        self.renderer.clone()
    }

    pub async fn rasterizer(&self) -> Result<Arc<Rasterizer>> {
        self.rasterizer
            .get_or_try_init(|| async {
                let rasterizer = tokio_rayon::spawn(Rasterizer::new).await?;
                Ok::<_, anyhow::Error>(Arc::new(rasterizer))
            })
            .await
            .cloned()
    }

    /// Creates one window's presentation over the shared renderer.
    #[must_use]
    pub fn presentation(&self) -> Presentation {
        Presentation {
            renderer: self.renderer.clone(),
            state: RwLock::new(PresentationState::default()),
            page_frames: SyncMutex::new(PageFrameCache::default()),
            preparation: Mutex::new(()),
            preparation_changed: Notify::new(),
        }
    }
}

/// One window's presented frame, frame generation, and prepared page frames.
///
/// Callers read project state under their project lock and present it after releasing that
/// lock, so requests can arrive out of order. A presentation is bound to one project at a time
/// and orders that project's requests by revision: revisions only grow, and the commit that
/// creates a revision is the earliest state at it. A request for older state is rejected before
/// it can supersede newer work, and a prefetch never supersedes a presentation request.
pub struct Presentation {
    renderer: Renderer,
    state: RwLock<PresentationState>,
    page_frames: SyncMutex<PageFrameCache>,
    preparation: Mutex<()>,
    preparation_changed: Notify,
}

impl Presentation {
    pub fn frame_manifest_bytes(&self, generation: u64) -> Result<Vec<u8>> {
        let frame = self.frame_at_generation(generation)?;
        frame.prepared().manifest()?.encode().map_err(Into::into)
    }

    pub fn frame_resource_bytes(&self, generation: u64, resource: ResourceId) -> Result<Vec<u8>> {
        self.frame_at_generation(generation)?
            .prepared()
            .resource_packet(resource)?
            .encode()
            .map_err(Into::into)
    }

    pub async fn prepare_page(&self, snapshot: &Snapshot, page: EntityId) -> Result<bool> {
        let Some(request) = self.claim_prefetch(snapshot.project_id(), snapshot.revision()) else {
            return Ok(false);
        };
        let _preparation = self.preparation.lock().await;
        if !self.is_current(request) {
            return Ok(false);
        }
        let revision = snapshot.revision();
        let cached = self.page_frames.lock().get(page, revision);
        if cached.is_some() {
            return Ok(true);
        }
        let Some(frame) = self
            .await_preparation(request, self.renderer.render(snapshot, page))
            .await
        else {
            return Ok(false);
        };
        let frame = frame?;
        if !self.is_current(request) {
            return Ok(false);
        }
        self.page_frames.lock().insert(frame);
        Ok(true)
    }

    pub fn page_manifest_bytes(&self, page: EntityId, revision: Revision) -> Result<Vec<u8>> {
        self.page_frame(page, revision)?
            .prepared()
            .manifest()?
            .encode()
            .map_err(Into::into)
    }

    pub fn page_resource_bytes(
        &self,
        page: EntityId,
        revision: Revision,
        resource: ResourceId,
    ) -> Result<Vec<u8>> {
        self.page_frame(page, revision)?
            .prepared()
            .resource_packet(resource)?
            .encode()
            .map_err(Into::into)
    }

    fn frame_at_generation(&self, generation: u64) -> Result<RenderedFrame> {
        let presentation = self.state.read();
        if presentation.generation != generation {
            bail!(
                "canvas generation {generation} is stale; current generation is {}",
                presentation.generation
            );
        }
        Ok(presentation
            .frame
            .as_ref()
            .context("there is no prepared canvas frame")?
            .clone())
    }

    fn page_frame(&self, page: EntityId, revision: Revision) -> Result<RenderedFrame> {
        self.page_frames
            .lock()
            .get(page, revision)
            .with_context(|| format!("canvas page {page} at revision {revision} is not prepared"))
    }

    /// Whether this presentation is bound to `project`.
    #[must_use]
    pub fn presents(&self, project: ProjectId) -> bool {
        self.state
            .read()
            .claimed
            .is_some_and(|(claimed, _)| claimed == project)
    }

    #[must_use]
    pub fn canvas_state(&self) -> CanvasState {
        let presentation = self.state.read();
        let Some(frame) = presentation.frame.as_ref() else {
            return CanvasState {
                page: None,
                revision: None,
                generation: presentation.generation,
                size: [0, 0],
                element_frames: Vec::new(),
            };
        };
        let (width, height) = frame.size();
        let element_frames = frame
            .layers()
            .iter()
            .filter(|layer| layer.entity() != frame.page())
            .filter(|layer| {
                let presentation = layer.presentation();
                presentation.visible && presentation.opacity > 0.0
            })
            .filter_map(|layer| {
                let value = layer.element_frame()?;
                Some(TransformFrame {
                    element: layer.entity(),
                    frame: Frame {
                        x: value.x,
                        y: value.y,
                        width: value.width,
                        height: value.height,
                        angle_degrees: value.angle_degrees,
                    },
                })
            })
            .collect();
        CanvasState {
            page: Some(frame.page()),
            revision: Some(frame.revision()),
            generation: presentation.generation,
            size: [width, height],
            element_frames,
        }
    }

    pub fn transform_geometries(
        &self,
        expected_revision: Revision,
        elements: &[TransformFrame],
    ) -> Result<Vec<(EntityId, Geometry, f32)>> {
        let presentation = self.state.read();
        let rendered = presentation
            .frame
            .as_ref()
            .context("there is no prepared canvas frame")?;
        if rendered.revision() != expected_revision {
            bail!(
                "canvas transform expected revision {expected_revision}, but the prepared frame is at {}",
                rendered.revision()
            );
        }
        let mut seen = HashSet::with_capacity(elements.len());
        let mut geometries = Vec::with_capacity(elements.len());
        for element in elements {
            if !seen.insert(element.element) {
                bail!("canvas transform repeats element {}", element.element);
            }
            if !element.frame.is_valid() {
                bail!("canvas transform frame must be finite and non-empty");
            }
            let layer = rendered.layer(element.element).with_context(|| {
                format!(
                    "canvas transform element {} is not rendered",
                    element.element
                )
            })?;
            let layer_presentation = layer.presentation();
            if !layer_presentation.visible
                || layer_presentation.opacity <= 0.0
                || !matches!(layer.kind(), LayerKind::Text(_))
            {
                bail!(
                    "canvas transform element {} is not selectable",
                    element.element
                );
            }
            let original = layer
                .element_frame()
                .context("canvas transform element has no control frame")?;
            let original = Frame {
                x: original.x,
                y: original.y,
                width: original.width,
                height: original.height,
                angle_degrees: original.angle_degrees,
            };
            let coefficients = frame_transform(original, element.frame);
            let geometry = Geometry {
                origin: layer.geometry().origin.clone(),
                points: layer
                    .geometry()
                    .points
                    .iter()
                    .map(|point| transform_point(coefficients, point))
                    .collect(),
            };
            if geometry != *layer.geometry()
                || element.frame.angle_degrees != original.angle_degrees
            {
                geometries.push((element.element, geometry, element.frame.angle_degrees));
            }
        }
        Ok(geometries)
    }

    #[tracing::instrument(skip_all)]
    pub async fn synchronize(
        &self,
        snapshot: &Snapshot,
        page: Option<EntityId>,
        commit: &Commit,
    ) -> Result<bool> {
        let Some(request) = self.claim(Claim::Commit(snapshot.project_id(), snapshot.revision()))
        else {
            return Ok(false);
        };
        let _preparation = self.preparation.lock().await;
        if !self.is_current(request) {
            return Ok(false);
        }
        let (current_page, revision, previous) = {
            let presentation = self.state.read();
            let frame = presentation.frame.as_ref();
            (
                frame.map(RenderedFrame::page),
                frame.map(RenderedFrame::revision),
                frame.cloned(),
            )
        };
        if current_page != page {
            return self.show_page_locked(snapshot, page, request).await;
        }
        if revision.is_some_and(|revision| revision >= snapshot.revision()) {
            return Ok(false);
        }
        let Some(page) = page else {
            self.replace_frame_if_current(request, None);
            return Ok(false);
        };
        let rendered = if commit.revision == snapshot.revision()
            && revision == Some(commit.changes.from)
            && let Some(previous) = previous.as_ref()
        {
            self.await_preparation(
                request,
                self.renderer.update(previous, snapshot, &commit.changes),
            )
            .await
        } else {
            self.await_preparation(request, self.renderer.render(snapshot, page))
                .await
        };
        let Some(rendered) = rendered else {
            return Ok(false);
        };
        let rendered = rendered?;
        if !self.replace_frame_if_current(request, Some(rendered)) {
            return Ok(false);
        }
        Ok(true)
    }

    /// Presents a project the window just opened in place of whatever it presented before.
    #[tracing::instrument(skip_all)]
    pub async fn show_project(&self, snapshot: &Snapshot, page: Option<EntityId>) -> Result<bool> {
        let Some(request) = self.claim(Claim::Project(snapshot.project_id(), snapshot.revision()))
        else {
            return Ok(false);
        };
        let _preparation = self.preparation.lock().await;
        if !self.is_current(request) {
            return Ok(false);
        }
        self.show_page_locked(snapshot, page, request).await
    }

    #[tracing::instrument(skip_all)]
    pub async fn show_page(&self, snapshot: &Snapshot, page: Option<EntityId>) -> Result<bool> {
        let Some(request) =
            self.claim(Claim::Selection(snapshot.project_id(), snapshot.revision()))
        else {
            return Ok(false);
        };
        let _preparation = self.preparation.lock().await;
        if !self.is_current(request) {
            return Ok(false);
        }
        self.show_page_locked(snapshot, page, request).await
    }

    pub async fn clear(&self) {
        let Some(request) = self.claim(Claim::Clear) else {
            return;
        };
        let _preparation = self.preparation.lock().await;
        if !self.is_current(request) {
            return;
        }
        self.replace_frame_if_current(request, None);
    }

    async fn show_page_locked(
        &self,
        snapshot: &Snapshot,
        page: Option<EntityId>,
        request: Ticket,
    ) -> Result<bool> {
        if !self.is_current(request) {
            return Ok(false);
        }
        let Some(page) = page else {
            return Ok(self.replace_frame_if_current(request, None));
        };
        let cached = self.page_frames.lock().get(page, snapshot.revision());
        if let Some(frame) = cached {
            return Ok(self.replace_frame_if_current(request, Some(frame)));
        }
        let Some(frame) = self
            .await_preparation(request, self.renderer.render(snapshot, page))
            .await
        else {
            return Ok(false);
        };
        let frame = frame?;
        if !self.replace_frame_if_current(request, Some(frame)) {
            return Ok(false);
        }
        Ok(true)
    }

    fn claim(&self, claim: Claim) -> Option<Ticket> {
        let request = {
            let mut presentation = self.state.write();
            let accepted = match (claim, presentation.claimed) {
                (Claim::Project(..) | Claim::Clear, _) => true,
                (Claim::Selection(project, revision), Some((claimed, latest))) => {
                    project == claimed && revision >= latest
                }
                (Claim::Commit(project, revision), Some((claimed, latest))) => {
                    project == claimed && revision > latest
                }
                (Claim::Selection(..) | Claim::Commit(..), None) => false,
            };
            if !accepted {
                return None;
            }
            presentation.claimed = match claim {
                Claim::Project(project, revision)
                | Claim::Selection(project, revision)
                | Claim::Commit(project, revision) => Some((project, revision)),
                Claim::Clear => None,
            };
            presentation.requested_presentation =
                presentation.requested_presentation.wrapping_add(1);
            Ticket {
                presentation: presentation.requested_presentation,
                prefetch: None,
            }
        };
        if matches!(claim, Claim::Project(..)) {
            // Page identities and revisions are only meaningful within one project.
            self.page_frames.lock().clear();
        }
        self.preparation_changed.notify_waiters();
        Some(request)
    }

    fn claim_prefetch(&self, project: ProjectId, revision: Revision) -> Option<Ticket> {
        let request = {
            let mut presentation = self.state.write();
            if !presentation
                .claimed
                .is_some_and(|(claimed, latest)| claimed == project && revision >= latest)
            {
                return None;
            }
            presentation.requested_prefetch = presentation.requested_prefetch.wrapping_add(1);
            Ticket {
                presentation: presentation.requested_presentation,
                prefetch: Some(presentation.requested_prefetch),
            }
        };
        self.preparation_changed.notify_waiters();
        Some(request)
    }

    fn is_current(&self, request: Ticket) -> bool {
        request.is_current(&self.state.read())
    }

    async fn await_preparation<F>(&self, request: Ticket, future: F) -> Option<F::Output>
    where
        F: Future,
    {
        tokio::select! {
            biased;
            () = self.preparation_superseded(request) => None,
            output = future => self.is_current(request).then_some(output),
        }
    }

    async fn preparation_superseded(&self, request: Ticket) {
        loop {
            let changed = self.preparation_changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if !self.is_current(request) {
                return;
            }
            changed.await;
        }
    }

    fn replace_frame_if_current(&self, request: Ticket, frame: Option<RenderedFrame>) -> bool {
        let mut presentation = self.state.write();
        if !request.is_current(&presentation) {
            return false;
        }
        presentation.generation = presentation.generation.wrapping_add(1).max(1);
        presentation.frame.clone_from(&frame);
        drop(presentation);
        if let Some(frame) = frame {
            self.page_frames.lock().insert(frame);
        } else {
            self.page_frames.lock().clear();
        }
        true
    }
}

impl PageFrameCache {
    fn get(&mut self, page: EntityId, revision: Revision) -> Option<RenderedFrame> {
        self.clock = self.clock.wrapping_add(1);
        let entry = self
            .entries
            .iter_mut()
            .find(|entry| entry.page == page && entry.revision == revision)?;
        entry.last_used = self.clock;
        Some(entry.frame.clone())
    }

    fn insert(&mut self, frame: RenderedFrame) {
        self.clock = self.clock.wrapping_add(1);
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.page == frame.page() && entry.revision == frame.revision())
        {
            entry.frame = frame;
            entry.last_used = self.clock;
            return;
        }
        self.entries.push(CachedPageFrame {
            page: frame.page(),
            revision: frame.revision(),
            frame,
            last_used: self.clock,
        });
        if self.entries.len() > MAX_CACHED_PAGE_FRAMES {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(index, _)| index)
                .expect("an over-capacity page frame cache is non-empty");
            self.entries.swap_remove(oldest);
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

fn frame_transform(original: Frame, preview: Frame) -> [f64; 6] {
    let original_angle = f64::from(original.angle_degrees).to_radians();
    let preview_angle = f64::from(preview.angle_degrees).to_radians();
    let (original_sin, original_cos) = original_angle.sin_cos();
    let (preview_sin, preview_cos) = preview_angle.sin_cos();
    let scale_x = f64::from(preview.width / original.width);
    let scale_y = f64::from(preview.height / original.height);
    let a = preview_cos * scale_x * original_cos + preview_sin * scale_y * original_sin;
    let b = preview_sin * scale_x * original_cos - preview_cos * scale_y * original_sin;
    let c = preview_cos * scale_x * original_sin - preview_sin * scale_y * original_cos;
    let d = preview_sin * scale_x * original_sin + preview_cos * scale_y * original_cos;
    let original_center_x = f64::from(original.x + original.width * 0.5);
    let original_center_y = f64::from(original.y + original.height * 0.5);
    let preview_center_x = f64::from(preview.x + preview.width * 0.5);
    let preview_center_y = f64::from(preview.y + preview.height * 0.5);
    [
        a,
        b,
        c,
        d,
        preview_center_x - a * original_center_x - c * original_center_y,
        preview_center_y - b * original_center_x - d * original_center_y,
    ]
}

fn transform_point([a, b, c, d, e, f]: [f64; 6], point: &Point) -> Point {
    Point {
        x: a * point.x + c * point.y + e,
        y: b * point.x + d * point.y + f,
    }
}

#[cfg(test)]
mod tests {
    use std::{future::pending, sync::Arc, time::Duration};

    use koharu_scene::{Revision, Session};
    use tokio::{sync::Notify, time::timeout};

    use super::{Claim, Desktop};

    #[tokio::test]
    async fn newer_request_preempts_current_preparation_and_acquires_ownership() {
        let presentation = Arc::new(Desktop::new().unwrap().presentation());
        let first_request = presentation.claim(Claim::Clear).unwrap();
        let started = Arc::new(Notify::new());
        let first = {
            let presentation = Arc::clone(&presentation);
            let started = Arc::clone(&started);
            tokio::spawn(async move {
                let _preparation = presentation.preparation.lock().await;
                started.notify_one();
                presentation
                    .await_preparation(first_request, pending::<()>())
                    .await
            })
        };
        started.notified().await;

        let latest_request = presentation.claim(Claim::Clear).unwrap();
        let _preparation = timeout(Duration::from_secs(1), presentation.preparation.lock())
            .await
            .expect("the obsolete preparation must release ownership");
        assert!(first.await.unwrap().is_none());
        assert_eq!(
            presentation
                .await_preparation(latest_request, std::future::ready(7))
                .await,
            Some(7)
        );
    }

    #[tokio::test]
    async fn obsolete_request_cannot_replace_the_presentation() {
        let presentation = Desktop::new().unwrap().presentation();
        let obsolete = presentation.claim(Claim::Clear).unwrap();
        let latest = presentation.claim(Claim::Clear).unwrap();

        assert!(!presentation.replace_frame_if_current(obsolete, None));
        assert_eq!(presentation.canvas_state().generation, 0);
        assert!(presentation.replace_frame_if_current(latest, None));
        assert_eq!(presentation.canvas_state().generation, 1);
    }

    #[tokio::test]
    async fn older_project_state_cannot_supersede_newer_state() {
        let presentation = Desktop::new().unwrap().presentation();
        let project = Session::memory().await.unwrap().project_id();
        let other = Session::memory().await.unwrap().project_id();
        let at = Revision::new;

        assert!(presentation.claim(Claim::Commit(project, at(1))).is_none());
        let opened = presentation.claim(Claim::Project(project, at(2))).unwrap();
        assert!(presentation.claim(Claim::Commit(project, at(2))).is_none());
        assert!(
            presentation
                .claim(Claim::Selection(project, at(1)))
                .is_none()
        );
        assert!(presentation.claim(Claim::Commit(other, at(3))).is_none());
        assert!(presentation.claim_prefetch(project, at(1)).is_none());
        let prefetch = presentation.claim_prefetch(project, at(2)).unwrap();
        assert!(presentation.is_current(opened));

        let committed = presentation.claim(Claim::Commit(project, at(3))).unwrap();
        assert!(!presentation.is_current(opened));
        assert!(!presentation.is_current(prefetch));
        assert!(
            presentation
                .claim(Claim::Selection(project, at(2)))
                .is_none()
        );
        let selected = presentation
            .claim(Claim::Selection(project, at(3)))
            .unwrap();
        assert!(!presentation.is_current(committed));

        presentation.claim(Claim::Clear).unwrap();
        assert!(!presentation.is_current(selected));
        assert!(
            presentation
                .claim(Claim::Selection(project, at(4)))
                .is_none()
        );
        assert!(presentation.claim_prefetch(project, at(4)).is_none());
    }
}
