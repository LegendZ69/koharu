use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context as _, Result};
use koharu_desktop::{CanvasState, Desktop, Presentation};
use koharu_scene::{Commit, EntityId, Snapshot};
use parking_lot::Mutex;
use tauri::{AppHandle, Manager as _, WebviewWindow, ipc::Channel};
use tauri_runtime_cef::CefRuntime;
use tokio::sync::{Notify, watch};

use super::{
    ChannelExt as _,
    agent::AgentSession,
    lifecycle::{Download, ModelResources},
    preferences::Preferences,
    processing::{Job, Processing},
    project::{Project, ProjectInfo},
};

pub(crate) const MAIN_WINDOW: &str = "main";

/// Every window's workspace, keyed by window label, and the project directories they hold.
#[derive(Default)]
pub(crate) struct Workspaces {
    windows: Mutex<HashMap<String, Arc<Workspace>>>,
    claims: Arc<Claims>,
    last_focused: Mutex<Option<String>>,
}

impl Workspaces {
    pub(crate) fn insert(&self, workspace: Arc<Workspace>) {
        self.windows
            .lock()
            .insert(workspace.label.clone(), workspace);
    }

    /// Unregisters a window's workspace. The window takes no new projects, and the ones it holds
    /// show as closing until its close has dropped them.
    pub(crate) fn remove(&self, label: &str) -> Option<Arc<Workspace>> {
        self.claims.close_window(label);
        self.windows.lock().remove(label)
    }

    pub(crate) fn get(&self, label: &str) -> Option<Arc<Workspace>> {
        self.windows.lock().get(label).cloned()
    }

    pub(crate) fn all(&self) -> Vec<Arc<Workspace>> {
        self.windows.lock().values().cloned().collect()
    }

    pub(crate) fn focus(&self, label: &str) {
        *self.last_focused.lock() = Some(label.to_owned());
    }

    /// Holds the project directory at `path` for the window `label`.
    pub(crate) fn claim(&self, path: &Path, label: &str) -> Claim {
        self.claims.claim(project_key(path), label)
    }

    /// Like [`Self::claim`], but waits while a window is closing the directory, so it never
    /// returns [`Claim::Closing`].
    pub(crate) async fn reserve(&self, path: &Path, label: &str) -> Claim {
        self.claims.reserve(project_key(path), label).await
    }

    /// The most recently focused window that is still open, else any open window.
    pub(crate) fn focus_target(
        &self,
        handle: &AppHandle<CefRuntime>,
    ) -> Option<WebviewWindow<CefRuntime>> {
        let last_focused = self.last_focused.lock().clone();
        last_focused
            .and_then(|label| handle.get_webview_window(&label))
            .or_else(|| handle.webview_windows().into_values().next())
    }
}

/// Which window holds each project directory. A window holds a directory from before it opens
/// the project until the project and the jobs, agent run, and exports that used it have been
/// dropped, so windows never race the directory's write lock.
#[derive(Default)]
struct Claims {
    holders: Mutex<Holders>,
    released: Notify,
}

#[derive(Default)]
struct Holders {
    directories: HashMap<PathBuf, Holder>,
    /// The labels of closed windows, which are never reused. A closed window takes no new
    /// directories, so every directory it still holds shows as closing.
    closed: HashSet<String>,
}

struct Holder {
    label: String,
    closing: bool,
}

/// What a window found when it tried to hold a project directory.
pub(crate) enum Claim {
    /// The directory is now held for the window.
    Reserved(Reservation),
    /// The window with this label is opening the project or has it open.
    Open(String),
    /// A window is closing or deleting the project and still holds the directory.
    Closing,
    /// The window has closed and takes no new directories.
    Closed,
}

impl Claims {
    fn claim(self: &Arc<Self>, key: PathBuf, label: &str) -> Claim {
        let mut holders = self.holders.lock();
        if holders.closed.contains(label) {
            return Claim::Closed;
        }
        if let Some(holder) = holders.directories.get(&key) {
            return if holder.closing {
                Claim::Closing
            } else {
                Claim::Open(holder.label.clone())
            };
        }
        holders.directories.insert(
            key.clone(),
            Holder {
                label: label.to_owned(),
                closing: false,
            },
        );
        Claim::Reserved(Reservation {
            claims: Arc::clone(self),
            key,
        })
    }

    async fn reserve(self: &Arc<Self>, key: PathBuf, label: &str) -> Claim {
        loop {
            let released = self.released.notified();
            match self.claim(key.clone(), label) {
                Claim::Closing => released.await,
                claim => return claim,
            }
        }
    }

    fn close_window(&self, label: &str) {
        let mut holders = self.holders.lock();
        holders.closed.insert(label.to_owned());
        for holder in holders.directories.values_mut() {
            if holder.label == label {
                holder.closing = true;
            }
        }
    }
}

/// A window's hold on a project directory. Dropping it releases the directory.
pub(crate) struct Reservation {
    claims: Arc<Claims>,
    key: PathBuf,
}

impl Reservation {
    /// Shows the directory as closing until this reservation is dropped, so windows that open
    /// the project wait for it instead of racing its write lock.
    pub(crate) fn close(&self) {
        if let Some(holder) = self.claims.holders.lock().directories.get_mut(&self.key) {
            holder.closing = true;
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.claims.holders.lock().directories.remove(&self.key);
        self.claims.released.notify_waiters();
    }
}

#[derive(Default)]
pub(crate) struct Channels {
    pub(crate) canvas: Mutex<Option<Channel<CanvasState>>>,
    pub(crate) job: Mutex<Option<Channel<Job>>>,
    pub(crate) project: Mutex<Option<Channel<Option<ProjectInfo>>>>,
    pub(crate) download: Mutex<Option<Channel<Download>>>,
    pub(crate) resources: Mutex<Option<Channel<ModelResources>>>,
    pub(crate) preferences: Mutex<Option<Channel<Preferences>>>,
}

/// One window's project, presentation, jobs, agent session, and channel sinks.
pub(crate) struct Workspace {
    pub(crate) label: String,
    pub(crate) handle: AppHandle<CefRuntime>,
    pub(crate) project: tokio::sync::Mutex<Option<Project>>,
    /// The open project's directory. It is read without awaiting `project`, which stays locked
    /// across disk commits, and declared after it so that a dropped workspace drops its
    /// project before releasing the directory.
    binding: Mutex<Option<Reservation>>,
    /// Serializes opening and closing this workspace's project.
    transition: tokio::sync::Mutex<()>,
    pub(crate) presentation: Presentation,
    pub(crate) channels: Channels,
    pub(crate) processing: Processing,
    /// Every export of the open project holds a receiver of this channel with its snapshot. A
    /// switch or close takes the channel and waits for those receivers to drop.
    exports: Mutex<watch::Sender<()>>,
    pub(crate) agent: AgentSession,
}

impl Workspace {
    pub(crate) fn new(label: String, handle: &AppHandle<CefRuntime>) -> Result<Arc<Self>> {
        let agent = AgentSession::new(&label, handle)?;
        Ok(Arc::new(Self {
            label,
            handle: handle.clone(),
            project: tokio::sync::Mutex::new(None),
            binding: Mutex::new(None),
            transition: tokio::sync::Mutex::new(()),
            presentation: handle.state::<Desktop>().presentation(),
            channels: Channels::default(),
            processing: Processing::default(),
            exports: Mutex::default(),
            agent,
        }))
    }

    /// The workspace of the window that invoked a command.
    pub(crate) fn of(window: &WebviewWindow<CefRuntime>) -> Result<Arc<Self>> {
        window
            .state::<Workspaces>()
            .get(window.label())
            .with_context(|| format!("window {} has no workspace", window.label()))
    }

    pub(crate) fn binding(&self) -> Option<PathBuf> {
        self.binding
            .lock()
            .as_ref()
            .map(|reservation| reservation.key.clone())
    }

    /// Takes a snapshot for an export, if the open project is still the one bound to `binding`.
    pub(crate) async fn export_snapshot(&self, binding: Option<&Path>) -> Result<ExportSnapshot> {
        let project = self.project.lock().await;
        let project = project
            .as_ref()
            .filter(|_| self.binding().as_deref() == binding)
            .context("the project is no longer open")?;
        Ok(ExportSnapshot {
            snapshot: project.snapshot(),
            _export: self.exports.lock().subscribe(),
        })
    }

    /// Opens `next` in place of the current project, or closes it when `next` is `None`.
    ///
    /// The previous project's directory shows as closing until its jobs, agent run, and
    /// exports have finished and the project has been dropped, because any snapshot they still
    /// hold keeps the project's write lock alive.
    pub(crate) async fn replace(&self, next: Option<(Reservation, Project)>) -> Result<()> {
        let _transition = self.transition.lock().await;
        let (reservation, next) = next.unzip();
        let snapshot = next.as_ref().map(Project::snapshot);
        let page = next.as_ref().and_then(Project::active_page);
        let info = next.as_ref().map(Project::info);

        if let Some(previous) = self.binding.lock().as_ref() {
            previous.close();
        }
        // Stopping the jobs first keeps them from committing while the agent run winds down.
        self.processing.stop_all();
        self.agent.reset().await;
        let (previous, binding, jobs, exports) = {
            let mut project = self.project.lock().await;
            let binding = std::mem::replace(&mut *self.binding.lock(), reservation);
            let jobs = self.processing.stop_all();
            let exports = std::mem::take(&mut *self.exports.lock());
            (
                std::mem::replace(&mut *project, next),
                binding,
                jobs,
                exports,
            )
        };
        self.processing.wait(&jobs).await;
        exports.closed().await;
        let closed = previous
            .as_ref()
            .map(|project| project.snapshot().project_id());
        drop(previous);
        drop(binding);

        let shown = match &snapshot {
            Some(snapshot) => self.presentation.show_project(snapshot, page).await,
            None => {
                self.presentation.clear().await;
                Ok(true)
            }
        };
        // The shared renderer keeps retained nodes only for projects that a window presents.
        if let Some(closed) = closed
            && !self
                .handle
                .state::<Workspaces>()
                .all()
                .iter()
                .any(|workspace| workspace.presentation.presents(closed))
        {
            self.handle
                .state::<Desktop>()
                .renderer()
                .discard_retained_nodes(closed);
        }
        let title = window_title(info.as_ref().map(|info| info.name.as_str()));
        if let Some(window) = self.handle.get_webview_window(&self.label)
            && let Err(error) = window.set_title(&title)
        {
            tracing::warn!(%error, "failed to update the window title");
        }
        self.publish_canvas();
        self.channels.project.publish(info);
        shown?;
        Ok(())
    }

    /// Presents a commit of this workspace's project and publishes the resulting canvas.
    pub(crate) async fn synchronize(&self, commit: &Commit, page: Option<EntityId>) -> Result<()> {
        self.presentation
            .synchronize(&commit.snapshot, page, commit)
            .await?;
        self.publish_canvas();
        Ok(())
    }

    /// Reads the canvas state while holding the channel, so concurrent publishers deliver
    /// generations in order.
    pub(crate) fn publish_canvas(&self) {
        let mut channel = self.channels.canvas.lock();
        let canvas = self.presentation.canvas_state();
        if channel
            .as_ref()
            .is_some_and(|channel| channel.send(canvas).is_err())
        {
            channel.take();
        }
    }
}

/// A snapshot of a window's project taken for an export. A switch or close of the project
/// waits for it to drop before releasing the project's directory, because the snapshot keeps
/// the directory's write lock alive.
pub(crate) struct ExportSnapshot {
    snapshot: Snapshot,
    /// Declared after the snapshot, so the export is released only once the snapshot is gone.
    _export: watch::Receiver<()>,
}

impl Deref for ExportSnapshot {
    type Target = Snapshot;

    fn deref(&self) -> &Snapshot {
        &self.snapshot
    }
}

pub(crate) fn window_title(project: Option<&str>) -> String {
    project.map_or_else(|| "Koharu".to_owned(), |name| format!("{name} — Koharu"))
}

pub(crate) fn focus_window(window: &WebviewWindow<CefRuntime>) {
    let _ = window.unminimize();
    let _ = window.show();
    let _ = window.set_focus();
}

/// Identifies a project directory independently of the case its name was typed in, and the
/// same before and after the directory exists.
fn project_key(path: &Path) -> PathBuf {
    if let Ok(key) = std::fs::canonicalize(path) {
        return key;
    }
    match (path.parent().map(std::fs::canonicalize), path.file_name()) {
        (Some(Ok(parent)), Some(name)) => parent.join(name),
        _ => path.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, pin::pin, sync::Arc};

    use super::{Claim, Claims, Reservation, project_key};

    fn reserved(claim: Claim) -> Reservation {
        match claim {
            Claim::Reserved(reservation) => reservation,
            Claim::Open(label) => panic!("the directory is open in {label}"),
            Claim::Closing => panic!("the directory is closing"),
            Claim::Closed => panic!("the window has closed"),
        }
    }

    fn holder(claim: Claim) -> Option<String> {
        match claim {
            Claim::Open(label) => Some(label),
            Claim::Reserved(_) | Claim::Closing | Claim::Closed => None,
        }
    }

    #[test]
    fn a_directory_stays_held_until_its_reservation_drops() {
        let claims = Arc::new(Claims::default());
        let project = PathBuf::from("project.khrproj");

        let opened = reserved(claims.claim(project.clone(), "main"));
        assert_eq!(
            holder(claims.claim(project.clone(), "second")).as_deref(),
            Some("main")
        );
        assert_eq!(
            holder(claims.claim(project.clone(), "main")).as_deref(),
            Some("main")
        );
        opened.close();
        assert!(matches!(
            claims.claim(project.clone(), "second"),
            Claim::Closing
        ));
        drop(opened);
        reserved(claims.claim(project, "second"));
    }

    #[test]
    fn a_closed_window_keeps_its_directories_closing_until_they_drop() {
        let claims = Arc::new(Claims::default());
        let project = PathBuf::from("project.khrproj");
        let other = PathBuf::from("other.khrproj");
        let opening = reserved(claims.claim(project.clone(), "main"));
        let _held = reserved(claims.claim(other.clone(), "second"));

        claims.close_window("main");
        assert!(matches!(
            claims.claim(project.clone(), "third"),
            Claim::Closing
        ));
        assert_eq!(
            holder(claims.claim(other, "third")).as_deref(),
            Some("second")
        );
        drop(opening);
        reserved(claims.claim(project, "third"));
    }

    #[tokio::test]
    async fn reserving_a_closing_directory_waits_for_its_release() {
        let claims = Arc::new(Claims::default());
        let project = PathBuf::from("project.khrproj");
        let closing = reserved(claims.claim(project.clone(), "main"));
        closing.close();

        let mut waiting = pin!(claims.reserve(project.clone(), "second"));
        assert!(futures::poll!(waiting.as_mut()).is_pending());
        drop(closing);
        let reservation = reserved(waiting.await);
        assert_eq!(
            holder(claims.claim(project, "main")).as_deref(),
            Some("second")
        );
        drop(reservation);
    }

    #[tokio::test]
    async fn a_closed_window_takes_no_new_directories() {
        let claims = Arc::new(Claims::default());
        let project = PathBuf::from("project.khrproj");
        let closing = reserved(claims.claim(project.clone(), "second"));
        closing.close();

        let mut waiting = pin!(claims.reserve(project.clone(), "main"));
        assert!(futures::poll!(waiting.as_mut()).is_pending());
        claims.close_window("main");
        drop(closing);
        assert!(matches!(waiting.await, Claim::Closed));
        assert!(matches!(
            claims.claim(PathBuf::from("other.khrproj"), "main"),
            Claim::Closed
        ));
        reserved(claims.claim(project, "third"));
    }

    #[test]
    fn a_project_keeps_its_key_when_its_directory_is_created() {
        let library = tempfile::tempdir().unwrap();
        let project = library.path().join("Project.khrproj");
        let before = project_key(&project);
        std::fs::create_dir(&project).unwrap();
        assert_eq!(project_key(&project), before);
    }
}
