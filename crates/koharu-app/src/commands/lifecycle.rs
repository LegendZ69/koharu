use std::path::PathBuf;

use anyhow::{Context as _, Result, anyhow, bail};
use koharu_desktop::CanvasState;
use koharu_scene::{AssetInput, AssetMetadata, AssetRole, At, PageDraft};
use serde::{Deserialize, Serialize};
use specta::Type;
use strum::{EnumMessage as _, IntoEnumIterator as _};
use tauri::{AppHandle, Manager as _, State, WebviewWindow, WebviewWindowBuilder, ipc::Channel};
use tauri_runtime_cef::CefRuntime;
use uuid::Uuid;
use walkdir::WalkDir;

use super::{
    Error, import,
    preferences::Preferences,
    processing::Job,
    project::{Page, PageSummary, Project, ProjectInfo, ProjectLibrary, ProjectSummary},
    workspace::{
        Claim, MAIN_WINDOW, Reservation, Workspace, Workspaces, focus_window, window_title,
    },
};

/// How far a new window is offset from the focused one, in logical pixels.
const WINDOW_CASCADE: f64 = 32.0;

#[derive(Clone, Debug, Serialize, Type)]
pub struct StartupState {
    pub preferences: Preferences,
    pub jobs: Vec<Job>,
    pub canvas: CanvasState,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct PageSelection {
    pub project: ProjectInfo,
    pub page: Page,
}

#[derive(Clone, Copy, Debug, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum PageImportSource {
    Files,
    Folder,
}

pub(crate) struct Initialization {
    ready: tokio::sync::watch::Sender<bool>,
}

impl Default for Initialization {
    fn default() -> Self {
        let (ready, _) = tokio::sync::watch::channel(false);
        Self { ready }
    }
}

impl Initialization {
    pub(crate) fn ready(&self) {
        self.ready.send_replace(true);
    }

    async fn wait(&self) -> Result<()> {
        let mut ready = self.ready.subscribe();
        while !*ready.borrow_and_update() {
            ready
                .changed()
                .await
                .context("startup state closed before initialization completed")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct Download {
    #[specta(type = f64)]
    pub id: u64,
    pub state: DownloadState,
    pub name: Option<String>,
    #[specta(type = f64)]
    pub completed: u64,
    #[specta(type = f64)]
    pub total: u64,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum DownloadState {
    Running,
    Finished,
    Failed,
}

#[derive(Clone, Debug, Default, Serialize, Type)]
pub struct ModelResources {
    #[specta(type = f64)]
    pub process_memory: u64,
    #[specta(type = f64)]
    pub system_memory: u64,
    pub process_cpu: f32,
    pub devices: Vec<DeviceResources>,
}

#[derive(Clone, Debug, Default, Serialize, Type)]
pub struct DeviceResources {
    pub name: String,
    pub selected: bool,
    #[specta(type = Option<f64>)]
    pub memory_budget: Option<u64>,
    #[specta(type = Option<f64>)]
    pub memory_used: Option<u64>,
    pub utilization: Option<f32>,
}

impl From<koharu_pipeline::ResourceSnapshot> for ModelResources {
    fn from(value: koharu_pipeline::ResourceSnapshot) -> Self {
        Self {
            process_memory: value.process_memory_bytes,
            system_memory: value.system_memory_bytes,
            process_cpu: value.process_cpu_percent,
            devices: value
                .devices
                .into_iter()
                .map(|device| DeviceResources {
                    name: device.name,
                    selected: device.selected,
                    memory_budget: device.memory_budget_bytes,
                    memory_used: device.memory_used_bytes,
                    utilization: device.utilization_percent,
                })
                .collect(),
        }
    }
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn subscribe(
    window: WebviewWindow<CefRuntime>,
    on_canvas: Channel<CanvasState>,
    on_job: Channel<Job>,
    on_download: Channel<Download>,
    on_resources: Channel<ModelResources>,
    on_project: Channel<Option<ProjectInfo>>,
    on_preferences: Channel<Preferences>,
) -> std::result::Result<StartupState, Error> {
    window.state::<Initialization>().wait().await?;
    let workspace = Workspace::of(&window)?;

    *workspace.channels.canvas.lock() = Some(on_canvas);
    *workspace.channels.job.lock() = Some(on_job);
    *workspace.channels.download.lock() = Some(on_download);
    *workspace.channels.resources.lock() = Some(on_resources);
    *workspace.channels.project.lock() = Some(on_project);
    *workspace.channels.preferences.lock() = Some(on_preferences);

    let preferences = Preferences::load()?;
    Ok(StartupState {
        preferences,
        jobs: workspace.processing.jobs(),
        canvas: workspace.presentation.canvas_state(),
    })
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn open_window(
    project: Option<String>,
    handle: AppHandle<CefRuntime>,
) -> std::result::Result<(), Error> {
    let label = format!("workspace-{}", Uuid::new_v4().simple());
    let project = match project {
        Some(name) => match reserve_project(&handle, &label, &name).await? {
            Some(project) => Some(project),
            None => return Ok(()),
        },
        None => None,
    };
    let workspace = Workspace::new(label.clone(), &handle)?;
    let workspaces = handle.state::<Workspaces>();
    workspaces.insert(workspace.clone());
    if let Err(error) = build_window(&handle, &workspace, project).await {
        workspaces.remove(&label);
        workspace.replace(None).await?;
        return Err(error.into());
    }
    Ok(())
}

async fn build_window(
    handle: &AppHandle<CefRuntime>,
    workspace: &Workspace,
    project: Option<(String, PathBuf, Reservation)>,
) -> Result<()> {
    let mut config = handle
        .config()
        .app
        .windows
        .iter()
        .find(|window| window.label == MAIN_WINDOW)
        .context("the main Tauri window configuration is unavailable")?
        .clone();
    config.label.clone_from(&workspace.label);
    if let Some((name, path, reservation)) = project {
        config.title = window_title(Some(&name));
        let opened = Project::open(name, path).await?;
        workspace.replace(Some((reservation, opened))).await?;
    }
    if let Some(anchor) = handle.state::<Workspaces>().focus_target(handle)
        && let (Ok(position), Ok(scale)) = (anchor.outer_position(), anchor.scale_factor())
    {
        let position = position.to_logical::<f64>(scale);
        config.x = Some(position.x + WINDOW_CASCADE);
        config.y = Some(position.y + WINDOW_CASCADE);
        config.center = false;
    }
    let window = WebviewWindowBuilder::from_config(handle, &config)?
        .build()
        .context("failed to create the window")?;
    focus_window(&window);
    Ok(())
}

/// Resolves `name` and reserves its project directory for the window `label`, after any window
/// closing the project has dropped it. A project that a window is opening or has open is
/// focused in that window instead, and `None` is returned.
async fn reserve_project(
    handle: &AppHandle<CefRuntime>,
    label: &str,
    name: &str,
) -> Result<Option<(String, PathBuf, Reservation)>> {
    let (name, path) = handle.state::<ProjectLibrary>().resolve(name)?;
    let workspaces = handle.state::<Workspaces>();
    let holder = match workspaces.reserve(&path, label).await {
        Claim::Reserved(reservation) => return Ok(Some((name, path, reservation))),
        Claim::Open(holder) => holder,
        Claim::Closed => bail!("the window has closed"),
        Claim::Closing => unreachable!("reserving waits while a window closes the project"),
    };
    match handle.get_webview_window(&holder) {
        Some(window) => focus_window(&window),
        // A registered workspace without a window is a new window, which is focused once built.
        None if workspaces.get(&holder).is_some() => {}
        None => bail!("the project is still opening or closing in another window"),
    }
    Ok(None)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_project(
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<Option<ProjectInfo>, Error> {
    let workspace = Workspace::of(&window)?;
    Ok(workspace.project.lock().await.as_ref().map(Project::info))
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_pages(
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<Vec<PageSummary>, Error> {
    let snapshot = Workspace::of(&window)?
        .project
        .lock()
        .await
        .as_ref()
        .context("no project is open")?
        .snapshot();
    Ok(Project::pages(&snapshot)?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_page(
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<Option<Page>, Error> {
    let workspace = Workspace::of(&window)?;
    let current = {
        let project = workspace.project.lock().await;
        project
            .as_ref()
            .map(|project| (project.snapshot(), project.active_page()))
    };
    Ok(current
        .and_then(|(snapshot, page)| page.map(|page| (snapshot, page)))
        .map(|(snapshot, page)| Project::page(&snapshot, page))
        .transpose()?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn list_projects(
    library: State<'_, ProjectLibrary>,
) -> std::result::Result<Vec<ProjectSummary>, Error> {
    Ok(library.list()?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn create_project(
    name: String,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<(), Error> {
    let workspace = Workspace::of(&window)?;
    let (name, path) = window.state::<ProjectLibrary>().resolve(&name)?;
    let reservation = match window
        .state::<Workspaces>()
        .reserve(&path, &workspace.label)
        .await
    {
        Claim::Reserved(reservation) => reservation,
        Claim::Open(holder) if holder == workspace.label => {
            return Err(anyhow!("a project with this name is open in this window").into());
        }
        Claim::Open(_) => {
            return Err(anyhow!("a project with this name is open in another window").into());
        }
        Claim::Closed => return Err(anyhow!("the window has closed").into()),
        Claim::Closing => unreachable!("reserving waits while a window closes the project"),
    };
    let created = Project::create(name, path).await?;
    workspace.replace(Some((reservation, created))).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn open_project(
    name: String,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<(), Error> {
    let workspace = Workspace::of(&window)?;
    let Some((name, path, reservation)) =
        reserve_project(window.app_handle(), &workspace.label, &name).await?
    else {
        return Ok(());
    };
    let opened = Project::open(name, path).await?;
    workspace.replace(Some((reservation, opened))).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn close_project(
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<(), Error> {
    Workspace::of(&window)?.replace(None).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn delete_project(
    name: String,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<(), Error> {
    let library = window.state::<ProjectLibrary>().inner().clone();
    let (_, path) = library.resolve(&name)?;
    let reservation = match window.state::<Workspaces>().claim(&path, window.label()) {
        Claim::Reserved(reservation) => reservation,
        Claim::Open(holder) if holder == window.label() => {
            return Err(anyhow!("the project is open in this window").into());
        }
        Claim::Open(_) => return Err(anyhow!("the project is open in another window").into()),
        Claim::Closing => return Err(anyhow!("the project is still closing").into()),
        Claim::Closed => return Err(anyhow!("the window has closed").into()),
    };
    // Windows that open the project meanwhile wait for the deletion instead of racing it.
    reservation.close();
    tokio::task::spawn_blocking(move || library.delete(&name))
        .await
        .context("project deletion task failed")??;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn import(
    source: PageImportSource,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<(), Error> {
    let workspace = Workspace::of(&window)?;
    if workspace.processing.is_running() {
        return Err(anyhow::anyhow!("pages cannot be imported while processing is running").into());
    }
    let extensions = import::Format::iter()
        .flat_map(|format| format.get_serializations())
        .collect::<Vec<_>>();
    let dialog = rfd::AsyncFileDialog::new()
        .add_filter("Images, archives, and PDF", &extensions)
        .set_parent(&window);
    let files = match source {
        PageImportSource::Files => dialog.pick_files().await.map(|files| {
            files
                .into_iter()
                .map(|file| file.path().to_owned())
                .collect::<Vec<_>>()
        }),
        PageImportSource::Folder => dialog.pick_folder().await.map(|folder| {
            WalkDir::new(folder.path())
                .follow_links(false)
                .into_iter()
                .filter_map(|entry| match entry {
                    Ok(entry) if entry.file_type().is_file() => Some(entry.into_path()),
                    Ok(_) => None,
                    Err(error) => {
                        tracing::warn!(%error, "could not inspect an import directory entry");
                        None
                    }
                })
                .filter(|path| {
                    path.extension()
                        .and_then(|extension| extension.to_str())
                        .is_some_and(|extension| extension.parse::<import::Format>().is_ok())
                })
                .collect::<Vec<_>>()
        }),
    };
    let Some(files) = files else {
        return Ok(());
    };
    if files.is_empty() {
        return Err(anyhow::anyhow!("no supported images were found in the selection").into());
    }
    let pages = tokio_rayon::spawn(move || import::import(files)).await?;

    let (commit, page) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let source = AssetRole::new("source")?;
        let patch = project.snapshot().patch(|edit| {
            for imported in pages {
                let page = edit.add_page(
                    PageDraft::new(
                        imported.name,
                        f64::from(imported.width),
                        f64::from(imported.height),
                    ),
                    At::End,
                )?;
                edit.set_asset(
                    page,
                    &source,
                    AssetInput::new(
                        imported.bytes,
                        imported.format.to_mime_type(),
                        AssetMetadata {
                            width: Some(imported.width),
                            height: Some(imported.height),
                            attributes: Default::default(),
                        },
                    ),
                )?;
            }
            Ok(())
        })?;
        let commit = project.session.commit(patch).await?;
        project.record(vec![commit.revision]);
        project.reconcile_page();
        let page = project.active_page();
        (commit, page)
    };
    workspace.synchronize(&commit, page).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn select_page(
    page: koharu_scene::EntityId,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<PageSelection, Error> {
    let workspace = Workspace::of(&window)?;
    let (snapshot, project_info, selected_page) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        project.select_page(page)?;
        let snapshot = project.snapshot();
        let project_info = project.info();
        let selected_page = Project::page(&snapshot, page)?;
        (snapshot, project_info, selected_page)
    };
    if workspace
        .presentation
        .show_page(&snapshot, Some(page))
        .await?
    {
        workspace.publish_canvas();
    }
    Ok(PageSelection {
        project: project_info,
        page: selected_page,
    })
}
