use anyhow::Context as _;
use koharu_scene::EntityId;
use serde::Deserialize;
use specta::Type;
use tauri::WebviewWindow;
use tauri_runtime_cef::CefRuntime;

use super::{
    Error,
    canvas::Point,
    project::{Page, Project, Typography},
    workspace::Workspace,
};

#[derive(Clone, Debug, Deserialize, Type)]
pub struct GeometryUpdate {
    pub layer: EntityId,
    pub points: Option<Vec<Point>>,
}

#[derive(Clone, Debug, Deserialize, Type)]
pub struct TypographyUpdate {
    pub layer: EntityId,
    pub typography: Typography,
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn rename_page(
    page: EntityId,
    label: String,
    window: WebviewWindow<CefRuntime>,
) -> Result<(), Error> {
    let workspace = Workspace::of(&window)?;
    let (commit, page) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.rename_page(page, label).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    workspace.synchronize(&commit, page).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn delete_pages(
    pages: Vec<EntityId>,
    window: WebviewWindow<CefRuntime>,
) -> Result<(), Error> {
    let workspace = Workspace::of(&window)?;
    let (commit, page) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.delete_pages(pages).await?;
        project.record_commit(&commit);
        project.reconcile_page();
        (commit, project.active_page())
    };
    workspace.synchronize(&commit, page).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn move_page(
    page: EntityId,
    index: u32,
    window: WebviewWindow<CefRuntime>,
) -> Result<(), Error> {
    let workspace = Workspace::of(&window)?;
    let (commit, page) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.move_page(page, index as usize).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    workspace.synchronize(&commit, page).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn set_source_text(
    layer: EntityId,
    text: String,
    window: WebviewWindow<CefRuntime>,
) -> Result<(), Error> {
    let workspace = Workspace::of(&window)?;
    let (commit, page) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_source_text(layer, text).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    workspace.synchronize(&commit, page).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn set_translation(
    layer: EntityId,
    text: Option<String>,
    window: WebviewWindow<CefRuntime>,
) -> Result<(), Error> {
    let workspace = Workspace::of(&window)?;
    let (commit, page) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_translation(layer, text).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    workspace.synchronize(&commit, page).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn set_typography(
    updates: Vec<TypographyUpdate>,
    window: WebviewWindow<CefRuntime>,
) -> Result<(), Error> {
    let workspace = Workspace::of(&window)?;
    let (commit, page) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_typography(updates).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    workspace.synchronize(&commit, page).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn set_geometry(
    updates: Vec<GeometryUpdate>,
    window: WebviewWindow<CefRuntime>,
) -> Result<(), Error> {
    let workspace = Workspace::of(&window)?;
    let (commit, page) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_geometry(updates).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    workspace.synchronize(&commit, page).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn set_visibility(
    layers: Vec<EntityId>,
    visible: Option<bool>,
    opacity: Option<f32>,
    window: WebviewWindow<CefRuntime>,
) -> Result<(), Error> {
    let workspace = Workspace::of(&window)?;
    let (commit, page) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_visibility(layers, visible, opacity).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    workspace.synchronize(&commit, page).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn delete_layers(
    layers: Vec<EntityId>,
    window: WebviewWindow<CefRuntime>,
) -> Result<(), Error> {
    let workspace = Workspace::of(&window)?;
    let (commit, page) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.delete_layers(layers).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    workspace.synchronize(&commit, page).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn move_layer(
    layer: EntityId,
    parent: EntityId,
    index: u32,
    window: WebviewWindow<CefRuntime>,
) -> Result<Page, Error> {
    let workspace = Workspace::of(&window)?;
    let (commit, page, view) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.move_layer(layer, parent, index as usize).await?;
        project.record_commit(&commit);
        let page = project.active_page().context("no active page")?;
        let view = Project::page(&commit.snapshot, page)?;
        (commit, page, view)
    };
    workspace.synchronize(&commit, Some(page)).await?;
    Ok(view)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn undo(window: WebviewWindow<CefRuntime>) -> Result<(), Error> {
    let workspace = Workspace::of(&window)?;
    let (commit, page) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.undo().await?;
        project.reconcile_page();
        (commit, project.active_page())
    };
    workspace.synchronize(&commit, page).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn redo(window: WebviewWindow<CefRuntime>) -> Result<(), Error> {
    let workspace = Workspace::of(&window)?;
    let (commit, page) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.redo().await?;
        project.reconcile_page();
        (commit, project.active_page())
    };
    workspace.synchronize(&commit, page).await?;
    Ok(())
}
