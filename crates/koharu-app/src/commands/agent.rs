mod host;

use std::collections::HashMap;

use anyhow::{Context as _, Result, anyhow};
use koharu_agent::{Account, Agent, Codex, CodexModel, Config, Control, Event, LoginEvent, RunId};
use parking_lot::Mutex;
use serde::Serialize;
use specta::Type;
use tauri::{AppHandle, Manager as _, State, WebviewWindow, ipc::Channel};
use tauri_runtime_cef::CefRuntime;
use tokio::sync::Notify;

use self::host::KoharuHost;
use super::{
    Error,
    workspace::{Workspace, Workspaces},
};

#[derive(Clone, Debug, Serialize, Type)]
pub struct AgentStatus {
    pub account: Option<Account>,
    pub models: Vec<CodexModel>,
    pub config: Config,
    pub running: Option<RunId>,
}

/// The Codex account and its sign-in, shared by every window's agent session.
pub(crate) struct AgentAccount {
    codex: Codex,
    login: Mutex<Option<Control>>,
    idle: Notify,
}

impl AgentAccount {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            codex: Codex::new()?,
            login: Mutex::new(None),
            idle: Notify::new(),
        })
    }

    async fn status(&self, workspace: &Workspace) -> Result<AgentStatus> {
        let mut account = self.codex.account()?;
        let models = if account.is_some() {
            match self.codex.models().await {
                Ok(models) => models,
                Err(error) => {
                    account = self.codex.account()?;
                    if account.is_some() {
                        return Err(error);
                    }
                    for workspace in workspace.handle.state::<Workspaces>().all() {
                        workspace.agent.agent.clear().await;
                    }
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };
        Ok(AgentStatus {
            account,
            models,
            config: workspace.agent.agent.config()?,
            running: workspace.agent.runs.lock().keys().next().copied(),
        })
    }

    async fn cancel_login(&self) {
        loop {
            let idle = self.idle.notified();
            if let Some(login) = self.login.lock().as_ref() {
                login.cancel();
            } else {
                break;
            }
            idle.await;
        }
    }
}

/// One window's agent conversation and runs.
pub(crate) struct AgentSession {
    agent: Agent<KoharuHost>,
    runs: Mutex<HashMap<RunId, Control>>,
    idle: Notify,
}

impl AgentSession {
    pub(crate) fn new(label: &str, handle: &AppHandle<CefRuntime>) -> Result<Self> {
        let codex = handle.state::<AgentAccount>().codex.clone();
        Ok(Self {
            agent: Agent::new(codex, KoharuHost::new(label.to_owned(), handle.clone()))?,
            runs: Mutex::new(HashMap::new()),
            idle: Notify::new(),
        })
    }

    pub(crate) async fn reset(&self) {
        loop {
            let idle = self.idle.notified();
            {
                let runs = self.runs.lock();
                if runs.is_empty() {
                    break;
                }
                for control in runs.values() {
                    control.cancel();
                }
            }
            idle.await;
        }
        self.agent.clear().await;
    }
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_agent_status(
    window: WebviewWindow<CefRuntime>,
    account: State<'_, AgentAccount>,
) -> std::result::Result<AgentStatus, Error> {
    let workspace = Workspace::of(&window)?;
    Ok(account.status(&workspace).await?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn login_agent(
    on_event: Channel<LoginEvent>,
    window: WebviewWindow<CefRuntime>,
    account: State<'_, AgentAccount>,
) -> std::result::Result<AgentStatus, Error> {
    let workspace = Workspace::of(&window)?;
    let control = Control::default();
    {
        let mut login = account.login.lock();
        if login.is_some() {
            return Err(anyhow!("Codex sign-in is already running").into());
        }
        *login = Some(control.clone());
    }
    let result = account
        .codex
        .login_device(&control, |event| {
            if let LoginEvent::DeviceCode {
                verification_url, ..
            } = &event
                && let Err(error) = open::that(verification_url)
            {
                tracing::warn!(%error, "failed to open the Codex device sign-in page");
            }
            let _ = on_event.send(event);
        })
        .await;
    account.login.lock().take();
    account.idle.notify_waiters();
    result?;
    Ok(account.status(&workspace).await?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn logout_agent(
    window: WebviewWindow<CefRuntime>,
    account: State<'_, AgentAccount>,
) -> std::result::Result<AgentStatus, Error> {
    let workspace = Workspace::of(&window)?;
    account.cancel_login().await;
    for workspace in window.state::<Workspaces>().all() {
        workspace.agent.reset().await;
    }
    account.codex.logout()?;
    Ok(account.status(&workspace).await?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn save_agent_config(
    config: Config,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<Config, Error> {
    Ok(Workspace::of(&window)?.agent.agent.save_config(config)?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn run_agent(
    prompt: String,
    on_event: Channel<Event>,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<RunId, Error> {
    let workspace = Workspace::of(&window)?;
    let prompt = prompt.trim().to_owned();
    if prompt.is_empty() {
        return Err(anyhow!("message cannot be empty").into());
    }
    if workspace.agent.agent.codex().account()?.is_none() {
        return Err(anyhow!("Codex is not signed in").into());
    }
    let run = RunId::new();
    let control = Control::default();
    {
        let mut runs = workspace.agent.runs.lock();
        if !runs.is_empty() {
            return Err(anyhow!("another agent request is already running").into());
        }
        runs.insert(run, control.clone());
    }
    drop(tauri::async_runtime::spawn(async move {
        let publish_control = control.clone();
        let result = workspace
            .agent
            .agent
            .run(run, prompt, control, |event| {
                if on_event.send(event).is_err() {
                    publish_control.cancel();
                }
            })
            .await;
        if let Err(error) = result {
            tracing::error!(%run, error = ?error, "agent request failed");
        }
        workspace.agent.runs.lock().remove(&run);
        workspace.agent.idle.notify_waiters();
    }));
    Ok(run)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn cancel_agent(
    run: RunId,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<(), Error> {
    Workspace::of(&window)?
        .agent
        .runs
        .lock()
        .get(&run)
        .with_context(|| format!("agent run {run} is not active"))?
        .cancel();
    Ok(())
}
