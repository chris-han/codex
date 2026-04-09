use crate::models::*;
use crate::providers::factory::ProviderFactory;
use crate::providers::llm_hook::FactoryAdapter;
use crate::ServerConfig;
use codex_client::set_llm_provider_factory;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// Codex client imports
use codex_app_server_client::AppServerEvent;
use codex_app_server_client::InProcessAppServerClient;
use codex_app_server_client::InProcessAppServerRequestHandle;
use codex_app_server_client::InProcessClientStartArgs;
use codex_app_server_protocol::ServerRequest;
use codex_core::config::Config;
use codex_core::config_loader::CloudRequirementsLoader;
use codex_core::config_loader::LoaderOverrides;
use codex_feedback::CodexFeedback;
use codex_protocol::protocol::SessionSource;

/// Per-user session: split request handle (clonable) from event stream.
#[derive(Clone)]
pub struct UserSession {
    /// Clone-able handle to send requests to the in-process codex runtime.
    pub request_handle: InProcessAppServerRequestHandle,
    /// Broadcast channel receiver — subscribe to get a stream of JSON events.
    pub event_tx: broadcast::Sender<serde_json::Value>,
}

/// Application state shared across all handlers
#[allow(dead_code)]
pub struct AppState {
    pub config: ServerConfig,
    /// Per-user sessions (user_id -> session)
    pub sessions: RwLock<HashMap<String, UserSession>>,
    /// Global settings
    pub settings: RwLock<Settings>,
    /// LLM provider factory (for direct LLM API calls)
    pub provider_factory: Option<ProviderFactory>,
}

/// Runtime settings
#[derive(Debug, Clone)]
pub struct Settings {
    pub codex_home: String,
    pub user_files_path: String,
    pub user_threads_path: String,
    pub sandbox_mode: String,
    pub network_access: bool,
    pub exclude_tmpdir_env_var: bool,
    pub exclude_slash_tmp: bool,
    pub markets: Vec<Market>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            codex_home: String::from("/tmp/codex"),
            user_files_path: String::from("/tmp/user_files"),
            user_threads_path: String::from("/tmp/user_threads"),
            sandbox_mode: String::from("workspace-write"),
            network_access: false,
            exclude_tmpdir_env_var: false,
            exclude_slash_tmp: false,
            markets: vec![
                Market {
                    owner: String::from("openai"),
                    repo: String::from("skills"),
                    active: true,
                },
            ],
        }
    }
}

impl AppState {
    pub async fn new(config: ServerConfig) -> anyhow::Result<Self> {
        let settings = Settings {
            codex_home: config.codex_home.clone(),
            user_files_path: config.user_files_path.clone(),
            user_threads_path: config.user_threads_path.clone(),
            ..Settings::default()
        };

        // Create directories
        tokio::fs::create_dir_all(&config.codex_home).await?;
        tokio::fs::create_dir_all(&config.user_files_path).await?;
        tokio::fs::create_dir_all(&config.user_threads_path).await?;

        // Initialize LLM provider factory if credentials are available
        let provider_factory = ProviderFactory::from_env().ok();
        if provider_factory.is_some() {
            info!("LLM provider factory initialized");
            // Register a second factory instance for the in-process codex-api hook.
            // ProviderFactory::from_env() is cheap (env reads + client construction).
            if let Ok(hook_factory) = ProviderFactory::from_env() {
                let adapter = Arc::new(FactoryAdapter::new(hook_factory));
                if set_llm_provider_factory(adapter) {
                    info!("In-process LLM provider hook registered");
                } else {
                    info!("In-process LLM provider hook already registered");
                }
            }
        } else {
            info!("No LLM provider configured (set KIMI_API_KEY or Azure credentials)");
        }

        info!("AppState initialized");
        debug!(
            codex_home = %config.codex_home,
            user_files_path = %config.user_files_path,
            user_threads_path = %config.user_threads_path,
            "AppState directories ensured"
        );

        Ok(Self {
            config,
            sessions: RwLock::new(HashMap::new()),
            settings: RwLock::new(settings),
            provider_factory,
        })
    }

    /// Get or create a session for a user.
    pub async fn get_session(
        &self,
        user_id: &str,
    ) -> anyhow::Result<UserSession> {
        {
            let sessions = self.sessions.read().await;
            if let Some(session) = sessions.get(user_id) {
                debug!(user_id, "Reusing existing codex session");
                return Ok(session.clone());
            }
        }

        // Create new client + session
        let session = self.create_session(user_id).await?;

        {
            let mut sessions = self.sessions.write().await;
            // Double-check in case another task created it while we were initializing
            if let Some(existing) = sessions.get(user_id) {
                return Ok(existing.clone());
            }
            sessions.insert(user_id.to_string(), session.clone());
        }

        info!("Created new codex session for user: {}", user_id);
        Ok(session)
    }

    /// Create a new session: start the in-process client, spawn the event-drain
    /// background task, and return handles to both sides.
    async fn create_session(
        &self,
        user_id: &str,
    ) -> anyhow::Result<UserSession> {
        info!(user_id, "Starting in-process codex client");
        let config = Config::load_default_with_cli_overrides(Vec::new())
            .map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?;
        debug!(user_id, "Loaded default codex config for in-process client");

        let args = InProcessClientStartArgs {
            arg0_paths: Default::default(),
            config: Arc::new(config),
            cli_overrides: vec![],
            loader_overrides: LoaderOverrides::default(),
            cloud_requirements: CloudRequirementsLoader::default(),
            feedback: CodexFeedback::new(),
            config_warnings: vec![],
            session_source: SessionSource::Exec,
            enable_codex_api_key_env: true,
            client_name: format!("codex-server-{}", user_id),
            client_version: String::from("0.1.0"),
            experimental_api: true,
            opt_out_notification_methods: vec![],
            channel_capacity: 128,
        };

        let mut client = InProcessAppServerClient::start(args)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start client: {}", e))?;

        info!(user_id, "In-process codex client started");

        // Extract a clonable request handle (does not take ownership)
        let request_handle = client.request_handle();

        // Broadcast channel: capacity 256 for SSE/WS subscribers
        let (event_tx, _) = broadcast::channel::<serde_json::Value>(256);
        let event_tx_bg = event_tx.clone();

        // Spawn background task: drain events and broadcast serialized JSON
        let uid = user_id.to_string();
        tokio::spawn(async move {
            loop {
                match client.next_event().await {
                    Some(event) => {
                        let json = event_to_json(event.into());
                        if let Some(json) = json {
                            // Ignore send error (no subscribers is fine)
                            let _ = event_tx_bg.send(json);
                        }
                    }
                    None => {
                        info!(user_id = %uid, "Codex client event stream ended");
                        // Send disconnected event
                        let disconnected = serde_json::json!({
                            "method": "backend/disconnected",
                            "params": { "message": "codex client disconnected" },
                            "atIso": chrono::Utc::now().to_rfc3339()
                        });
                        let _ = event_tx_bg.send(disconnected);
                        break;
                    }
                }
            }
        });

        Ok(UserSession {
            request_handle,
            event_tx,
        })
    }

    /// Get current settings
    pub async fn get_settings(&self) -> Settings {
        self.settings.read().await.clone()
    }

    /// Update settings
    pub async fn update_settings(&self, update: SettingsUpdate) -> anyhow::Result<()> {
        let mut settings = self.settings.write().await;

        if let Some(codex_home) = update.codex_home {
            settings.codex_home = codex_home;
        }
        if let Some(user_files_path) = update.user_files_path {
            settings.user_files_path = user_files_path;
        }
        if let Some(user_threads_path) = update.user_threads_path {
            settings.user_threads_path = user_threads_path;
        }
        if let Some(sandbox_mode) = update.sandbox_mode {
            settings.sandbox_mode = sandbox_mode;
        }
        if let Some(network_access) = update.network_access {
            settings.network_access = network_access;
        }
        if let Some(exclude_tmpdir) = update.exclude_tmpdir_env_var {
            settings.exclude_tmpdir_env_var = exclude_tmpdir;
        }
        if let Some(exclude_slash) = update.exclude_slash_tmp {
            settings.exclude_slash_tmp = exclude_slash;
        }
        if let Some(markets) = update.markets {
            settings.markets = markets;
        }

        Ok(())
    }
}

/// Convert an `AppServerEvent` into the JSON wire format the frontend expects.
///
/// `ServerNotification` serializes with `#[serde(tag = "method", content = "params")]`
/// producing `{ "method": "...", "params": {...} }` — the exact shape the frontend
/// `toNotification()` helper looks for.
///
/// `ServerRequest` is wrapped as `{ "method": "server/request", "params": { "id", "method", ...params } }`
/// mirroring what the TypeScript bridge does.
fn event_to_json(event: AppServerEvent) -> Option<serde_json::Value> {
    let at_iso = chrono::Utc::now().to_rfc3339();
    match event {
        AppServerEvent::ServerNotification(notification) => {
            match serde_json::to_value(&notification) {
                Ok(mut v) => {
                    if let Some(obj) = v.as_object_mut() {
                        obj.insert("atIso".to_string(), serde_json::Value::String(at_iso));
                    }
                    Some(v)
                }
                Err(e) => {
                    warn!("Failed to serialize server notification: {}", e);
                    None
                }
            }
        }
        AppServerEvent::ServerRequest(request) => {
            // Wrap server requests the same way the TypeScript bridge does
            match wrap_server_request(request, at_iso) {
                Ok(v) => Some(v),
                Err(e) => {
                    warn!("Failed to serialize server request: {}", e);
                    None
                }
            }
        }
        AppServerEvent::Lagged { skipped } => {
            // Inform the frontend that events were dropped
            Some(serde_json::json!({
                "method": "backend/lagged",
                "params": { "skipped": skipped },
                "atIso": at_iso
            }))
        }
        AppServerEvent::Disconnected { message } => {
            Some(serde_json::json!({
                "method": "backend/disconnected",
                "params": { "message": message },
                "atIso": at_iso
            }))
        }
    }
}

fn wrap_server_request(request: ServerRequest, at_iso: String) -> Result<serde_json::Value, serde_json::Error> {
    // ServerRequest serializes as { "method": "...", "id": ..., "params": {...} }
    let v = serde_json::to_value(&request)?;
    let method = v.get("method").and_then(|m| m.as_str()).unwrap_or("unknown").to_string();
    let id = v.get("id").cloned().unwrap_or(serde_json::Value::Null);
    let params = v.get("params").cloned().unwrap_or(serde_json::Value::Null);

    // Merge id and method into params, matching TypeScript bridge behavior
    let mut params_obj = serde_json::Map::new();
    params_obj.insert("id".to_string(), id);
    params_obj.insert("method".to_string(), serde_json::Value::String(method));
    if let Some(p) = params.as_object() {
        for (k, v) in p {
            params_obj.insert(k.clone(), v.clone());
        }
    }

    Ok(serde_json::json!({
        "method": "server/request",
        "params": params_obj,
        "atIso": at_iso
    }))
}
