use axum::{
    response::{IntoResponse, Json},
    routing::{delete, get, post, put},
    Router,
};
use std::sync::Arc;
use tracing::info;
use tracing::Level;
use tracing_subscriber::EnvFilter;

// Module declarations
mod handlers;
mod models;
mod providers;
mod state;
mod websocket;

use handlers::*;
use state::AppState;

const TOKIO_WORKER_STACK_SIZE: usize = 8 * 1024 * 1024;

fn main() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("codex-server-worker")
        .thread_stack_size(TOKIO_WORKER_STACK_SIZE)
        .build()?
        .block_on(async_main())
}

async fn async_main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .or_else(|_| EnvFilter::try_new("info,codex_server=debug,tower_http=info"))
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(true)
        .init();

    info!(
        "Starting Codex Server with Tokio worker stack size {} bytes...",
        TOKIO_WORKER_STACK_SIZE
    );

    // Load configuration
    let config = load_config().await?;
    info!(
        port = config.port,
        codex_home = %config.codex_home,
        user_files_path = %config.user_files_path,
        user_threads_path = %config.user_threads_path,
        kimi_api_key_configured = config.kimi_api_key.is_some(),
        azure_openai_endpoint_configured = config.azure_openai_endpoint.is_some(),
        azure_openai_key_configured = config.azure_openai_key.is_some(),
        "Loaded codex-server configuration"
    );

    // Create application state
    let state = Arc::new(AppState::new(config).await?);

    // Build router
    let app = create_router(state);

    // Start server
    let port = std::env::var("PORT").unwrap_or_else(|_| "3457".to_string());
    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    info!("Codex Server listening on http://{}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}

fn create_router(state: Arc<AppState>) -> Router {
    // CORS layer
    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);
    let http_trace = tower_http::trace::TraceLayer::new_for_http()
        .make_span_with(tower_http::trace::DefaultMakeSpan::new().level(Level::INFO))
        .on_request(tower_http::trace::DefaultOnRequest::new().level(Level::INFO))
        .on_response(tower_http::trace::DefaultOnResponse::new().level(Level::INFO))
        .on_failure(tower_http::trace::DefaultOnFailure::new().level(Level::ERROR));

    Router::new()
        // Health check
        .route("/health", get(health_handler))
        // OpenAI-compatible LLM API (internal)
        .route("/v1/models", get(handlers::llm::list_models_handler))
        .route("/v1/chat/completions", post(handlers::llm::chat_completions_handler))
        // RPC endpoint
        .route("/codex-api/rpc", post(rpc_handler))
        // Events (SSE)
        .route("/codex-api/events", get(events_handler))
        // WebSocket
        .route("/codex-api/ws", get(websocket::ws_handler))
        // Server requests
        .route("/codex-api/server-requests/pending", get(pending_requests_handler))
        .route("/codex-api/server-requests/respond", post(respond_request_handler))
        // Meta
        .route("/codex-api/meta/methods", get(meta_methods_handler))
        .route("/codex-api/meta/notifications", get(meta_notifications_handler))
        // Provider models
        .route("/codex-api/provider-models", get(provider_models_handler))
        // Home directory
        .route("/codex-api/home-directory", get(home_directory_handler))
        // Workspace roots
        .route("/codex-api/workspace-roots-state", get(get_workspace_roots_handler))
        .route("/codex-api/workspace-roots-state", put(put_workspace_roots_handler))
        // Projects
        .route("/codex-api/project-root", post(create_project_handler))
        .route("/codex-api/project", delete(delete_project_handler))
        // Directory browser
        .route("/codex-api/browse-directory", get(browse_directory_handler))
        // File upload
        .route("/codex-api/upload-file", post(upload_file_handler))
        // Skills
                .route("/codex-api/skills/{*path}", get(skills_assets_handler))
        .route("/codex-api/skills-hub", get(skills_hub_handler))
        .route("/codex-api/skills-hub/readme", get(skills_readme_handler))
        .route("/codex-api/skills-hub/install", post(skills_install_handler))
        .route("/codex-api/skills-hub/uninstall", post(skills_uninstall_handler))
        // Composer file search
        .route("/codex-api/composer-file-search", post(composer_file_search_handler))
        // Review
        .route("/codex-api/review/snapshot", get(review_snapshot_handler))
        .route("/codex-api/review/action", post(review_action_handler))
        .route("/codex-api/review/git/init", post(review_git_init_handler))
        // Settings
        .route("/codex-api/settings", get(get_settings_handler))
        .route("/codex-api/settings", put(put_settings_handler))
        .route("/codex-api/settings/reload", post(reload_settings_handler))
        // User files
        .route("/codex-api/user-files/write", post(user_files_write_handler))
        .route("/codex-api/user-files/list", get(user_files_list_handler))
        // Static files (dist)
 .fallback_service(tower_http::services::ServeDir::new("dist"))
        // Add CORS and state
        .layer(http_trace)
        .layer(cors)
        .with_state(state)
}

async fn load_config() -> anyhow::Result<ServerConfig> {
    dotenvy::dotenv().ok();

    Ok(ServerConfig {
        port: std::env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(3457),
        codex_home: std::env::var("CODEXUI_CODEX_HOME")
            .unwrap_or_else(|_| String::from("/tmp/codex")),
        user_files_path: std::env::var("CODEXUI_USER_FILES_PATH")
            .unwrap_or_else(|_| String::from("/tmp/user_files")),
        user_threads_path: std::env::var("CODEXUI_USER_THREADS_PATH")
            .unwrap_or_else(|_| String::from("/tmp/user_threads")),
        kimi_api_key: std::env::var("KIMI_API_KEY").ok(),
        azure_openai_endpoint: std::env::var("AZURE_OPENAI_ENDPOINT").ok(),
        azure_openai_key: std::env::var("AZURE_OPENAI_API_KEY").ok(),
    })
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub port: u16,
    pub codex_home: String,
    pub user_files_path: String,
    pub user_threads_path: String,
    pub kimi_api_key: Option<String>,
    pub azure_openai_endpoint: Option<String>,
    pub azure_openai_key: Option<String>,
}

// Health check handler
async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok"
    }))
}
