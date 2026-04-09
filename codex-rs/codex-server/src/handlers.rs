pub mod llm;
use crate::models::*;
use crate::state::AppState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use std::sync::Arc;

use serde_json::Value;
use tracing::{debug, error, info, warn};

// Codex protocol imports
use codex_app_server_protocol::ClientRequest;
use codex_app_server_protocol::RequestId;
use codex_app_server_protocol::CollaborationModeListParams;
use codex_app_server_protocol::ConfigBatchWriteParams;
use codex_app_server_protocol::ConfigReadParams;
use codex_app_server_protocol::ConfigValueWriteParams;
use codex_app_server_protocol::GetAccountParams;
use codex_app_server_protocol::ModelListParams;
use codex_app_server_protocol::SkillsListParams;

// RPC Handler
pub async fn rpc_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RpcRequest>,
) -> impl IntoResponse {
    let params_summary = summarize_json_value(&request.params);
    info!(method = %request.method, params = %params_summary, "RPC request received");

    // Get or create session for this request
    let user_id = "default";

    let session = match state.get_session(user_id).await {
        Ok(session) => session,
        Err(e) => {
            error!(user_id, error = %e, "Failed to get codex session");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::<serde_json::Value>::error(format!(
                    "Failed to initialize codex client: {}",
                    e
                ))),
            )
                .into_response();
        }
    };

    // Parse the request into a ClientRequest
    let client_request = match parse_rpc_request(&request).await {
        Ok(req) => req,
        Err(e) => {
            error!(
                method = %request.method,
                params = %params_summary,
                error = %e,
                "Failed to parse RPC request"
            );
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::<serde_json::Value>::error(format!(
                    "Invalid request: {}",
                    e
                ))),
            )
                .into_response();
        }
    };

    // Execute the request via the clonable request handle
    match session.request_handle.request(client_request).await {
        Ok(result) => {
            match result {
                Ok(json_result) => {
                    let json_result = sanitize_rpc_result(&request.method, json_result).await;
                    info!(method = %request.method, "RPC request completed successfully");
                    Json(RpcResponse { result: json_result }).into_response()
                }
                Err(rpc_error) => {
                    warn!(
                        method = %request.method,
                        error = %rpc_error.message,
                        "RPC request returned protocol error"
                    );
                    Json(ApiResponse::<serde_json::Value>::error(rpc_error.message))
                        .into_response()
                }
            }
        }
        Err(e) => {
            error!(method = %request.method, error = %e, "RPC request failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::<serde_json::Value>::error(format!(
                    "Request failed: {}",
                    e
                ))),
            )
                .into_response()
        }
    }
}

async fn sanitize_rpc_result(method: &str, result: Value) -> Value {
    if method == "thread/list" {
        return prune_missing_thread_summaries(result).await;
    }

    result
}

async fn prune_missing_thread_summaries(mut result: Value) -> Value {
    let Some(entries) = result
        .get_mut("data")
        .and_then(Value::as_array_mut)
    else {
        return result;
    };

    let mut retained = Vec::with_capacity(entries.len());
    let mut removed = Vec::new();

    for entry in entries.drain(..) {
        let Some(path) = entry.get("path").and_then(Value::as_str) else {
            retained.push(entry);
            continue;
        };

        match tokio::fs::try_exists(path).await {
            Ok(true) => retained.push(entry),
            Ok(false) => {
                let thread_id = entry
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("<unknown>")
                    .to_string();
                removed.push((thread_id, path.to_string()));
            }
            Err(error) => {
                warn!(path, error = %error, "Failed to verify thread rollout path; keeping summary");
                retained.push(entry);
            }
        }
    }

    if !removed.is_empty() {
        info!(
            removed_count = removed.len(),
            removed_threads = ?removed,
            "Filtered stale thread summaries with missing rollout files"
        );
    }

    *entries = retained;
    result
}

fn summarize_json_value(value: &serde_json::Value) -> String {
    const MAX_LEN: usize = 512;

    if value.is_null() {
        return String::from("null");
    }

    match serde_json::to_string(value) {
        Ok(serialized) if serialized.len() > MAX_LEN => {
            format!("{}...<truncated {} chars>", &serialized[..MAX_LEN], serialized.len() - MAX_LEN)
        }
        Ok(serialized) => serialized,
        Err(error) => format!("<failed to serialize params: {error}>"),
    }
}

/// Parse RPC request into ClientRequest
async fn parse_rpc_request(request: &RpcRequest) -> anyhow::Result<ClientRequest> {
    // Generate a unique request ID
    let request_id = RequestId::Integer(rand::random::<i64>());
    debug!(method = %request.method, request_id = ?request_id, "Parsing RPC request");

    // Parse based on method
    let client_request = match request.method.as_str() {
        "thread/start" => {
            let params = serde_json::from_value(request.params.clone())?;
            ClientRequest::ThreadStart { request_id, params }
        }
        "thread/resume" => {
            let params = serde_json::from_value(request.params.clone())?;
            ClientRequest::ThreadResume { request_id, params }
        }
        "thread/fork" => {
            let params = serde_json::from_value(request.params.clone())?;
            ClientRequest::ThreadFork { request_id, params }
        }
        "thread/archive" => {
            let params = serde_json::from_value(request.params.clone())?;
            ClientRequest::ThreadArchive { request_id, params }
        }
        "thread/name/set" => {
            let params = serde_json::from_value(request.params.clone())?;
            ClientRequest::ThreadSetName { request_id, params }
        }
        "thread/rollback" => {
            let params = serde_json::from_value(request.params.clone())?;
            ClientRequest::ThreadRollback { request_id, params }
        }
        "thread/list" => {
            let params = serde_json::from_value(request.params.clone())?;
            ClientRequest::ThreadList { request_id, params }
        }
        "thread/read" => {
            let params = serde_json::from_value(request.params.clone())?;
            ClientRequest::ThreadRead { request_id, params }
        }
        "turn/start" => {
            let params = serde_json::from_value(request.params.clone())?;
            ClientRequest::TurnStart { request_id, params }
        }
        "turn/interrupt" => {
            let params = serde_json::from_value(request.params.clone())?;
            ClientRequest::TurnInterrupt { request_id, params }
        }
        "config/read" => {
            let params = ConfigReadParams {
                include_layers: false,
                cwd: None,
            };
            ClientRequest::ConfigRead { request_id, params }
        }
        "skills/list" => {
            let params = SkillsListParams {
                cwds: vec![],
                force_reload: false,
                per_cwd_extra_user_roots: None,
            };
            ClientRequest::SkillsList { request_id, params }
        }
        "account/read" => {
            let params = GetAccountParams { refresh_token: false };
            ClientRequest::GetAccount { request_id, params }
        }
        "account/rateLimits/read" => {
            ClientRequest::GetAccountRateLimits {
                request_id,
                params: None,
            }
        }
        "collaborationMode/list" => {
            let params = if request.params.is_null() {
                CollaborationModeListParams::default()
            } else {
                serde_json::from_value(request.params.clone())?
            };
            ClientRequest::CollaborationModeList { request_id, params }
        }
        "model/list" => {
            let params = ModelListParams {
                cursor: None,
                include_hidden: Some(false),
                limit: None,
            };
            ClientRequest::ModelList { request_id, params }
        }
        "config/value/write" => {
            let params: ConfigValueWriteParams = serde_json::from_value(request.params.clone())?;
            ClientRequest::ConfigValueWrite { request_id, params }
        }
        "config/batchWrite" => {
            let params: ConfigBatchWriteParams = serde_json::from_value(request.params.clone())?;
            ClientRequest::ConfigBatchWrite { request_id, params }
        }
        // Add more methods as needed
        _ => {
            warn!(method = %request.method, "Unknown RPC method received");
            return Err(anyhow::anyhow!("Unknown method: {}", request.method));
        }
    };

    Ok(client_request)
}

// Events (SSE) Handler — streams real codex events to the frontend
pub async fn events_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    use axum::body::Body;
    use tokio_stream::wrappers::BroadcastStream;
    use tokio_stream::StreamExt as _;

    let user_id = "default";
    let broadcast_tx = match state.get_session(user_id).await {
        Ok(session) => session.event_tx,
        Err(e) => {
            error!(error = %e, "Failed to get session for SSE");
            // Return a disconnected event as response body and close
            let body = format!(
                "data: {}\n\n",
                serde_json::json!({ "method": "backend/disconnected", "params": { "message": format!("{}", e) } })
            );
            return (
                StatusCode::OK,
                [
                    ("Content-Type", "text/event-stream"),
                    ("Cache-Control", "no-cache"),
                    ("Connection", "keep-alive"),
                ],
                body,
            ).into_response();
        }
    };

    let rx = broadcast_tx.subscribe();
    let event_stream = BroadcastStream::new(rx).filter_map(|result| {
        match result {
            Ok(json) => {
                let data = format!("data: {}\n\n", json);
                Some(Ok::<axum::body::Bytes, std::convert::Infallible>(
                    axum::body::Bytes::from(data),
                ))
            }
            Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(n)) => {
                warn!("SSE subscriber lagged, dropped {} events", n);
                None
            }
        }
    });

    let headers = [
        ("Content-Type", "text/event-stream"),
        ("Cache-Control", "no-cache"),
        ("Connection", "keep-alive"),
    ];

    (headers, Body::from_stream(event_stream)).into_response()
}

// Pending requests handler
pub async fn pending_requests_handler() -> impl IntoResponse {
    Json(PendingRequestsResponse { data: vec![] })
}

// Respond to request handler
pub async fn respond_request_handler(
    Json(_request): Json<RespondRequest>,
) -> impl IntoResponse {
    Json(ApiResponse::<serde_json::Value>::ok())
}

// Meta handlers
pub async fn meta_methods_handler() -> impl IntoResponse {
    // These are handled by RPC now, but keeping for compatibility
    Json(ApiResponse::success(vec![
        "initialize",
        "thread/start",
        "thread/resume",
        "thread/fork",
        "thread/archive",
        "thread/name/set",
        "thread/rollback",
        "thread/list",
        "thread/read",
        "turn/start",
        "turn/interrupt",
        "config/read",
        "skills/list",
        "account/read",
        "account/rateLimits/read",
        "collaborationMode/list",
        "model/list",
    ]))
}

pub async fn meta_notifications_handler() -> impl IntoResponse {
    Json(ApiResponse::success(vec![
        "turn/completed",
        "item/completed",
        "agent/message_delta",
        "backend/disconnected",
    ]))
}

// Provider models handler
pub async fn provider_models_handler() -> impl IntoResponse {
    Json(ProviderModelsResponse {
        data: vec![
            "kimi-for-coding".to_string(),
            "kimi-k2.5".to_string(),
            "gpt-4o".to_string(),
        ],
        provider_id: "kimi".to_string(),
        source: "provider".to_string(),
    })
}

// Home directory handler
pub async fn home_directory_handler() -> impl IntoResponse {
    let home = dirs::home_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "/".to_string());

    Json(ApiResponse::success(HomeDirectoryResponse { path: home }))
}

// Workspace roots handlers
pub async fn get_workspace_roots_handler(
    State(_state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Return empty state for now
    Json(ApiResponse::success(WorkspaceRootsState {
        order: vec![],
        labels: Default::default(),
        active: vec![],
    }))
}

pub async fn put_workspace_roots_handler(
    State(_state): State<Arc<AppState>>,
    Json(_update): Json<WorkspaceRootsState>,
) -> impl IntoResponse {
    // Update would be saved here
    Json(ApiResponse::<serde_json::Value>::ok())
}

// Project handlers
pub async fn create_project_handler(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<CreateProjectRequest>,
) -> impl IntoResponse {
    // Create directory if requested
    if request.create_if_missing {
        let _ = tokio::fs::create_dir_all(&request.path).await;
    }

    Json(ApiResponse::success(CreateProjectResponse {
        path: request.path,
    }))
}

pub async fn delete_project_handler(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<DeleteProjectRequest>,
) -> impl IntoResponse {
    let _ = tokio::fs::remove_dir_all(&request.path).await;
    Json(ApiResponse::<serde_json::Value>::ok())
}

// Directory browser handler
pub async fn browse_directory_handler(
    Query(query): Query<BrowseDirectoryQuery>,
) -> impl IntoResponse {
    let path = std::path::Path::new(&query.path);

    match tokio::fs::read_dir(path).await {
        Ok(mut entries) => {
            let mut result_entries = vec![];

            while let Ok(Some(entry)) = entries.next_entry().await {
                let metadata = entry.metadata().await.ok();
                let is_directory = metadata.map(|m| m.is_dir()).unwrap_or(false);

                result_entries.push(DirectoryEntry {
                    name: entry.file_name().to_string_lossy().to_string(),
                    path: entry.path().to_string_lossy().to_string(),
                    is_directory,
                });
            }

            Json(ApiResponse::success(DirectoryBrowseResponse {
                path: query.path.clone(),
                parent_path: path.parent().map(|p| p.to_string_lossy().to_string()),
                entries: result_entries,
            })).into_response()
        }
        Err(_) => {
            (StatusCode::BAD_REQUEST, Json(ApiResponse::<DirectoryBrowseResponse>::error("Failed to read directory"))).into_response()
        }
    }
}

// Upload file handler
pub async fn upload_file_handler() -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, "File upload not yet implemented")
}

// Skills handlers
pub async fn skills_assets_handler(Path(_path): Path<String>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, "Skills assets not yet implemented")
}

pub async fn skills_hub_handler(Query(_query): Query<SkillsHubQuery>) -> impl IntoResponse {
    Json(ApiResponse::success(SkillsHubResponse {
        data: vec![],
        installed: vec![],
        system_installed: None,
        total: 0,
        partial_error: None,
    }))
}

pub async fn skills_readme_handler(
    Query(_query): Query<SkillsReadmeQuery>,
) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, "Skills readme not yet implemented")
}

pub async fn skills_install_handler(
    Json(_request): Json<SkillsInstallRequest>,
) -> impl IntoResponse {
    Json(ApiResponse::<serde_json::Value>::ok())
}

pub async fn skills_uninstall_handler(
    Json(_request): Json<SkillsUninstallRequest>,
) -> impl IntoResponse {
    Json(ApiResponse::<serde_json::Value>::ok())
}

// Composer file search handler
pub async fn composer_file_search_handler(
    Json(_request): Json<ComposerFileSearchRequest>,
) -> impl IntoResponse {
    Json(ApiResponse::success(ComposerFileSearchResponse { data: vec![] }))
}

// Review handlers
pub async fn review_snapshot_handler(
    Query(_query): Query<ReviewSnapshotQuery>,
) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, "Review snapshot not yet implemented")
}

pub async fn review_action_handler(
    Json(_request): Json<ReviewActionRequest>,
) -> impl IntoResponse {
    Json(ApiResponse::<serde_json::Value>::ok())
}

pub async fn review_git_init_handler(
    Json(_request): Json<ReviewGitInitRequest>,
) -> impl IntoResponse {
    Json(ApiResponse::<serde_json::Value>::ok())
}

// Settings handlers
pub async fn get_settings_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let settings = state.get_settings().await;

    Json(ApiResponse::success(SettingsResponse {
        codex_home: settings.codex_home.clone(),
        saved_codex_home: None,
        default_codex_home: "/tmp/codex".to_string(),
        skills_dir: format!("{}/skills", settings.codex_home),
        memories_dir: format!("{}/memories", settings.codex_home),
        sessions_dir: format!("{}/sessions", settings.codex_home),
        archived_sessions_dir: format!("{}/archived_sessions", settings.codex_home),
        subdirectories: vec![],
        settings_file: format!("{}/settings.json", settings.codex_home),
        user_files_path: settings.user_files_path.clone(),
        saved_user_files_path: None,
        default_user_files_path: "/tmp/user_files".to_string(),
        user_threads_path: settings.user_threads_path.clone(),
        saved_user_threads_path: None,
        default_user_threads_path: "/tmp/user_threads".to_string(),
        sandbox_mode: settings.sandbox_mode.clone(),
        saved_sandbox_mode: None,
        default_sandbox_mode: "workspace-write".to_string(),
        network_access: settings.network_access,
        exclude_tmpdir_env_var: settings.exclude_tmpdir_env_var,
        exclude_slash_tmp: settings.exclude_slash_tmp,
        markets: settings.markets.iter().map(|m| Market {
            owner: m.owner.clone(),
            repo: m.repo.clone(),
            active: m.active,
        }).collect(),
    }))
}

pub async fn put_settings_handler(
    State(state): State<Arc<AppState>>,
    Json(update): Json<SettingsUpdate>,
) -> impl IntoResponse {
    // Validate sandbox mode
    if let Some(ref mode) = update.sandbox_mode {
        if mode != "workspace-write" && mode != "danger-full-access" {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse::<serde_json::Value>::error(
                    "sandboxMode must be 'workspace-write' or 'danger-full-access'",
                )),
            )
                .into_response();
        }
    }

    match state.update_settings(update).await {
        Ok(_) => Json(ApiResponse::<serde_json::Value>::ok()).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<serde_json::Value>::error(e.to_string())),
        )
            .into_response(),
    }
}

pub async fn reload_settings_handler() -> impl IntoResponse {
    Json(ApiResponse::success(SettingsReloadResponse {
        ok: true,
        hot_reload_applied: true,
        restart_required: false,
        restart_recommended: false,
        message: "Settings hot-reloaded safely.".to_string(),
    }))
}

// User files handlers
pub async fn user_files_write_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<UserFilesWriteRequest>,
) -> impl IntoResponse {
    let settings = state.get_settings().await;
    let base_path = std::path::Path::new(&settings.user_files_path);
    let full_path = base_path.join(&request.path);

    // Ensure path is within user_files directory
    if !full_path.starts_with(base_path) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<serde_json::Value>::error(
                "Path must be inside the user_files directory",
            )),
        )
            .into_response();
    }

    // Create parent directories
    if let Some(parent) = full_path.parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }

    match tokio::fs::write(&full_path, &request.content).await {
        Ok(_) => Json(serde_json::json!({
            "ok": true,
            "path": full_path.to_string_lossy().to_string(),
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<serde_json::Value>::error(format!(
                "Failed to write file: {}",
                e
            ))),
        )
            .into_response(),
    }
}

pub async fn user_files_list_handler(
    State(state): State<Arc<AppState>>,
    Query(query): Query<UserFilesListQuery>,
) -> impl IntoResponse {
    let settings = state.get_settings().await;
    let base_path = std::path::Path::new(&settings.user_files_path);
    let target_path = if query.path.is_empty() {
        base_path.to_path_buf()
    } else {
        base_path.join(&query.path)
    };

    // Ensure path is within user_files directory
    if !target_path.starts_with(base_path) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<serde_json::Value>::error(
                "Path must be inside the user_files directory",
            )),
        )
            .into_response();
    }

    match tokio::fs::read_dir(&target_path).await {
        Ok(mut entries) => {
            let mut result_entries = vec![];

            while let Ok(Some(entry)) = entries.next_entry().await {
                let metadata = entry.metadata().await.ok();
                let is_directory = metadata.map(|m| m.is_dir()).unwrap_or(false);

                result_entries.push(UserFileEntry {
                    name: entry.file_name().to_string_lossy().to_string(),
                    is_directory,
                    path: entry.path().to_string_lossy().to_string(),
                });
            }

            Json(ApiResponse::success(UserFilesListResponse {
                path: target_path.to_string_lossy().to_string(),
                entries: result_entries,
            }))
            .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<serde_json::Value>::error(format!(
                "Failed to list files: {}",
                e
            ))),
        )
            .into_response(),
    }
}
