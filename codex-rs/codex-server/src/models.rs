use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Common response wrapper
#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ok: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            data: Some(data),
            ok: None,
            error: None,
        }
    }

    pub fn ok() -> Self {
        Self {
            data: None,
            ok: Some(true),
            error: None,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            data: None,
            ok: None,
            error: Some(message.into()),
        }
    }
}

// RPC Request/Response
#[derive(Debug, Deserialize)]
pub struct RpcRequest {
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct RpcResponse {
    pub result: serde_json::Value,
}

// Settings
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsResponse {
    pub codex_home: String,
    pub saved_codex_home: Option<String>,
    pub default_codex_home: String,
    pub skills_dir: String,
    pub memories_dir: String,
    pub sessions_dir: String,
    pub archived_sessions_dir: String,
    pub subdirectories: Vec<SubdirectoryInfo>,
    pub settings_file: String,
    pub user_files_path: String,
    pub saved_user_files_path: Option<String>,
    pub default_user_files_path: String,
    pub user_threads_path: String,
    pub saved_user_threads_path: Option<String>,
    pub default_user_threads_path: String,
    pub sandbox_mode: String,
    pub saved_sandbox_mode: Option<String>,
    pub default_sandbox_mode: String,
    pub network_access: bool,
    pub exclude_tmpdir_env_var: bool,
    pub exclude_slash_tmp: bool,
    pub markets: Vec<Market>,
}

#[derive(Debug, Serialize)]
pub struct SubdirectoryInfo {
    pub name: String,
    pub path: String,
    pub exists: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permissions: Option<String>,
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Market {
    pub owner: String,
    pub repo: String,
    pub active: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codex_home: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_files_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_threads_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sandbox_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network_access: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude_tmpdir_env_var: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude_slash_tmp: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub markets: Option<Vec<Market>>,
}

// Workspace roots
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkspaceRootsState {
    pub order: Vec<String>,
    pub labels: HashMap<String, String>,
    pub active: Vec<String>,
}

// Directory browser
#[derive(Debug, Serialize)]
pub struct DirectoryBrowseResponse {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_path: Option<String>,
    pub entries: Vec<DirectoryEntry>,
}

#[derive(Debug, Serialize)]
pub struct DirectoryEntry {
    pub name: String,
    pub path: String,
    pub is_directory: bool,
}

#[derive(Debug, Deserialize)]
pub struct BrowseDirectoryQuery {
    #[serde(default = "default_path")]
    pub path: String,
}

fn default_path() -> String {
    "/".to_string()
}

// Project
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct CreateProjectRequest {
    pub path: String,
    #[serde(default)]
    pub create_if_missing: bool,
    #[serde(default)]
    pub label: String,
}

#[derive(Debug, Serialize)]
pub struct CreateProjectResponse {
    pub path: String,
}

#[derive(Debug, Deserialize)]
pub struct DeleteProjectRequest {
    pub path: String,
}

// User files
#[derive(Debug, Deserialize)]
pub struct UserFilesWriteRequest {
    pub path: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct UserFilesWriteResponse {
    pub ok: bool,
    pub path: String,
}

#[derive(Debug, Serialize)]
pub struct UserFilesListResponse {
    pub path: String,
    pub entries: Vec<UserFileEntry>,
}

#[derive(Debug, Serialize)]
pub struct UserFileEntry {
    pub name: String,
    pub is_directory: bool,
    pub path: String,
}

#[derive(Debug, Deserialize)]
pub struct UserFilesListQuery {
    #[serde(default)]
    pub path: String,
}

// Home directory
#[derive(Debug, Serialize)]
pub struct HomeDirectoryResponse {
    pub path: String,
}

// Meta
#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct MetaMethodsResponse {
    pub data: Vec<String>,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct MetaNotificationsResponse {
    pub data: Vec<String>,
}

// Provider models
#[derive(Debug, Serialize)]
pub struct ProviderModelsResponse {
    pub data: Vec<String>,
    pub provider_id: String,
    pub source: String,
}

// Skills hub
#[derive(Debug, Serialize)]
pub struct SkillsHubResponse {
    pub data: Vec<SkillHubEntry>,
    pub installed: Vec<SkillEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_installed: Option<Vec<SkillEntry>>,
    pub total: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SkillHubEntry {
    pub name: String,
    pub owner: String,
    pub description: String,
    pub display_name: String,
    pub published_at: u64,
    pub avatar_url: String,
    pub url: String,
    pub installed: bool,
    pub market_owner: String,
    pub market_repo: String,
}

#[derive(Debug, Serialize)]
pub struct SkillEntry {
    pub name: String,
    pub path: String,
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct SkillsHubQuery {
    #[serde(default)]
    pub q: String,
    #[serde(default)]
    pub sort: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    100
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct SkillsReadmeQuery {
    pub owner: String,
    pub name: String,
    #[serde(default)]
    pub installed: bool,
    #[serde(default)]
    pub path: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct SkillsInstallRequest {
    pub owner: String,
    pub name: String,
    #[serde(default)]
    pub market_owner: String,
    #[serde(default)]
    pub market_repo: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct SkillsUninstallRequest {
    pub name: String,
    #[serde(default)]
    pub path: String,
}

// Review
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ReviewSnapshotQuery {
    pub cwd: String,
    #[serde(default)]
    pub scope: String,
    #[serde(default)]
    pub workspace_view: String,
    #[serde(default)]
    pub base_branch: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ReviewActionRequest {
    // Fields depend on action type
    #[serde(flatten)]
    pub params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ReviewGitInitRequest {
    pub cwd: String,
}

// File search
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ComposerFileSearchRequest {
    pub cwd: String,
    #[serde(default)]
    pub query: String,
    #[serde(default = "default_search_limit")]
    pub limit: usize,
}

fn default_search_limit() -> usize {
    20
}

#[derive(Debug, Serialize)]
pub struct ComposerFileSearchResponse {
    pub data: Vec<FileSearchResult>,
}

#[derive(Debug, Serialize)]
pub struct FileSearchResult {
    pub path: String,
}

// Server requests
#[derive(Debug, Serialize)]
pub struct PendingRequestsResponse {
    pub data: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct RespondRequest {
    pub id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
}

// Settings reload response
#[derive(Debug, Serialize)]
pub struct SettingsReloadResponse {
    pub ok: bool,
    pub hot_reload_applied: bool,
    pub restart_required: bool,
    pub restart_recommended: bool,
    pub message: String,
}
