//! Configuration for LLM provider in-process routing.
//!
//! This module provides a global hook that allows the app-server to register
//! its provider factory, enabling in-process LLM calls instead of HTTP proxy.

use std::any::Any;
use std::sync::Arc;
use std::sync::OnceLock;

/// Factory trait for creating LLM providers.
///
/// This is implemented by the app-server to route LLM requests to providers.
pub trait LlmProviderFactory: Send + Sync {
    /// Route a request to the appropriate provider based on model
    fn route_request(
        &self,
        model: &str,
    ) -> Option<Arc<dyn LlmProvider>>;

    /// List available models
    fn list_models(&self) -> Vec<ModelInfo>;

    /// Returns self as Any for downcasting
    fn as_any(&self) -> &dyn Any;
}

/// LLM Provider trait for making chat completion requests.
pub trait LlmProvider: Send + Sync {
    /// Provider name
    fn name(&self) -> &'static str;

    /// Supported models
    fn supported_models(&self) -> &[&'static str];

    /// Make a chat completion request (non-streaming)
    fn chat(
        &self,
        request: ChatRequest,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<ChatResponse, ProviderError>>
                + Send
                + '_,
        >,
    >;

    /// Make a streaming chat completion request
    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        futures::stream::BoxStream<'static, Result<ChatResponseChunk, ProviderError>>,
                        ProviderError,
                    >,
                > + Send
                + '_,
        >,
    >;
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

/// Chat request (OpenAI-compatible)
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub tool_choice: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
}

/// Chat message
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Tool definition
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: Function,
}

/// Function definition
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Function {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Tool call in response
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionCall,
}

/// Function call details
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Chat response (OpenAI-compatible)
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

/// Choice in response
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

/// Streaming response chunk
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ChatResponseChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

/// Choice in streaming chunk
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Delta in streaming chunk
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Delta {
    pub role: String,
    pub content: String,
}

/// Usage statistics
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Provider error
#[derive(Debug)]
pub struct ProviderError {
    pub message: String,
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ProviderError {}

// Global provider factory storage
static GLOBAL_PROVIDER_FACTORY: OnceLock<Arc<dyn LlmProviderFactory>> = OnceLock::new();

/// Set the global LLM provider factory.
///
/// This should be called by the app-server during initialization to enable
/// in-process LLM routing. Can only be set once.
///
/// Returns true if the factory was set, false if it was already set.
pub fn set_llm_provider_factory(factory: Arc<dyn LlmProviderFactory>) -> bool {
    GLOBAL_PROVIDER_FACTORY.set(factory).is_ok()
}

/// Get the global LLM provider factory if one has been set.
pub fn get_llm_provider_factory() -> Option<Arc<dyn LlmProviderFactory>> {
    GLOBAL_PROVIDER_FACTORY.get().cloned()
}

/// Check if in-process LLM routing is enabled.
#[allow(dead_code)]
pub fn is_in_process_llm_enabled() -> bool {
    GLOBAL_PROVIDER_FACTORY.get().is_some()
}
