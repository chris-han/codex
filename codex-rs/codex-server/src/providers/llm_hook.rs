//! Bridge between codex-server's ProviderFactory and codex-client's LlmProviderFactory hook.
//!
//! This allows the embedded codex CLI to route LLM requests through codex-server's
//! in-process providers (Kimi, Azure) instead of going out via HTTP proxy.

use super::factory::ProviderFactory;
use super::types as server_types;
use codex_client::ChatRequest;
use codex_client::ChatResponse;
use codex_client::ChatResponseChunk;
use codex_client::ChunkChoice;
use codex_client::Choice;
use codex_client::Delta;
use codex_client::FunctionCall;
use codex_client::LlmProvider;
use codex_client::LlmProviderFactory;
use codex_client::Message;
use codex_client::ModelInfo;
use codex_client::ProviderError;
use codex_client::Tool;
use codex_client::ToolCall;
use codex_client::Usage;
use futures::StreamExt;
use std::any::Any;
use std::sync::Arc;

/// Adapter that wraps a server LLMProvider to expose the codex-client LlmProvider trait.
struct ProviderAdapter {
    inner: Arc<dyn server_types::LLMProvider>,
    name: &'static str,
    models: &'static [&'static str],
}

impl LlmProvider for ProviderAdapter {
    fn name(&self) -> &'static str {
        self.name
    }

    fn supported_models(&self) -> &[&'static str] {
        self.models
    }

    fn chat(
        &self,
        request: ChatRequest,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<ChatResponse, ProviderError>>
                + Send
                + '_,
        >,
    > {
        let inner = self.inner.clone();
        Box::pin(async move {
            let server_req = to_server_request(request);
            let resp = inner
                .chat(server_req)
                .await
                .map_err(|e| ProviderError { message: e.to_string() })?;
            Ok(from_server_response(resp))
        })
    }

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
    > {
        let inner = self.inner.clone();
        Box::pin(async move {
            let server_req = to_server_request(request);
            let stream = inner
                .chat_stream(server_req)
                .await
                .map_err(|e| ProviderError { message: e.to_string() })?;

            let mapped: futures::stream::BoxStream<
                'static,
                Result<ChatResponseChunk, ProviderError>,
            > = Box::pin(stream.map(|result| {
                result
                    .map(from_server_chunk)
                    .map_err(|e| ProviderError { message: e.to_string() })
            }));
            Ok(mapped)
        })
    }
}

/// Adapter that wraps ProviderFactory to expose the codex-client LlmProviderFactory trait.
pub struct FactoryAdapter {
    factory: ProviderFactory,
}

impl FactoryAdapter {
    pub fn new(factory: ProviderFactory) -> Self {
        Self { factory }
    }
}

impl LlmProviderFactory for FactoryAdapter {
    fn route_request(&self, model: &str) -> Option<Arc<dyn LlmProvider>> {
        // Only route if the model is in our known set
        let models = self.factory.list_models();
        let known = models.iter().any(|m| m.id == model);
        if !known {
            return None;
        }
        let inner = self.factory.get_provider(model);
        // We need static lifetime for name/models — use leaked refs for simplicity.
        // Since FactoryAdapter is set once via OnceLock, this is acceptable.
        let name: &'static str = Box::leak(inner.name().to_string().into_boxed_str());
        let supported: Vec<&'static str> = inner
            .supported_models()
            .iter()
            .map(|s| -> &'static str { Box::leak(s.to_string().into_boxed_str()) })
            .collect();
        let models_slice: &'static [&'static str] = Box::leak(supported.into_boxed_slice());
        Some(Arc::new(ProviderAdapter {
            inner,
            name,
            models: models_slice,
        }))
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        self.factory
            .list_models()
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                object: m.object,
                created: m.created,
                owned_by: m.owned_by,
            })
            .collect()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ─── Type conversions ────────────────────────────────────────────────────────

fn to_server_request(req: ChatRequest) -> server_types::ChatRequest {
    server_types::ChatRequest {
        model: req.model,
        messages: req.messages.into_iter().map(to_server_message).collect(),
        tools: req.tools.map(|tools| tools.into_iter().map(to_server_tool).collect()),
        tool_choice: req.tool_choice,
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        stream: req.stream,
        top_p: req.top_p,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
    }
}

fn to_server_message(msg: Message) -> server_types::Message {
    let role = match msg.role.as_str() {
        "system" => server_types::Role::System,
        "assistant" => server_types::Role::Assistant,
        "tool" => server_types::Role::Tool,
        _ => server_types::Role::User,
    };
    server_types::Message {
        role,
        content: msg.content,
        name: None,
        tool_calls: msg.tool_calls.map(|tcs| {
            tcs.into_iter()
                .map(|tc| server_types::ToolCall {
                    id: tc.id,
                    r#type: tc.tool_type,
                    function: server_types::FunctionCall {
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                    },
                })
                .collect()
        }),
        tool_call_id: msg.tool_call_id,
    }
}

fn to_server_tool(tool: Tool) -> server_types::Tool {
    server_types::Tool {
        r#type: tool.tool_type,
        function: server_types::Function {
            name: tool.function.name,
            description: tool.function.description.unwrap_or_default(),
            parameters: tool.function.parameters.unwrap_or(serde_json::Value::Null),
        },
    }
}

fn from_server_response(resp: server_types::ChatResponse) -> ChatResponse {
    ChatResponse {
        id: resp.id,
        object: resp.object,
        created: resp.created,
        model: resp.model,
        choices: resp
            .choices
            .into_iter()
            .map(|c| Choice {
                index: c.index,
                message: from_server_message(c.message),
                finish_reason: c.finish_reason,
                logprobs: None,
            })
            .collect(),
        usage: Usage {
            prompt_tokens: resp.usage.prompt_tokens,
            completion_tokens: resp.usage.completion_tokens,
            total_tokens: resp.usage.total_tokens,
        },
    }
}

fn from_server_message(msg: server_types::Message) -> Message {
    let role = match &msg.role {
        server_types::Role::System => "system".to_string(),
        server_types::Role::User => "user".to_string(),
        server_types::Role::Assistant => "assistant".to_string(),
        server_types::Role::Tool => "tool".to_string(),
    };
    Message {
        role,
        content: msg.content,
        tool_calls: msg.tool_calls.map(|tcs| {
            tcs.into_iter()
                .map(|tc| ToolCall {
                    id: tc.id,
                    tool_type: tc.r#type,
                    function: FunctionCall {
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                    },
                })
                .collect()
        }),
        tool_call_id: msg.tool_call_id,
    }
}

fn from_server_chunk(chunk: server_types::ChatResponseChunk) -> ChatResponseChunk {
    ChatResponseChunk {
        id: chunk.id,
        object: chunk.object,
        created: chunk.created,
        model: chunk.model,
        choices: chunk
            .choices
            .into_iter()
            .map(|c| ChunkChoice {
                index: c.index,
                delta: Delta {
                    role: match c.delta.role.as_ref() {
                        Some(server_types::Role::System) => "system".to_string(),
                        Some(server_types::Role::User) => "user".to_string(),
                        Some(server_types::Role::Assistant) => "assistant".to_string(),
                        Some(server_types::Role::Tool) => "tool".to_string(),
                        None => "assistant".to_string(),
                    },
                    content: c.delta.content.unwrap_or_default(),
                },
                finish_reason: c.finish_reason,
            })
            .collect(),
    }
}
