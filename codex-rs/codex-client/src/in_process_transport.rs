use async_trait::async_trait;
use bytes::Bytes;
use codex_protocol::models::{ContentItem, ResponseItem};
use futures::stream;
use http::HeaderValue;
use serde_json::Value;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::TransportError;
use crate::llm_config::{
    get_llm_provider_factory, ChatRequest, ChatResponse, FunctionCall as LlmFunctionCall,
    LlmProvider, Message, Tool, ToolCall,
};
use crate::request::{Request, RequestBody, Response};
use crate::transport::{HttpTransport, StreamResponse};

/// In-process LLM provider transport that routes requests directly to providers
/// without making HTTP calls.
///
/// This transport uses the provider factory registered via
/// [`set_llm_provider_factory`](crate::llm_config::set_llm_provider_factory)
/// to route LLM requests to the appropriate providers (Kimi, Azure, etc.)
/// in-process.
#[derive(Clone, Debug)]
pub struct InProcessTransport;

impl InProcessTransport {
    /// Create a new in-process transport.
    ///
    /// Returns `None` if no provider factory has been registered.
    pub fn new() -> Option<Self> {
        get_llm_provider_factory().map(|_| Self)
    }

    /// Check if this transport can handle the given URL.
    pub fn can_handle(url: &str) -> bool {
        url.contains("/chat/completions") || url.contains("/models") || url.contains("/responses")
    }

    #[cfg(test)]
    fn current_timestamp() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
    }

    fn synthetic_id(prefix: &str) -> String {
        format!(
            "{prefix}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        )
    }

    fn provider_for_model(model: &str) -> Result<Arc<dyn LlmProvider>, TransportError> {
        let factory = get_llm_provider_factory().ok_or_else(|| {
            TransportError::Build("No LLM provider factory registered".to_string())
        })?;

        factory.route_request(model).ok_or_else(|| {
            TransportError::Build(format!("No provider available for model: {model}"))
        })
    }

    fn request_json_body(req: &Request, expected: &str) -> Result<Value, TransportError> {
        match &req.body {
            Some(RequestBody::Json(json)) => Ok(json.clone()),
            _ => Err(TransportError::Build(format!(
                "Expected JSON body for {expected}"
            ))),
        }
    }

    fn build_json_response(response_body: Value) -> Result<Response, TransportError> {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        let body_bytes = serde_json::to_vec(&response_body)
            .map_err(|err| TransportError::Build(err.to_string()))?;

        Ok(Response {
            status: http::StatusCode::OK,
            headers,
            body: Bytes::from(body_bytes),
        })
    }

    fn build_models_response() -> Result<Response, TransportError> {
        let factory = get_llm_provider_factory().ok_or_else(|| {
            TransportError::Build("No LLM provider factory registered".to_string())
        })?;

        let response_body = serde_json::json!({
            "object": "list",
            "data": factory
                .list_models()
                .into_iter()
                .map(|model| serde_json::json!({
                    "id": model.id,
                    "object": model.object,
                    "created": model.created,
                    "owned_by": model.owned_by,
                }))
                .collect::<Vec<_>>(),
        });

        Self::build_json_response(response_body)
    }

    fn build_chat_request_from_responses(body_json: Value) -> Result<ChatRequest, TransportError> {
        let model = body_json
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| TransportError::Build("Responses request is missing `model`".to_string()))?;

        let mut messages = Vec::new();

        if let Some(instructions) = body_json.get("instructions").and_then(Value::as_str) {
            let instructions = instructions.trim();
            if !instructions.is_empty() {
                messages.push(Message {
                    role: "system".to_string(),
                    content: instructions.to_string(),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
        }

        if let Some(input) = body_json.get("input") {
            messages.extend(Self::messages_from_responses_input(input)?);
        }

        let tools = match body_json.get("tools") {
            Some(Value::Null) | None => None,
            Some(value) => Some(
                serde_json::from_value::<Vec<Tool>>(value.clone())
                    .map_err(|err| TransportError::Build(err.to_string()))?,
            ),
        };

        let tool_choice = body_json.get("tool_choice").map(|value| match value {
            Value::String(raw) => raw.clone(),
            other => other.to_string(),
        });

        let max_tokens = body_json
            .get("max_output_tokens")
            .or_else(|| body_json.get("max_tokens"))
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok());

        Ok(ChatRequest {
            model,
            messages,
            tools,
            tool_choice,
            temperature: body_json
                .get("temperature")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            max_tokens,
            stream: body_json.get("stream").and_then(Value::as_bool),
            top_p: body_json
                .get("top_p")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            frequency_penalty: body_json
                .get("frequency_penalty")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            presence_penalty: body_json
                .get("presence_penalty")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
        })
    }

    fn messages_from_responses_input(input: &Value) -> Result<Vec<Message>, TransportError> {
        let mut messages = Vec::new();

        match input {
            Value::Null => {}
            Value::String(text) => {
                let text = text.trim();
                if !text.is_empty() {
                    messages.push(Message {
                        role: "user".to_string(),
                        content: text.to_string(),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
            }
            Value::Array(items) => {
                for item in items {
                    Self::append_messages_from_response_item(&mut messages, item)?;
                }
            }
            Value::Object(_) => {
                Self::append_messages_from_response_item(&mut messages, input)?;
            }
            _ => {
                return Err(TransportError::Build(
                    "Unsupported `input` payload in Responses request".to_string(),
                ));
            }
        }

        Ok(messages)
    }

    fn append_messages_from_response_item(
        messages: &mut Vec<Message>,
        item: &Value,
    ) -> Result<(), TransportError> {
        if let Ok(parsed) = serde_json::from_value::<ResponseItem>(item.clone()) {
            match parsed {
                ResponseItem::Message { role, content, .. } => {
                    let text = Self::content_to_text(&content);
                    if !text.is_empty() {
                        messages.push(Message {
                            role,
                            content: text,
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    }
                }
                ResponseItem::FunctionCall {
                    id,
                    name,
                    arguments,
                    call_id,
                    ..
                } => {
                    messages.push(Self::assistant_tool_call_message(
                        id.unwrap_or(call_id.clone()),
                        name,
                        arguments,
                    ));
                }
                ResponseItem::CustomToolCall {
                    id,
                    call_id,
                    name,
                    input,
                    ..
                } => {
                    messages.push(Self::assistant_tool_call_message(
                        id.unwrap_or(call_id.clone()),
                        name,
                        input,
                    ));
                }
                ResponseItem::LocalShellCall {
                    id,
                    call_id,
                    action,
                    ..
                } => {
                    messages.push(Self::assistant_tool_call_message(
                        call_id.or(id).unwrap_or_else(|| Self::synthetic_id("local-shell")),
                        "local_shell".to_string(),
                        serde_json::to_string(&action).unwrap_or_else(|_| "{}".to_string()),
                    ));
                }
                ResponseItem::ToolSearchCall {
                    id,
                    call_id,
                    execution,
                    arguments,
                    ..
                } => {
                    messages.push(Self::assistant_tool_call_message(
                        call_id.or(id).unwrap_or_else(|| Self::synthetic_id("tool-search")),
                        execution,
                        arguments.to_string(),
                    ));
                }
                ResponseItem::FunctionCallOutput { call_id, output } => {
                    messages.push(Message {
                        role: "tool".to_string(),
                        content: output
                            .text_content()
                            .map(str::to_owned)
                            .unwrap_or_else(|| {
                                serde_json::to_string(&output).unwrap_or_else(|_| String::new())
                            }),
                        tool_calls: None,
                        tool_call_id: Some(call_id),
                    });
                }
                ResponseItem::CustomToolCallOutput {
                    call_id,
                    output,
                    ..
                } => {
                    messages.push(Message {
                        role: "tool".to_string(),
                        content: output
                            .text_content()
                            .map(str::to_owned)
                            .unwrap_or_else(|| {
                                serde_json::to_string(&output).unwrap_or_else(|_| String::new())
                            }),
                        tool_calls: None,
                        tool_call_id: Some(call_id),
                    });
                }
                ResponseItem::ToolSearchOutput {
                    call_id,
                    status,
                    execution,
                    tools,
                } => {
                    messages.push(Message {
                        role: "tool".to_string(),
                        content: serde_json::json!({
                            "status": status,
                            "execution": execution,
                            "tools": tools,
                        })
                        .to_string(),
                        tool_calls: None,
                        tool_call_id: call_id,
                    });
                }
                ResponseItem::WebSearchCall { .. }
                | ResponseItem::ImageGenerationCall { .. }
                | ResponseItem::Reasoning { .. }
                | ResponseItem::GhostSnapshot { .. }
                | ResponseItem::Compaction { .. }
                | ResponseItem::Other => {}
            }
            return Ok(());
        }

        if let Some(text) = item.get("text").and_then(Value::as_str) {
            messages.push(Message {
                role: item
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("user")
                    .to_string(),
                content: text.to_string(),
                tool_calls: None,
                tool_call_id: None,
            });
            return Ok(());
        }

        Err(TransportError::Build(
            "Unsupported response item in in-process Responses request".to_string(),
        ))
    }

    fn assistant_tool_call_message(call_id: String, name: String, arguments: String) -> Message {
        Message {
            role: "assistant".to_string(),
            content: String::new(),
            tool_calls: Some(vec![ToolCall {
                id: call_id,
                tool_type: "function".to_string(),
                function: LlmFunctionCall { name, arguments },
            }]),
            tool_call_id: None,
        }
    }

    fn content_to_text(content: &[ContentItem]) -> String {
        content
            .iter()
            .filter_map(|item| match item {
                ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                    Some(text.as_str())
                }
                ContentItem::InputImage { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn response_items_from_chat_response(chat_response: &ChatResponse) -> Vec<ResponseItem> {
        let Some(choice) = chat_response.choices.first() else {
            return Vec::new();
        };

        let mut items = Vec::new();
        if !choice.message.content.trim().is_empty() {
            items.push(ResponseItem::Message {
                id: Some(format!("msg_{}", chat_response.id)),
                role: choice.message.role.clone(),
                content: vec![ContentItem::OutputText {
                    text: choice.message.content.clone(),
                }],
                end_turn: None,
                phase: None,
            });
        }

        if let Some(tool_calls) = &choice.message.tool_calls {
            for tool_call in tool_calls {
                items.push(ResponseItem::FunctionCall {
                    id: Some(tool_call.id.clone()),
                    name: tool_call.function.name.clone(),
                    namespace: None,
                    arguments: tool_call.function.arguments.clone(),
                    call_id: tool_call.id.clone(),
                });
            }
        }

        items
    }

    fn usage_json(chat_response: &ChatResponse) -> Value {
        serde_json::json!({
            "input_tokens": chat_response.usage.prompt_tokens,
            "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens": chat_response.usage.completion_tokens,
            "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": chat_response.usage.total_tokens,
        })
    }

    fn build_responses_body(chat_response: &ChatResponse) -> Result<Value, TransportError> {
        let output = Self::response_items_from_chat_response(chat_response)
            .into_iter()
            .map(|item| serde_json::to_value(item).map_err(|err| TransportError::Build(err.to_string())))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(serde_json::json!({
            "id": chat_response.id,
            "object": "response",
            "created_at": chat_response.created,
            "model": chat_response.model,
            "status": "completed",
            "output": output,
            "usage": Self::usage_json(chat_response),
        }))
    }

    fn sse_event(name: &str, payload: Value) -> Result<Bytes, TransportError> {
        let data = serde_json::to_string(&payload)
            .map_err(|err| TransportError::Build(err.to_string()))?;
        Ok(Bytes::from(format!("event: {name}\ndata: {data}\n\n")))
    }

    fn build_responses_stream(chat_response: &ChatResponse) -> Result<StreamResponse, TransportError> {
        let mut events = Vec::new();
        events.push(Ok(Self::sse_event(
            "response.created",
            serde_json::json!({
                "type": "response.created",
                "response": {
                    "id": chat_response.id,
                    "model": chat_response.model,
                    "status": "in_progress",
                }
            }),
        )?));

        for item in Self::response_items_from_chat_response(chat_response) {
            let item_json = serde_json::to_value(&item)
                .map_err(|err| TransportError::Build(err.to_string()))?;
            events.push(Ok(Self::sse_event(
                "response.output_item.added",
                serde_json::json!({
                    "type": "response.output_item.added",
                    "item": item_json,
                }),
            )?));

            if let ResponseItem::Message { content, .. } = &item {
                let delta = Self::content_to_text(content);
                if !delta.is_empty() {
                    events.push(Ok(Self::sse_event(
                        "response.output_text.delta",
                        serde_json::json!({
                            "type": "response.output_text.delta",
                            "delta": delta,
                        }),
                    )?));
                }
            }

            events.push(Ok(Self::sse_event(
                "response.output_item.done",
                serde_json::json!({
                    "type": "response.output_item.done",
                    "item": item_json,
                }),
            )?));
        }

        events.push(Ok(Self::sse_event(
            "response.completed",
            serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": chat_response.id,
                    "model": chat_response.model,
                    "usage": Self::usage_json(chat_response),
                }
            }),
        )?));

        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        headers.insert(
            http::header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache"),
        );

        Ok(StreamResponse {
            status: http::StatusCode::OK,
            headers,
            bytes: Box::pin(stream::iter(events)),
        })
    }

    fn build_chat_completions_stream(
        chat_response: &ChatResponse,
    ) -> Result<StreamResponse, TransportError> {
        let mut events = Vec::new();
        let Some(choice) = chat_response.choices.first() else {
            events.push(Ok(Bytes::from("data: [DONE]\n\n")));
            return Ok(StreamResponse {
                status: http::StatusCode::OK,
                headers: http::HeaderMap::new(),
                bytes: Box::pin(stream::iter(events)),
            });
        };

        let delta = serde_json::json!({
            "role": choice.message.role,
            "content": choice.message.content,
            "tool_calls": choice.message.tool_calls,
        });

        let chunk = serde_json::json!({
            "id": chat_response.id,
            "object": "chat.completion.chunk",
            "created": chat_response.created,
            "model": chat_response.model,
            "choices": [{
                "index": choice.index,
                "delta": delta,
                "finish_reason": choice.finish_reason,
            }],
        });

        events.push(Ok(Bytes::from(format!(
            "data: {}\n\n",
            serde_json::to_string(&chunk).map_err(|err| TransportError::Build(err.to_string()))?
        ))));
        events.push(Ok(Bytes::from("data: [DONE]\n\n")));

        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );

        Ok(StreamResponse {
            status: http::StatusCode::OK,
            headers,
            bytes: Box::pin(stream::iter(events)),
        })
    }
}

#[async_trait]
impl HttpTransport for InProcessTransport {
    async fn execute(&self, req: Request) -> Result<Response, TransportError> {
        if !Self::can_handle(&req.url) {
            return Err(TransportError::Build(format!(
                "InProcessTransport cannot handle URL: {}",
                req.url
            )));
        }

        if req.url.contains("/models") {
            return Self::build_models_response();
        }

        if req.url.contains("/chat/completions") {
            let body_json = Self::request_json_body(&req, "chat completions")?;
            let chat_request: ChatRequest = serde_json::from_value(body_json)
                .map_err(|err| TransportError::Build(err.to_string()))?;
            let provider = Self::provider_for_model(&chat_request.model)?;
            let chat_response = provider
                .chat(chat_request)
                .await
                .map_err(|err| TransportError::Network(err.message))?;
            return Self::build_json_response(
                serde_json::to_value(chat_response)
                    .map_err(|err| TransportError::Build(err.to_string()))?,
            );
        }

        if req.url.contains("/responses") {
            let body_json = Self::request_json_body(&req, "responses")?;
            let chat_request = Self::build_chat_request_from_responses(body_json)?;
            let provider = Self::provider_for_model(&chat_request.model)?;
            let chat_response = provider
                .chat(chat_request)
                .await
                .map_err(|err| TransportError::Network(err.message))?;
            return Self::build_json_response(Self::build_responses_body(&chat_response)?);
        }

        Err(TransportError::Build(format!(
            "Unknown endpoint: {}",
            req.url
        )))
    }

    async fn stream(&self, req: Request) -> Result<StreamResponse, TransportError> {
        if !Self::can_handle(&req.url) {
            return Err(TransportError::Build(format!(
                "InProcessTransport cannot handle URL: {}",
                req.url
            )));
        }

        if req.url.contains("/chat/completions") {
            let body_json = Self::request_json_body(&req, "chat completions")?;
            let mut chat_request: ChatRequest = serde_json::from_value(body_json)
                .map_err(|err| TransportError::Build(err.to_string()))?;
            let provider = Self::provider_for_model(&chat_request.model)?;
            chat_request.stream = Some(false);
            let chat_response = provider
                .chat(chat_request)
                .await
                .map_err(|err| TransportError::Network(err.message))?;
            return Self::build_chat_completions_stream(&chat_response);
        }

        if req.url.contains("/responses") {
            let body_json = Self::request_json_body(&req, "responses")?;
            let mut chat_request = Self::build_chat_request_from_responses(body_json)?;
            let provider = Self::provider_for_model(&chat_request.model)?;
            chat_request.stream = Some(false);
            let chat_response = provider
                .chat(chat_request)
                .await
                .map_err(|err| TransportError::Network(err.message))?;
            return Self::build_responses_stream(&chat_response);
        }

        Err(TransportError::Build(
            "Only chat completions and responses support streaming".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_config::{
        set_llm_provider_factory, Choice, LlmProviderFactory, ModelInfo, ProviderError, Usage,
    };
    use crate::request::RequestCompression;
    use crate::transport::ReqwestTransport;
    use futures::{stream::BoxStream, StreamExt};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Once;

    struct MockFactory;

    impl LlmProviderFactory for MockFactory {
        fn route_request(&self, model: &str) -> Option<Arc<dyn LlmProvider>> {
            if model == "kimi-for-coding" {
                Some(Arc::new(MockProvider))
            } else {
                None
            }
        }

        fn list_models(&self) -> Vec<ModelInfo> {
            vec![ModelInfo {
                id: "kimi-for-coding".to_string(),
                object: "model".to_string(),
                created: 0,
                owned_by: "mock".to_string(),
            }]
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    struct MockProvider;

    impl LlmProvider for MockProvider {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn supported_models(&self) -> &[&'static str] {
            &["kimi-for-coding"]
        }

        fn chat(
            &self,
            request: ChatRequest,
        ) -> Pin<Box<dyn Future<Output = Result<ChatResponse, ProviderError>> + Send + '_>> {
            Box::pin(async move {
                let wants_tool = request
                    .messages
                    .iter()
                    .any(|message| message.content.contains("weather"));

                let message = if wants_tool {
                    Message {
                        role: "assistant".to_string(),
                        content: String::new(),
                        tool_calls: Some(vec![ToolCall {
                            id: "call_mock_weather".to_string(),
                            tool_type: "function".to_string(),
                            function: LlmFunctionCall {
                                name: "get_weather".to_string(),
                                arguments: r#"{"location":"Beijing"}"#.to_string(),
                            },
                        }]),
                        tool_call_id: None,
                    }
                } else {
                    Message {
                        role: "assistant".to_string(),
                        content: "mock response".to_string(),
                        tool_calls: None,
                        tool_call_id: None,
                    }
                };

                Ok(ChatResponse {
                    id: "resp_mock".to_string(),
                    object: "chat.completion".to_string(),
                    created: InProcessTransport::current_timestamp(),
                    model: request.model,
                    choices: vec![Choice {
                        index: 0,
                        message,
                        finish_reason: Some("stop".to_string()),
                        logprobs: None,
                    }],
                    usage: Usage {
                        prompt_tokens: 3,
                        completion_tokens: 2,
                        total_tokens: 5,
                    },
                })
            })
        }

        fn chat_stream(
            &self,
            _request: ChatRequest,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            BoxStream<'static, Result<crate::llm_config::ChatResponseChunk, ProviderError>>,
                            ProviderError,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            let empty: BoxStream<'static, Result<crate::llm_config::ChatResponseChunk, ProviderError>> =
                Box::pin(stream::empty());
            Box::pin(async move { Ok(empty) })
        }
    }

    fn install_test_factory() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = set_llm_provider_factory(Arc::new(MockFactory));
        });
    }

    fn request(url: &str, body: Value) -> Request {
        Request {
            method: http::Method::POST,
            url: url.to_string(),
            headers: http::HeaderMap::new(),
            body: Some(RequestBody::Json(body)),
            compression: RequestCompression::None,
            timeout: None,
        }
    }

    #[tokio::test]
    async fn responses_execute_returns_openai_responses_shape() {
        install_test_factory();
        let transport = InProcessTransport::new().expect("in-process transport should exist");

        let response = transport
            .execute(request(
                "https://api.openai.com/v1/responses",
                serde_json::json!({
                    "model": "kimi-for-coding",
                    "instructions": "Be concise",
                    "input": "Say hi",
                    "stream": false,
                }),
            ))
            .await
            .expect("responses execute should succeed");

        let body: Value = serde_json::from_slice(&response.body).expect("valid json body");
        assert_eq!(body.get("object").and_then(Value::as_str), Some("response"));
        assert_eq!(body.get("status").and_then(Value::as_str), Some("completed"));
        assert!(body.get("output").and_then(Value::as_array).is_some());
    }

    #[tokio::test]
    async fn responses_stream_emits_expected_sse_lifecycle() {
        install_test_factory();
        let transport = InProcessTransport::new().expect("in-process transport should exist");

        let mut response = transport
            .stream(request(
                "https://api.openai.com/v1/responses",
                serde_json::json!({
                    "model": "kimi-for-coding",
                    "input": "Say hi",
                    "stream": true,
                }),
            ))
            .await
            .expect("responses stream should succeed");

        let mut sse = String::new();
        while let Some(chunk) = response.bytes.next().await {
            let chunk = chunk.expect("stream chunk should be ok");
            sse.push_str(std::str::from_utf8(&chunk).expect("valid utf8"));
        }

        assert!(sse.contains("event: response.created"));
        assert!(sse.contains("event: response.output_item.added"));
        assert!(sse.contains("event: response.output_text.delta"));
        assert!(sse.contains("event: response.output_item.done"));
        assert!(sse.contains("event: response.completed"));
        assert!(sse.contains("mock response"));
    }

    #[tokio::test]
    async fn reqwest_transport_short_circuits_models_to_in_process_backend() {
        install_test_factory();
        let transport = ReqwestTransport::new(reqwest::Client::new());

        let response = transport
            .execute(Request {
                method: http::Method::GET,
                url: "https://api.openai.com/v1/models".to_string(),
                headers: http::HeaderMap::new(),
                body: None,
                compression: RequestCompression::None,
                timeout: None,
            })
            .await
            .expect("models request should be intercepted");

        let body: Value = serde_json::from_slice(&response.body).expect("valid json body");
        let models = body
            .get("data")
            .and_then(Value::as_array)
            .expect("models array present");
        assert!(models.iter().any(|model| {
            model.get("id").and_then(Value::as_str) == Some("kimi-for-coding")
        }));
    }
}
