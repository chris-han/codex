use super::super::types::*;
use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use std::pin::Pin;
use tracing::{debug, error, info};

/// Kimi LLM Provider
pub struct KimiProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl KimiProvider {
    /// Create new Kimi provider
    pub fn new(api_key: String) -> anyhow::Result<Self> {
        Ok(Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()?,
            api_key,
            base_url: "https://api.kimi.com/coding/v1".to_string(),
        })
    }

    /// Create with custom base URL (for testing)
    #[allow(dead_code)]
    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url,
        }
    }

    /// Translate OpenAI request to Kimi format
    fn translate_request(&self, request: &ChatRequest) -> KimiRequest {
        KimiRequest {
            model: request.model.clone(),
            messages: request.messages.clone(),
            tools: request.tools.clone(),
            tool_choice: request.tool_choice.clone(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: request.stream.unwrap_or(false),
            top_p: request.top_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
        }
    }

    /// Translate Kimi response to OpenAI format
    fn translate_response(&self, response: KimiResponse) -> ChatResponse {
        ChatResponse {
            id: response.id,
            object: "chat.completion".to_string(),
            created: response.created,
            model: response.model,
            choices: response
                .choices
                .into_iter()
                .map(|c| Choice {
                    index: c.index,
                    message: c.message,
                    finish_reason: c.finish_reason,
                    logprobs: None,
                })
                .collect(),
            usage: response.usage,
        }
    }

    /// Translate streaming chunk
    #[allow(dead_code)]
    fn translate_chunk(&self, chunk: KimiStreamChunk) -> ChatResponseChunk {
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
                    delta: c.delta,
                    finish_reason: c.finish_reason,
                })
                .collect(),
        }
    }
}

#[async_trait]
impl LLMProvider for KimiProvider {
    fn name(&self) -> &'static str {
        "kimi"
    }

    fn supported_models(&self) -> &[&'static str] {
        &[
            "kimi-k2.5",
            "kimi-for-coding",
            "kimi-k2",
            "kimi-k2-thinking",
        ]
    }

    async fn chat(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
        let kimi_request = self.translate_request(&request);

        info!(
            model = %request.model,
            messages = request.messages.len(),
            "Calling Kimi API"
        );

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&kimi_request)
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "Kimi API request failed");
                anyhow::anyhow!("Kimi API request failed: {}", e)
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            error!(status = %status, error = %error_text, "Kimi API error");
            return Err(anyhow::anyhow!("Kimi API error {}: {}", status, error_text));
        }

        let kimi_response: KimiResponse = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse Kimi response");
            anyhow::anyhow!("Failed to parse Kimi response: {}", e)
        })?;

        debug!(response_id = %kimi_response.id, "Kimi API response received");

        Ok(self.translate_response(kimi_response))
    }


    async fn chat_stream(
        &self,
        _request: ChatRequest,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = anyhow::Result<ChatResponseChunk>> + Send>>> {
        // TODO: Implement streaming
        todo!("Streaming not yet implemented")
    }
}

// Kimi-specific request/response types

#[derive(Debug, Clone, serde::Serialize)]
struct KimiRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct KimiResponse {
    id: String,
    created: i64,
    model: String,
    choices: Vec<KimiChoice>,
    usage: Usage,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct KimiChoice {
    index: u32,
    message: Message,
    finish_reason: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
struct KimiStreamChunk {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<KimiChunkChoice>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
struct KimiChunkChoice {
    index: u32,
    delta: Delta,
    finish_reason: Option<String>,
}
