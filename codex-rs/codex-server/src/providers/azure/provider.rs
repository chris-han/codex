use super::super::types::*;
use async_trait::async_trait;
use futures::Stream;
use futures::StreamExt;
use reqwest::Client;
use std::pin::Pin;
use tracing::{debug, error, info};

/// Azure OpenAI LLM Provider
pub struct AzureProvider {
    client: Client,
    api_key: String,
    endpoint: String,
    deployment: String,
    api_version: String,
}

impl AzureProvider {
    /// Create new Azure provider
    pub fn new(
        endpoint: String,
        api_key: String,
        deployment: Option<String>,
    ) -> anyhow::Result<Self> {
        let deployment = deployment.unwrap_or_else(|| "gpt-4o".to_string());
        let api_version = std::env::var("AZURE_OPENAI_API_VERSION")
            .unwrap_or_else(|_| "2024-02-01".to_string());

        Ok(Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()?,
            api_key,
            endpoint,
            deployment,
            api_version,
        })
    }

    /// Build Azure API URL
    fn build_url(&self, path: &str) -> String {
        format!(
            "{}/openai/deployments/{}/{}?api-version={}",
            self.endpoint,
            self.deployment,
            path,
            self.api_version
        )
    }
}

#[async_trait]
impl LLMProvider for AzureProvider {
    fn name(&self) -> &'static str {
        "azure-openai"
    }

    fn supported_models(&self) -> &[&'static str] {
        &["gpt-4o", "gpt-4", "gpt-35-turbo"]
    }

    async fn chat(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
        let azure_request = AzureRequest {
            messages: request.messages,
            tools: request.tools,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: request.stream.unwrap_or(false),
            top_p: request.top_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
        };

        info!(
            deployment = %self.deployment,
            messages = azure_request.messages.len(),
            "Calling Azure OpenAI API"
        );

        let response = self
            .client
            .post(self.build_url("chat/completions"))
            .header("api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&azure_request)
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "Azure API request failed");
                anyhow::anyhow!("Azure API request failed: {}", e)
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            error!(status = %status, error = %error_text, "Azure API error");
            return Err(anyhow::anyhow!(
                "Azure API error {}: {}",
                status,
                error_text
            ));
        }

        let azure_response: AzureResponse = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse Azure response");
            anyhow::anyhow!("Failed to parse Azure response: {}", e)
        })?;

        debug!(response_id = %azure_response.id, "Azure API response received");

        Ok(self.translate_response(azure_response))
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = anyhow::Result<ChatResponseChunk>> + Send>>> {
        let azure_request = AzureRequest {
            messages: request.messages,
            tools: request.tools,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: true,
            top_p: request.top_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
        };

        info!(
            deployment = %self.deployment,
            messages = azure_request.messages.len(),
            "Calling Azure OpenAI API (streaming)"
        );

        let response = self
            .client
            .post(self.build_url("chat/completions"))
            .header("api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&azure_request)
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "Azure streaming request failed");
                anyhow::anyhow!("Azure streaming request failed: {}", e)
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            error!(status = %status, error = %error_text, "Azure API streaming error");
            return Err(anyhow::anyhow!("Azure API error {}: {}", status, error_text));
        }

        let byte_stream = response.bytes_stream();

        let stream = byte_stream
            .map(|chunk_result| {
                chunk_result
                    .map_err(|e| anyhow::anyhow!("Stream read error: {}", e))
                    .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            })
            .flat_map(|line_result: anyhow::Result<String>| {
                let lines: Vec<anyhow::Result<ChatResponseChunk>> = match line_result {
                    Err(e) => vec![Err(e)],
                    Ok(text) => text
                        .split('\n')
                        .filter_map(|line| {
                            // Strip "data:" prefix (with or without trailing space)
                            let data = line.strip_prefix("data:")?.trim();
                            if data == "[DONE]" || data.is_empty() {
                                return None;
                            }
                            Some(
                                serde_json::from_str::<AzureStreamChunk>(data)
                                    .map_err(|e| anyhow::anyhow!("Parse SSE chunk error: {}", e))
                                    .map(|chunk| ChatResponseChunk {
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
                                    }),
                            )
                        })
                        .collect(),
                };
                futures::stream::iter(lines)
            });

        Ok(Box::pin(stream))
    }
}

impl AzureProvider {
    fn translate_response(&self, response: AzureResponse) -> ChatResponse {
        ChatResponse {
            id: response.id,
            object: response.object,
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

    #[allow(dead_code)]
    fn translate_chunk(&self, chunk: AzureStreamChunk) -> ChatResponseChunk {
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

// Azure-specific types
#[derive(Debug, Clone, serde::Serialize)]
struct AzureRequest {
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
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
struct AzureResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<AzureChoice>,
    usage: Usage,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct AzureChoice {
    index: u32,
    message: Message,
    finish_reason: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
struct AzureStreamChunk {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<AzureChunkChoice>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
struct AzureChunkChoice {
    index: u32,
    delta: Delta,
    finish_reason: Option<String>,
}
