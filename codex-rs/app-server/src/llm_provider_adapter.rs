//! Env-driven in-process LLM provider registration for the app-server runtime.
//!
//! This keeps Codex on the normal Responses API path while short-circuiting the
//! matching transport calls to Kimi/Azure providers in-process, without relying
//! on the legacy local proxy.

use std::any::Any;
use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;

use codex_client::{
    ChatRequest, ChatResponse, ChatResponseChunk, ChunkChoice, Delta, LlmProvider,
    LlmProviderFactory, ModelInfo, ProviderError,
};
use futures::{stream, stream::BoxStream};
use reqwest::Client;
use tracing::{debug, error, info};

pub struct AppServerProviderFactory {
    providers: Vec<Arc<dyn LlmProvider>>,
}

impl AppServerProviderFactory {
    pub fn from_env() -> anyhow::Result<Self> {
        let mut providers: Vec<Arc<dyn LlmProvider>> = Vec::new();

        if let Ok(api_key) = std::env::var("KIMI_API_KEY")
            && !api_key.trim().is_empty()
        {
            providers.push(Arc::new(KimiProvider::new(api_key)?));
        }

        if let (Ok(endpoint), Ok(api_key)) = (
            std::env::var("AZURE_OPENAI_ENDPOINT"),
            std::env::var("AZURE_OPENAI_API_KEY"),
        ) && !endpoint.trim().is_empty() && !api_key.trim().is_empty()
        {
            providers.push(Arc::new(AzureProvider::new(
                endpoint,
                api_key,
                std::env::var("AZURE_OPENAI_DEPLOYMENT_NAME").ok(),
            )?));
        }

        if providers.is_empty() {
            anyhow::bail!("No in-process LLM provider configured");
        }

        Ok(Self { providers })
    }

    fn default_provider(&self) -> Option<Arc<dyn LlmProvider>> {
        self.providers
            .iter()
            .find(|provider| provider.name() == "kimi")
            .cloned()
            .or_else(|| self.providers.first().cloned())
    }
}

impl LlmProviderFactory for AppServerProviderFactory {
    fn route_request(&self, model: &str) -> Option<Arc<dyn LlmProvider>> {
        self.providers
            .iter()
            .find(|provider| {
                provider.name() == model
                    || provider.supported_models().contains(&model)
                    || (provider.name() == "kimi" && model.starts_with("kimi"))
                    || (provider.name() == "azure-openai" && model.starts_with("gpt-"))
            })
            .cloned()
            .or_else(|| self.default_provider())
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        let mut seen = HashSet::new();
        let mut models = Vec::new();

        for provider in &self.providers {
            for model_id in provider.supported_models() {
                if seen.insert((*model_id).to_string()) {
                    models.push(ModelInfo {
                        id: (*model_id).to_string(),
                        object: "model".to_string(),
                        created: 0,
                        owned_by: provider.name().to_string(),
                    });
                }
            }
        }

        models
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct KimiProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl KimiProvider {
    fn new(api_key: String) -> anyhow::Result<Self> {
        Ok(Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()?,
            api_key,
            base_url: "https://api.kimi.com/coding/v1".to_string(),
        })
    }
}

impl LlmProvider for KimiProvider {
    fn name(&self) -> &'static str {
        "kimi"
    }

    fn supported_models(&self) -> &[&'static str] {
        &["kimi-k2.5", "kimi-for-coding", "kimi-k2", "kimi-k2-thinking"]
    }

    fn chat(
        &self,
        request: ChatRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ChatResponse, ProviderError>> + Send + '_>> {
        Box::pin(async move {
            info!(model = %request.model, "Calling in-process Kimi provider");

            let response = self
                .client
                .post(format!("{}/chat/completions", self.base_url))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await
                .map_err(|err| ProviderError {
                    message: format!("Kimi API request failed: {err}"),
                })?;

            let status = response.status();
            if !status.is_success() {
                let error_text = response.text().await.unwrap_or_default();
                error!(status = %status, error = %error_text, "Kimi API error");
                return Err(ProviderError {
                    message: format!("Kimi API error {status}: {error_text}"),
                });
            }

            response.json::<ChatResponse>().await.map_err(|err| {
                error!(error = %err, "Failed to parse Kimi response");
                ProviderError {
                    message: format!("Failed to parse Kimi response: {err}"),
                }
            })
        })
    }

    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        BoxStream<'static, Result<ChatResponseChunk, ProviderError>>,
                        ProviderError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            let response = self.chat(request).await?;
            let chunks = response
                .choices
                .iter()
                .map(|choice| {
                    Ok(ChatResponseChunk {
                        id: response.id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: response.created,
                        model: response.model.clone(),
                        choices: vec![ChunkChoice {
                            index: choice.index,
                            delta: Delta {
                                role: choice.message.role.clone(),
                                content: choice.message.content.clone(),
                            },
                            finish_reason: choice.finish_reason.clone(),
                        }],
                    })
                })
                .collect::<Vec<_>>();
            Ok(Box::pin(stream::iter(chunks)) as BoxStream<'static, _>)
        })
    }
}

struct AzureProvider {
    client: Client,
    api_key: String,
    endpoint: String,
    deployment: String,
    api_version: String,
}

impl AzureProvider {
    fn new(
        endpoint: String,
        api_key: String,
        deployment: Option<String>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()?,
            api_key,
            endpoint,
            deployment: deployment.unwrap_or_else(|| "gpt-4o".to_string()),
            api_version: std::env::var("AZURE_OPENAI_API_VERSION")
                .unwrap_or_else(|_| "2024-02-01".to_string()),
        })
    }

    fn build_url(&self, path: &str) -> String {
        format!(
            "{}/openai/deployments/{}/{}?api-version={}",
            self.endpoint, self.deployment, path, self.api_version
        )
    }
}

impl LlmProvider for AzureProvider {
    fn name(&self) -> &'static str {
        "azure-openai"
    }

    fn supported_models(&self) -> &[&'static str] {
        &["gpt-4o", "gpt-4", "gpt-35-turbo"]
    }

    fn chat(
        &self,
        request: ChatRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ChatResponse, ProviderError>> + Send + '_>> {
        Box::pin(async move {
            info!(deployment = %self.deployment, "Calling in-process Azure provider");

            let response = self
                .client
                .post(self.build_url("chat/completions"))
                .header("api-key", &self.api_key)
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await
                .map_err(|err| ProviderError {
                    message: format!("Azure API request failed: {err}"),
                })?;

            let status = response.status();
            if !status.is_success() {
                let error_text = response.text().await.unwrap_or_default();
                error!(status = %status, error = %error_text, "Azure API error");
                return Err(ProviderError {
                    message: format!("Azure API error {status}: {error_text}"),
                });
            }

            response.json::<ChatResponse>().await.map_err(|err| {
                error!(error = %err, "Failed to parse Azure response");
                ProviderError {
                    message: format!("Failed to parse Azure response: {err}"),
                }
            })
        })
    }

    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        BoxStream<'static, Result<ChatResponseChunk, ProviderError>>,
                        ProviderError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            let response = self.chat(request).await?;
            let chunks = response
                .choices
                .iter()
                .map(|choice| {
                    Ok(ChatResponseChunk {
                        id: response.id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: response.created,
                        model: response.model.clone(),
                        choices: vec![ChunkChoice {
                            index: choice.index,
                            delta: Delta {
                                role: choice.message.role.clone(),
                                content: choice.message.content.clone(),
                            },
                            finish_reason: choice.finish_reason.clone(),
                        }],
                    })
                })
                .collect::<Vec<_>>();
            Ok(Box::pin(stream::iter(chunks)) as BoxStream<'static, _>)
        })
    }
}

/// Initialize the in-process LLM provider factory.
///
/// This should be called during app-server startup to enable in-process LLM routing.
pub fn init_in_process_llm() {
    match AppServerProviderFactory::from_env() {
        Ok(factory) => {
            let _ = codex_client::set_llm_provider_factory(Arc::new(factory));
            debug!("Registered in-process LLM provider factory");
        }
        Err(err) => {
            debug!(error = %err, "No in-process LLM provider available; falling back to configured HTTP provider path");
        }
    }
}
