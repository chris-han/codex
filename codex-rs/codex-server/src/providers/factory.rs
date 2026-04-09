use super::types::*;
use super::kimi::KimiProvider;
use super::azure::AzureProvider;
use std::collections::HashMap;
use std::sync::Arc;

/// Factory for creating LLM providers
pub struct ProviderFactory {
    /// Provider by model name
    models: HashMap<String, Arc<dyn LLMProvider>>,
    /// Default provider
    default_provider: Arc<dyn LLMProvider>,
}

impl ProviderFactory {
    /// Create factory from environment variables
    pub fn from_env() -> anyhow::Result<Self> {
        let mut models: HashMap<String, Arc<dyn LLMProvider>> = HashMap::new();

        // Register Kimi provider if configured
        if let Ok(api_key) = std::env::var("KIMI_API_KEY") {
            let kimi = Arc::new(KimiProvider::new(api_key)?);

            // Register by provider name
            models.insert("kimi".to_string(), kimi.clone());

            // Register by model names
            for model in ["kimi-for-coding", "kimi-k2.5", "kimi-k2", "kimi-k2-thinking"] {
                models.insert(model.to_string(), kimi.clone());
            }
        }

        // Register Azure provider if configured
        if let (Ok(endpoint), Ok(api_key)) = (
            std::env::var("AZURE_OPENAI_ENDPOINT"),
            std::env::var("AZURE_OPENAI_API_KEY"),
        ) {
            let deployment = std::env::var("AZURE_OPENAI_DEPLOYMENT_NAME").ok();
            let azure = Arc::new(AzureProvider::new(endpoint, api_key, deployment)?);

            // Register by provider name
            models.insert("azure".to_string(), azure.clone());

            // Register by model names
            for model in ["gpt-4o", "gpt-4", "gpt-35-turbo"] {
                models.insert(model.to_string(), azure.clone());
            }
        }

        // Determine default provider (prefer Kimi)
        let default_provider = models.get("kimi-for-coding")
            .or(models.get("kimi"))
            .or(models.values().next())
            .ok_or_else(|| anyhow::anyhow!("No LLM provider configured. Set KIMI_API_KEY or Azure credentials."))?
            .clone();

        Ok(Self { models, default_provider })
    }

    /// Get provider for a specific model
    pub fn get_provider(&self, model: &str) -> Arc<dyn LLMProvider> {
        self.models.get(model)
            .unwrap_or(&self.default_provider)
            .clone()
    }

    /// Get provider by name
    #[allow(dead_code)]
    pub fn get_by_name(&self, name: &str) -> Option<Arc<dyn LLMProvider>> {
        self.models.get(name).cloned()
    }

    /// List all available models
    pub fn list_models(&self) -> Vec<Model> {
        let mut seen = std::collections::HashSet::new();
        let mut models = Vec::new();

        for (_name, provider) in &self.models {
            for model_id in provider.supported_models() {
                if seen.insert(model_id.to_string()) {
                    models.push(Model {
                        id: model_id.to_string(),
                        object: "model".to_string(),
                        created: 0,
                        owned_by: provider.name().to_string(),
                    });
                }
            }
        }

        models
    }

    /// Check if any provider is configured
    #[allow(dead_code)]
    pub fn is_configured(&self) -> bool {
        !self.models.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_requires_provider() {
        // Clear env vars
        std::env::remove_var("KIMI_API_KEY");
        std::env::remove_var("AZURE_OPENAI_ENDPOINT");

        let result = ProviderFactory::from_env();
        assert!(result.is_err());
    }
}
