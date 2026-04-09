mod custom_ca;
mod default_client;
mod error;
mod in_process_transport;
mod llm_config;
mod request;
mod retry;
mod sse;
mod telemetry;
mod transport;

pub use crate::custom_ca::BuildCustomCaTransportError;
/// Test-only subprocess hook for custom CA coverage.
///
/// This stays public only so the `custom_ca_probe` binary target can reuse the shared helper. It
/// is hidden from normal docs because ordinary callers should use
/// [`build_reqwest_client_with_custom_ca`] instead.
#[doc(hidden)]
pub use crate::custom_ca::build_reqwest_client_for_subprocess_tests;
pub use crate::custom_ca::build_reqwest_client_with_custom_ca;
pub use crate::custom_ca::maybe_build_rustls_client_config_with_custom_ca;
pub use crate::default_client::CodexHttpClient;
pub use crate::default_client::CodexRequestBuilder;
pub use crate::error::StreamError;
pub use crate::error::TransportError;
pub use crate::request::Request;
pub use crate::request::RequestBody;
pub use crate::request::RequestCompression;
pub use crate::request::Response;
pub use crate::retry::RetryOn;
pub use crate::retry::RetryPolicy;
pub use crate::retry::backoff;
pub use crate::retry::run_with_retry;
pub use crate::sse::sse_stream;
pub use crate::telemetry::RequestTelemetry;
pub use crate::in_process_transport::InProcessTransport;
pub use crate::llm_config::ChatRequest;
pub use crate::llm_config::ChatResponse;
pub use crate::llm_config::ChatResponseChunk;
pub use crate::llm_config::Choice;
pub use crate::llm_config::ChunkChoice;
pub use crate::llm_config::Delta;
pub use crate::llm_config::Function;
pub use crate::llm_config::FunctionCall;
pub use crate::llm_config::LlmProvider;
pub use crate::llm_config::LlmProviderFactory;
pub use crate::llm_config::Message;
pub use crate::llm_config::ModelInfo;
pub use crate::llm_config::ProviderError;
pub use crate::llm_config::Tool;
pub use crate::llm_config::ToolCall;
pub use crate::llm_config::Usage;
pub use crate::llm_config::get_llm_provider_factory;
pub use crate::llm_config::set_llm_provider_factory;
pub use crate::transport::ByteStream;
pub use crate::transport::HttpTransport;
pub use crate::transport::ReqwestTransport;
pub use crate::transport::StreamResponse;
