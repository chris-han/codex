use crate::providers::types::*;
use crate::state::AppState;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Sse},
    Json,
};
use futures::StreamExt;
use std::sync::Arc;

/// List available models
pub async fn list_models_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let factory = match &state.provider_factory {
        Some(factory) => factory,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "LLM provider not configured"
                })),
            )
                .into_response();
        }
    };

    let models = factory.list_models();

    Json(ModelsResponse {
        object: "list".to_string(),
        data: models,
    })
    .into_response()
}

/// Chat completions handler
pub async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> impl IntoResponse {
    let factory = match &state.provider_factory {
        Some(factory) => factory,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "LLM provider not configured. Set KIMI_API_KEY or Azure credentials."
                })),
            )
                .into_response();
        }
    };

    let provider = factory.get_provider(&request.model);

    // Check if streaming is requested
    if request.stream.unwrap_or(false) {
        // Streaming response
        match provider.chat_stream(request).await {
            Ok(stream) => {
                let sse_stream = stream.map(|result| {
                    match result {
                        Ok(chunk) => {
                            let data = serde_json::to_string(&chunk).unwrap_or_default();
                            Ok::<_, std::convert::Infallible>(
                                axum::response::sse::Event::default().data(data)
                            )
                        }
                        Err(e) => {
                            Ok(axum::response::sse::Event::default()
                                .data(format!("{{\"error\": \"{}\"}}", e)))
                        }
                    }
                });

                Sse::new(sse_stream)
                    .keep_alive(axum::response::sse::KeepAlive::new())
                    .into_response()
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response(),
        }
    } else {
        // Non-streaming response
        match provider.chat(request).await {
            Ok(response) => Json(response).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response(),
        }
    }
}

/// Models list response
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<Model>,
}
