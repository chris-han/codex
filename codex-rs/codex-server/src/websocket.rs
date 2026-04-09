use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    response::IntoResponse,
};
use futures::stream::StreamExt;
use std::sync::Arc;
use tracing::{info, warn};

use crate::state::AppState;

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    info!("WebSocket client connected");

    // Get or create session to subscribe to its event stream
    let user_id = "default";
    let (event_tx, ready_msg) = match state.get_session(user_id).await {
        Ok(session) => {
            let ready = serde_json::json!({
                "method": "ready",
                "params": { "ok": true },
                "atIso": chrono::Utc::now().to_rfc3339()
            });
            (Some(session.event_tx), ready.to_string())
        }
        Err(e) => {
            warn!("Failed to get session for WebSocket: {}", e);
            let ready = serde_json::json!({
                "method": "ready",
                "params": { "ok": false, "error": format!("{}", e) },
                "atIso": chrono::Utc::now().to_rfc3339()
            });
            (None, ready.to_string())
        }
    };

    // Send ready notification
    if let Err(e) = socket.send(Message::Text(ready_msg.into())).await {
        warn!("Failed to send ready message: {}", e);
        return;
    }

    // If we have an event channel, stream events. Otherwise just echo.
    if let Some(broadcast_tx) = event_tx {
        let mut broadcast_rx = broadcast_tx.subscribe();

        loop {
            tokio::select! {
                // Forward events from codex to the WebSocket client
                result = broadcast_rx.recv() => {
                    match result {
                        Ok(json) => {
                            if let Err(e) = socket.send(Message::Text(json.to_string().into())).await {
                                warn!("Failed to send WebSocket event: {}", e);
                                break;
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            warn!("WebSocket subscriber lagged, dropped {} events", n);
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            info!("Event broadcast channel closed");
                            break;
                        }
                    }
                }
                // Handle incoming messages from the frontend
                msg = socket.next() => {
                    match msg {
                        Some(Ok(Message::Text(_text))) => {
                            // Frontend messages are handled via RPC, not WS
                        }
                        Some(Ok(Message::Close(_))) | None => {
                            info!("WebSocket client disconnected");
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }
    } else {
        // No event channel — just drain incoming messages until close
        while let Some(Ok(msg)) = socket.next().await {
            if let Message::Close(_) = msg {
                info!("WebSocket client disconnected");
                break;
            }
        }
    }
}
