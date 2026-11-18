//! The llama.cpp side of llama-gate.
//!
//! - [`Server`] talks to a `llama-server` control plane: health, shape
//!   (`/props`, `/slots`), tokenization (`/apply-template`, `/tokenize`), and
//!   slot persistence (`/slots/{id}?action=save|restore|erase`). It discovers
//!   which of these the server supports instead of assuming a build.
//! - [`Dialect`] is one request format the server accepts (OpenAI chat,
//!   completions and responses, native `/completion`, `/infill`, Anthropic
//!   messages, embeddings, reranking). A dialect knows where the prompt, the
//!   generation limit, and the server's slot options live in a body, nothing
//!   more. Add a dialect to support a new endpoint.
//! - [`Measurer`] turns a request body into a token count and a content
//!   address, exactly when the server can tokenize and conservatively when
//!   it cannot. [`TokenCache`] makes that cheap for long conversations by
//!   tokenizing only what is new since their previous turn.
//! - [`usage`] reads the final token counts from a finished response.
#![forbid(unsafe_code)]

mod dialect;
mod measure;
mod server;
pub mod tokens;
pub mod usage;

pub use dialect::{
    AnthropicMessages, Dialect, Dialects, Embeddings, Infill, NativeCompletion, OpenAiChat,
    OpenAiCompletions, OpenAiResponses, Prompt, Rerank,
};
pub use measure::{Measurement, Measurer, MeasurerConfig};
pub use server::{
    Capabilities, Health, Piece, Server, ServerError, ServerInfo, SlotReport, SlotView, Timeouts,
};
pub use tokens::{Boundaries, Method, TokenCache};
