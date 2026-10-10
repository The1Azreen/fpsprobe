//! fpsprobe: GPU-accelerated frame-rate and HDR analysis for captured gameplay video.
//!
//! Pipeline: FFmpeg (hardware decode) -> per-tile frame differences on the GPU ->
//! noise-adaptive duplicate detection; optionally per-pixel nits and HDR scopes;
//! rendered as an overlay on the video and/or as a transparent graphs-only clip.

pub mod analyze;
pub mod canvas;
pub mod export;
mod gpu;
pub mod layout;
pub mod nits;
pub mod progress;
pub mod render;
pub mod video;
