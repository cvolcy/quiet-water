# Quiet Water

A Rust-based transcription and executive summary pipeline that turns raw audio into polished, business-ready notes using local AI models.

This project combines speech recognition, local model management, and LLM-based post-processing to transform noisy transcripts into clean, executive summaries that are easier to review, share, and act on.

## Project highlights

- Rust-first implementation with strong separation of concerns
- Local Whisper transcription using `whisper-rs`
- Automatic model provisioning for `ggml-base.bin`
- Local summarization using Ollama + Rig with `gemma4:12b`
- Output saved as timestamped markdown summaries for traceability
- Clear modular architecture for extensibility and maintenance

## Tech stack

- Rust
- `whisper-rs` for speech-to-text
- `hound` for WAV decoding
- `reqwest` for model download
- `rig` for LLM orchestration
- Ollama as the local inference layer

## Example workflow

```bash
cargo run
```

This produces a transcript summary and saves a file similar to:

```text
outputs/summary-20260830-123754.md
```

## Architecture

The application is structured around separation of concerns:

- `src/main.rs` orchestrates the end-to-end flow
- `src/audio.rs` handles transcription and audio preprocessing
- `src/model.rs` manages model download and validation
- `src/summary.rs` handles LLM summarization and output writing

This keeps the codebase modular and easier to evolve as the project grows.

## Use cases

- Meeting notes and recap generation
- Interview transcription cleanup
- Voice memo summarization
- Internal documentation support for executive communication

## Getting started

### Prerequisites

- Rust toolchain
- Ollama installed and running
- Local model pulled:

```bash
ollama pull gemma4:12b
```

### Run the app

```bash
cargo run
```
