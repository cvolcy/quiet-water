# Quiet Water

A Rust-based transcription and executive summary pipeline that turns raw audio into polished, business-ready notes using local AI models.

This project combines speech recognition, local model management, and LLM-powered post-processing to turn long recordings into cleaner, more actionable summaries while keeping everything local to the machine running it.

## Project highlights

- Rust-first implementation with strong separation of concerns
- WAV validation for 16 kHz mono input before transcription
- Chunked audio processing with overlap for long recordings
- Cumulative transcript building across chunks before summarization
- Summary refreshed after each chunk to the same timestamped output file
- Local Whisper transcription using `whisper-rs`
- Automatic model provisioning for `ggml-base.bin`
- Local summarization using Ollama + Rig with `gemma4:12b`
- Clear modular architecture for extensibility and maintenance

## Tech stack

- Rust
- `whisper-rs` for speech-to-text
- `hound` for WAV decoding and validation
- `reqwest` for model download
- `rig` for LLM orchestration
- Ollama as the local inference layer

## Example workflow

```bash
cargo run
```

This reads the audio file, processes it in chunks, updates the running transcript, and writes the latest summary to a timestamped file such as:

```text
outputs/summary-yyyymmdd-hhmmss.md
```

For a single run, the summary file remains stable across chunk updates, so the project continues refining the same document rather than creating a new summary file for each chunk.

## Architecture

The application is structured around a clear separation of concerns:

- `src/main.rs` provides the binary entry point
- `src/lib.rs` orchestrates the chunked transcription and summary refresh flow
- `src/audio.rs` handles audio validation, chunking, and transcription
- `src/model.rs` manages model download and validation
- `src/summary.rs` loads the prompt instructions and writes the summary output

This keeps the pipeline modular and easier to evolve as the project grows.

## Use cases

- Meeting notes and recap generation
- Interview transcription cleanup
- Voice memo summarization
- Internal documentation support for executive communication
- Processing longer recordings without sending everything through a single giant transcription pass

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
