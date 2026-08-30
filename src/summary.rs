use anyhow::{Context, Result};
use rig::{
    client::{AgentClientExt, Nothing},
    completion::Prompt,
    providers::ollama,
};
use std::{fs, path::Path};

pub const SUMMARY_OUTPUT_PATH: &str = "./outputs/summary.md";
const TRANSCRIPT_PROMPT_PATH: &str = "./src/transcriptor.md";

pub async fn summarize_transcript(raw_transcript: &str) -> Result<String> {
    let client = ollama::Client::new(Nothing)?;
    let instructions = load_transcription_instructions()?;

    let agent = client
        .agent("gemma4:12b")
        .preamble(&instructions)
        .build();

    Ok(agent.prompt(raw_transcript).await?)
}

pub fn write_summary(summary: &str) -> Result<()> {
    let output_path = Path::new(SUMMARY_OUTPUT_PATH);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).context("Failed to create output directory")?;
    }
    fs::write(output_path, summary).context("Failed to write summary file")?;
    Ok(())
}

fn load_transcription_instructions() -> Result<String> {
    let raw = fs::read_to_string(TRANSCRIPT_PROMPT_PATH).context("Missing transcription instruction file")?;
    let instructions = raw
        .split("## Raw Transcript:")
        .next()
        .unwrap_or(&raw)
        .trim();

    if instructions.is_empty() {
        anyhow::bail!("Transcript instruction file is empty or malformed.");
    }

    Ok(instructions.to_string())
}
