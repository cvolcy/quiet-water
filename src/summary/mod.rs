use anyhow::{Context, Result};
use chrono::Utc;
use rig::{
    client::{AgentClientExt, Nothing},
    completion::Prompt,
    providers::ollama,
};
use std::{ collections::HashMap, fs, path::{Path, PathBuf}, sync::{Mutex, OnceLock }};

pub const SUMMARY_OUTPUT_DIR: &str = "./outputs";
const TRANSCRIPT_PROMPT_PATH: &str = "./src/transcriptor.md";

pub async fn summarize_transcript(raw_transcript: &str, model_name: Option<&str>) -> Result<String> {
    let model_name = model_name.unwrap_or("gemma4:e4b");
    let client = ollama::Client::new(Nothing)?;
    let instructions = load_transcription_instructions()?;

    let agent = client
        .agent(model_name)
        .preamble(&instructions)
        .build();

    Ok(agent.prompt(raw_transcript).await?)
}

pub fn write_summary(summary: &str, output_path: Option<&Path>) -> Result<PathBuf> {
    let output_path = timestamped_summary_path(output_path.unwrap_or(Path::new(SUMMARY_OUTPUT_DIR)));
    write_summary_to_path(summary, &output_path)
}

fn timestamped_summary_path(output_dir: &Path) -> PathBuf {
    static SUMMARY_PATH_CACHE: OnceLock<Mutex<HashMap<String, PathBuf>>> = OnceLock::new();

    let cache = SUMMARY_PATH_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = output_dir.to_string_lossy().to_string();
    let mut cache = cache.lock().unwrap();

    cache
        .entry(key)
        .or_insert_with(|| {
            let timestamp = Utc::now().format("%Y%m%d-%H%M%S");
            output_dir.join(format!("summary-{timestamp}.md"))
        })
        .clone()
}

fn write_summary_to_path(summary: &str, output_path: &Path) -> Result<PathBuf> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).context("Failed to create output directory")?;
    }
    fs::write(output_path, summary).context("Failed to write summary file")?;
    Ok(output_path.to_path_buf())
}

pub fn timestamped_summary_path_in(output_dir: &Path) -> PathBuf {
    let timestamp = Utc::now().format("%Y%m%d-%H%M%S");
    output_dir.join(format!("summary-{timestamp}.md"))
}

fn load_transcription_instructions() -> Result<String> {
    load_transcription_instructions_from_path(Path::new(TRANSCRIPT_PROMPT_PATH))
}

pub fn load_transcription_instructions_from_path(path: &Path) -> Result<String> {
    let raw = fs::read_to_string(path).context("Missing transcription instruction file")?;
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
