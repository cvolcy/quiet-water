use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use rig::{
    agent::Agent, client::{AgentClientExt, Nothing}, completion::{Prompt, PromptError}, providers::ollama,
};
use std::{collections::HashMap, fs, path::{Path, PathBuf}, sync::{Mutex, OnceLock}};

use crate::cli::CliArgs;

const SUMMARY_OUTPUT_DIR: &str = "./outputs";
const TRANSCRIPT_PROMPT_PATH: &str = "./src/transcriptor.md";

pub struct SummaryService {
    model_name: String,
    prompt_path: PathBuf,
    output_dir: PathBuf,
    agent: Option<Agent>,
}

impl SummaryService {
    pub fn new() -> Self {
        let default_args = vec!["", "--input-audio", "./default"];
        let args = CliArgs::parse_from(default_args);

        Self {
            model_name: args.model,
            prompt_path: PathBuf::from("./src/transcriptor.md"),
            output_dir: PathBuf::from("./outputs"),
            agent: None,
        }
    }

    pub fn with_model(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = model_name.into();
        self
    }

    pub fn with_prompt_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.prompt_path = path.into();
        self
    }

    pub fn with_output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = dir.into();
        self
    }

    fn load_transcription_instructions(path: &Path) -> Result<String> {
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

    pub async fn summarize_transcript(&mut self, raw_transcript: &str) -> Result<String, PromptError> {
        let agent = self.get_agent();
        Ok(agent.prompt(raw_transcript).await?)
    }

    pub fn get_agent(&mut self) -> &Agent {
        if self.agent.is_none() {
            let instructions = SummaryService::load_transcription_instructions(Path::new(TRANSCRIPT_PROMPT_PATH))
                .unwrap_or_else(|e| panic!("Failed to load transcription instructions: {}", e));
                
            let client = ollama::Client::new(Nothing)
                .unwrap_or_else(|e| panic!("Failed to create Ollama client: {}", e));
            
            let agent = client.agent(self.model_name.clone())
                .preamble(&instructions)
                .build();

            self.agent = Some(agent);
        }

        self.agent.as_ref().unwrap()
    }

    pub fn write_summary(&self, summary: &str, output_path: Option<&Path>) -> Result<PathBuf> {
        let output_path = SummaryService::timestamped_summary_path(output_path.unwrap_or(Path::new(SUMMARY_OUTPUT_DIR)));
        SummaryService::write_summary_to_path(summary, &output_path)
    }

    fn write_summary_to_path(summary: &str, output_path: &Path) -> Result<PathBuf> {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).context("Failed to create output directory")?;
        }
        fs::write(output_path, summary).context("Failed to write summary file")?;
        Ok(output_path.to_path_buf())
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
}

impl Default for SummaryService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_test_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}-{}", std::process::id()))
    }

    #[test]
    fn test_default_initialization() {
        // Arrange

        // Act
        let service = SummaryService::new();

        // Assert
        assert_eq!(service.model_name, "gemma4:e4b");
        assert_eq!(service.prompt_path, PathBuf::from(TRANSCRIPT_PROMPT_PATH));
        assert_eq!(service.output_dir, PathBuf::from(SUMMARY_OUTPUT_DIR));
        assert!(service.agent.is_none(), "Agent should not be initialized by default");
    }

    #[test]
    fn test_default_trait_implementation() {
        // Arrange

        // Act
        let service = SummaryService::default();

        // Assert
        assert_eq!(service.model_name, "gemma4:e4b");
        assert_eq!(service.prompt_path, PathBuf::from(TRANSCRIPT_PROMPT_PATH));
        assert_eq!(service.output_dir, PathBuf::from(SUMMARY_OUTPUT_DIR));
        assert!(service.agent.is_none(), "Agent should not be initialized by default");
    }

    #[test]
    fn test_builder_with_model() {
        // Arrange
        let service = SummaryService::new();
        let expected_model = "llama3.1:latest";

        // Act
        let updated_service = service.with_model(expected_model);

        // Assert
        assert_eq!(updated_service.model_name, expected_model);
        assert_eq!(updated_service.prompt_path, PathBuf::from("./src/transcriptor.md"));
        assert_eq!(updated_service.output_dir, PathBuf::from("./outputs"));
    }

    #[test]
    fn test_builder_with_prompt_path() {
        // Arrange
        let service = SummaryService::new();
        let custom_path = PathBuf::from("/custom/path/prompt.md");

        // Act
        let updated_service = service.with_prompt_path(custom_path.clone());

        // Assert
        assert_eq!(updated_service.prompt_path, custom_path);
    }

    #[test]
    fn test_builder_with_output_dir() {
        // Arrange
        let service = SummaryService::new();
        let custom_dir = PathBuf::from("/tmp/custom_outputs");

        // Act
        let updated_service = service.with_output_dir(custom_dir.clone());

        // Assert
        assert_eq!(updated_service.output_dir, custom_dir);
    }

    #[test]
    fn test_builder_chaining() {
        // Arrange
        let initial_service = SummaryService::new();
        let target_model = "mistral:7b";
        let target_prompt = "./prompts/custom.md";
        let target_output = "./custom_out";

        // Act
        let configured_service = initial_service
            .with_model(target_model)
            .with_prompt_path(target_prompt)
            .with_output_dir(target_output);

        // Assert
        assert_eq!(configured_service.model_name, target_model);
        assert_eq!(configured_service.prompt_path, PathBuf::from(target_prompt));
        assert_eq!(configured_service.output_dir, PathBuf::from(target_output));
    }

    #[test]
    fn loads_instructions_before_raw_transcript_marker() {
        let path = unique_test_dir("quiet-water-instructions");
        let content = "Write a summary.\n\n## Raw Transcript:\nThis is ignored.";

        fs::write(&path, content).unwrap();

        let instructions = SummaryService::load_transcription_instructions(&path).unwrap();
        assert_eq!(instructions, "Write a summary.");
    }

    #[test]
    fn write_summary_to_dir_creates_file_and_parent_directory() {
        let output_dir = unique_test_dir("quiet-water-summary");
        let summary = "# Executive summary\n\n- Done";
        let service = SummaryService {
            model_name: String::new(),
            prompt_path: PathBuf::new(),
            output_dir: PathBuf::new(),
            agent: None,
        };

        let output_path = service.write_summary(summary, Some(&output_dir)).unwrap();

        assert!(output_path.starts_with(&output_dir));
        assert!(output_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("summary-"));
        assert!(output_path.is_file());
        assert_eq!(fs::read_to_string(&output_path).unwrap(), summary);
    }

    #[test]
    fn generated_summary_name_starts_with_summary_prefix() {
        let output_dir = unique_test_dir("quiet-water-name");

        let output_path = SummaryService::timestamped_summary_path(&output_dir);

        assert!(output_path.starts_with(&output_dir));
        assert!(output_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("summary-"));
        assert_eq!(output_path.extension().and_then(|ext| ext.to_str()), Some("md"));
    }

    #[test]
    fn write_summary_to_dir_overwrites_the_same_summary_file() {
        let output_dir = unique_test_dir("quiet-water-refresh");
        let service = SummaryService {
            model_name: String::new(),
            prompt_path: PathBuf::new(),
            output_dir: PathBuf::new(),
            agent: None,
        };

        let first = service.write_summary("# first", Some(&output_dir)).unwrap();
        let second = service.write_summary("# second", Some(&output_dir)).unwrap();

        assert_eq!(first, second);
        assert_eq!(fs::read_to_string(&second).unwrap(), "# second");
    }
}