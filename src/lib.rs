pub mod audio;
pub mod cli;
pub mod model;
pub mod summary;

use anyhow::Result;
use clap::Parser;
use cli::CliArgs;

pub async fn run() -> Result<()> {
    whisper_rs::install_logging_hooks();

    let args = CliArgs::parse();
    let whisper_model_path = args.whisper_model_path.as_path();
    let audio_path = args.input_audio.as_path();
    let output_dir = args.output_dir.as_path();

    model::ensure_model_exists(whisper_model_path)?;

    let samples = audio::read_wav_samples(audio_path)?;
    let chunks = audio::chunk_samples(&samples, 16_000, 30, 5);

    let mut cumulative_transcript = String::new();
    let total_chunks = chunks.len();

    println!("--- Transcribing {} chunks ---", total_chunks);
    for (index, chunk) in chunks.iter().enumerate() {
        let chunk_transcript = audio::transcribe_samples(chunk, whisper_model_path)?;
        let trimmed = chunk_transcript.trim();

        if !trimmed.is_empty() {
            cumulative_transcript.push_str(trimmed);
            cumulative_transcript.push(' ');
        }

        let summary = summary::summarize_transcript(&cumulative_transcript, Some(&args.model)).await?;
        let output_path = summary::write_summary(&summary, Some(output_dir))?;

        println!("\n--- Summary after chunk {} of {} ---\n{summary}\n", index + 1, total_chunks);
        println!("Summary saved to {}", output_path.display());
    }

    println!("\n--- Done ---");
    Ok(())
}
