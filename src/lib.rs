pub mod audio;
pub mod model;
pub mod summary;

use anyhow::Result;
use std::path::Path;

pub async fn run() -> Result<()> {
    whisper_rs::install_logging_hooks();

    let model_path = Path::new("./models/ggml-base.bin");
    let audio_path = Path::new("./files/audio.wav");

    model::ensure_model_exists(model_path)?;

    let samples = audio::read_wav_samples(audio_path)?;
    let chunks = audio::chunk_samples(&samples, 16_000, 30, 5);

    let mut cumulative_transcript = String::new();
    let total_chunks = chunks.len();

    println!("--- Transcribing {} chunks ---", total_chunks);
    for (index, chunk) in chunks.iter().enumerate() {
        let chunk_transcript = audio::transcribe_samples(chunk, model_path)?;
        let trimmed = chunk_transcript.trim();

        if !trimmed.is_empty() {
            cumulative_transcript.push_str(trimmed);
            cumulative_transcript.push(' ');
        }

        let summary = summary::summarize_transcript(&cumulative_transcript).await?;
        let output_path = summary::write_summary(&summary)?;

        println!("\n--- Summary after chunk {} of {} ---\n{summary}\n", index + 1, total_chunks);
        println!("Summary saved to {}", output_path.display());
    }

    println!("\n--- Done ---");
    Ok(())
}
