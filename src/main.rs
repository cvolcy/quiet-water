mod audio;
mod model;
mod summary;

use anyhow::Result;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    whisper_rs::install_logging_hooks();

    let model_path = Path::new("./models/ggml-base.bin");
    let audio_path = Path::new("./files/audio.wav");

    model::ensure_model_exists(model_path)?;

    let transcript = audio::transcribe_audio(audio_path, model_path)?;
    println!("\n--- Done ---");

    let summary = summary::summarize_transcript(&transcript).await?;
    let output_path = summary::write_summary(&summary)?;

    println!("\n--- Summary ---\n{summary}\n");
    println!("Summary saved to {}", output_path.display());
    Ok(())
}