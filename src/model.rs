use anyhow::{Context, Result};
use std::{fs, io, path::Path};

const MODEL_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin";

pub fn ensure_model_exists(model_path: &Path) -> Result<()> {
    if model_path.exists() {
        return Ok(());
    }

    if let Some(parent) = model_path.parent() {
        fs::create_dir_all(parent).context("Failed to create model directory")?;
    }

    println!("Downloading model to {}...", model_path.display());

    let response = reqwest::blocking::get(MODEL_URL).context("Failed to request model download")?;
    if !response.status().is_success() {
        anyhow::bail!("Failed to download model: HTTP {}", response.status());
    }

    let bytes = response.bytes().context("Failed to read model bytes")?;
    let mut file = fs::File::create(model_path).context("Failed to create model file")?;
    io::copy(&mut bytes.as_ref(), &mut file).context("Failed to write model file")?;

    println!("Model downloaded successfully.");
    Ok(())
}
