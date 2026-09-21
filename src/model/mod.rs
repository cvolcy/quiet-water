pub mod model_downloader;

use anyhow::{Context, Result};
use std::{fs, io, path::Path};

use model_downloader::{ ModelDownloader, ReqwestModelDownloader};

const MODEL_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

fn model_download_url(model_path: &Path) -> Result<String> {
    let filename = model_path
        .file_name()
        .and_then(|os_str| os_str.to_str())
        .context("Invalid or missing filename in model path")?;

    Ok(format!("{}/{}", MODEL_BASE_URL, filename))
}

pub async fn ensure_model_exists(model_path: &Path) -> Result<()> {
    let downloader = ReqwestModelDownloader;
    ensure_model_exists_with_downloader(model_path, &downloader).await
}

async fn ensure_model_exists_with_downloader(
    model_path: &Path,
    downloader: &impl ModelDownloader,
) -> Result<()> {
    if model_path.exists() {
        return Ok(());
    }

    if let Some(parent) = model_path.parent() {
        fs::create_dir_all(parent).context("Failed to create model directory")?;
    }

    let download_url = model_download_url(model_path)?;
    println!("Downloading model to {}...", model_path.display());

    let bytes = downloader.fetch(&download_url).await?;
    let mut file = fs::File::create(model_path).context("Failed to create model file")?;
    io::copy(&mut bytes.as_slice(), &mut file).context("Failed to write model file")?;

    println!("Model downloaded successfully.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn unique_test_path(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}-{}", std::process::id()))
    }

    struct FakeDownloader {
        payload: Vec<u8>,
    }

    impl ModelDownloader for FakeDownloader {
        async fn fetch(&self, _url: &str) -> Result<Vec<u8>> {
            Ok(self.payload.clone())
        }
    }

    #[tokio::test]
    async fn ensure_model_exists_skips_download_when_file_already_exists() {
        let path = unique_test_path("existing-model");
        fs::write(&path, b"already-present").unwrap();

        let downloader = FakeDownloader {
            payload: b"should-not-be-used".to_vec(),
        };

        ensure_model_exists_with_downloader(&path, &downloader)
            .await
            .unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"already-present");
    }

    #[tokio::test]
    async fn ensure_model_exists_downloads_bytes_when_missing() {
        let path = unique_test_path("missing-model");
        let downloader = FakeDownloader {
            payload: b"model-content".to_vec(),
        };

        ensure_model_exists_with_downloader(&path, &downloader)
            .await
            .unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"model-content");
        assert!(path.exists());
    }

    #[test]
    fn model_download_url_uses_filename_from_path() {
        let path = Path::new("/tmp/model/ggml-small.bin");
        assert_eq!(model_download_url(path).unwrap(), format!("{}/{}", MODEL_BASE_URL, "ggml-small.bin"));
    }

    #[test]
    fn model_download_url_requires_filename() {
        let path = Path::new("");
        assert!(model_download_url(path).is_err());
    }
}
