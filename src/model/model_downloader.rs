use anyhow::{Context, Result};

pub trait ModelDownloader {
    fn fetch(&self, url: &str) -> impl std::future::Future<Output = Result<Vec<u8>>> + Send;
}

pub struct ReqwestModelDownloader;

impl ModelDownloader for ReqwestModelDownloader {
    async fn fetch(&self, url: &str) -> Result<Vec<u8>> {
        let response = reqwest::get(url)
            .await
            .context("Failed to request model download")?;
        if !response.status().is_success() {
            anyhow::bail!("Failed to download model: HTTP {}", response.status());
        }

        let bytes = response.bytes().await.context("Failed to read model bytes")?;
        Ok(bytes.to_vec())
    }
}
