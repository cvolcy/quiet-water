use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    quiet_water::run().await
}