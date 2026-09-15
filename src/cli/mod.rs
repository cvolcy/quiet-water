use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "quiet-water")]
pub struct CliArgs {
    #[arg(long, required = true, value_name = "INPUT_AUDIO_PATH")]
    pub input_audio: PathBuf,

    #[arg(long, default_value = "./outputs", value_name = "OUTPUT_DIR")]
    pub output_dir: PathBuf,

    #[arg(long, default_value = "./models/ggml-base.bin", value_name = "WHISPER_MODEL_PATH")]
    pub whisper_model_path: PathBuf,

    #[arg(long, default_value = "gemma4:e4b", value_name = "MODEL")]
    pub model: String,
}

#[cfg(test)]
mod tests {
    use super::CliArgs;
    use clap::{Parser, error::ErrorKind};
    use std::path::PathBuf;

    #[test]
    fn parses_default_cli_values() {
        let args = CliArgs::try_parse_from([
            "quiet-water",
            "--input-audio",
            "./files/audio.wav",
        ])
        .unwrap();

        assert_eq!(args.input_audio, PathBuf::from("./files/audio.wav"));
        assert_eq!(args.output_dir, PathBuf::from("./outputs"));
        assert_eq!(args.whisper_model_path, PathBuf::from("./models/ggml-base.bin"));
        assert_eq!(args.model, "gemma4:e4b");
    }

    #[test]
    fn parses_custom_cli_values() {
        let args = CliArgs::try_parse_from([
            "quiet-water",
            "--input-audio",
            "./custom/audio.wav",
            "--output-dir",
            "./custom/outputs",
            "--whisper-model-path",
            "./custom/model.bin",
            "--model",
            "llama3.2:latest",
        ])
        .unwrap();

        assert_eq!(args.input_audio, PathBuf::from("./custom/audio.wav"));
        assert_eq!(args.output_dir, PathBuf::from("./custom/outputs"));
        assert_eq!(args.whisper_model_path, PathBuf::from("./custom/model.bin"));
        assert_eq!(args.model, "llama3.2:latest");
    }

    #[test]
    fn invalid_arguments_produce_readable_error() {
        let err = CliArgs::try_parse_from(["quiet-water", "--output-dir", "./outputs"]).unwrap_err();

        assert_eq!(err.kind(), ErrorKind::MissingRequiredArgument);
        assert!(err.to_string().contains("--input-audio"));
    }
}
