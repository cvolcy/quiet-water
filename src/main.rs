use hound::{SampleFormat, WavReader};
use std::{fs, io, path::Path};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const MODEL_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    whisper_rs::install_logging_hooks();

    let model_path = Path::new("./models/ggml-base.bin");
    let audio_path = Path::new("./files/audio.wav");

    ensure_model_exists(model_path)?;

    let ctx = WhisperContext::new_with_params(
        model_path
            .to_str()
            .ok_or("Invalid UTF-8 in model path")?,
        WhisperContextParameters::default(),
    )?;
    let mut state = ctx.create_state()?;

    let samples = read_wav_samples(audio_path)?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(None);
    params.set_debug_mode(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_timestamps(false);
    params.set_print_realtime(false);

    params.set_segment_callback_safe_lossy(|segment: whisper_rs::SegmentCallbackData| {
        println!(
            "[{} - {}]: {}",
            to_duration_format(segment.start_timestamp),
            to_duration_format(segment.end_timestamp),
            segment.text
        );
    });

    println!("--- Transcribing (Streaming) ---");

    state.full(params, &samples[..])?;

    println!("\n--- Done ---");
    Ok(())
}

fn ensure_model_exists(model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if model_path.exists() {
        return Ok(());
    }

    if let Some(parent) = model_path.parent() {
        fs::create_dir_all(parent)?;
    }

    println!("Downloading model to {}...", model_path.display());

    let response = reqwest::blocking::get(MODEL_URL)?;
    if !response.status().is_success() {
        return Err(format!("Failed to download model: HTTP {}", response.status()).into());
    }

    let bytes = response.bytes()?;
    let mut file = fs::File::create(model_path)?;
    io::copy(&mut bytes.as_ref(), &mut file)?;

    println!("Model downloaded successfully.");
    Ok(())
}

fn read_wav_samples<P: AsRef<Path>>(path: P) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    if spec.sample_rate != 16000 || spec.channels != 1 {
        return Err("Audio file must be 16kHz mono WAV format.".into());
    }

    let samples: Result<Vec<f32>, _> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader.samples::<f32>().collect(),
        (SampleFormat::Int, 16) => {
            let max_val = i16::MAX as f32;
            reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect()
        }
        (SampleFormat::Int, 24) | (SampleFormat::Int, 32) => {
            let max_val = match spec.bits_per_sample {
                24 => 8_388_607.0, // 2^23 - 1
                _ => i32::MAX as f32,
            };
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect()
        }
        (format, bits) => {
            return Err(format!("Unsupported audio format: {:?}, {} bits", format, bits).into())
        }
    };

    Ok(samples?)
}

fn to_duration_format(centiseconds: i64) -> String {
    let total_secs = centiseconds / 100;
    let cs = centiseconds % 100;

    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    format!("{:02}:{:02}:{:02}.{:02}", hours, minutes, seconds, cs)
}