use anyhow::{Context, Result};
use hound::{SampleFormat, WavReader};
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
};

#[derive(Debug, Clone, PartialEq)]
pub struct AudioChunk {
    pub start_sample_offset: usize,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionSegment {
    pub start_timestamp: u64,
    pub end_timestamp: u64,
    pub text: String,
}

pub fn transcribe_audio(audio_path: &Path, model_path: &Path) -> Result<String> {
    let samples = read_wav_samples(audio_path)?;
    transcribe_samples(&samples, model_path)
}

pub fn transcribe_samples(samples: &[f32], model_path: &Path) -> Result<String> {
    let segments = transcribe_segments(samples, model_path, 0)?;
    Ok(segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join(" "))
}

pub fn transcribe_segments(
    samples: &[f32],
    model_path: &Path,
    chunk_offset_centiseconds: u64,
) -> Result<Vec<TranscriptionSegment>> {
    let ctx = WhisperContext::new_with_params(
        model_path
            .to_str()
            .context("Invalid UTF-8 in model path")?,
        WhisperContextParameters::default(),
    )?;
    let mut state = ctx.create_state()?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(None);
    params.set_debug_mode(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_timestamps(false);
    params.set_print_realtime(false);

    let segments = Arc::new(Mutex::new(Vec::new()));
    let callback_buffer = Arc::clone(&segments);

    params.set_segment_callback_safe_lossy(move |segment: whisper_rs::SegmentCallbackData| {
        let start_timestamp = segment.start_timestamp as u64 + chunk_offset_centiseconds;
        let end_timestamp = segment.end_timestamp as u64 + chunk_offset_centiseconds;
        let text = segment.text.trim().to_string();

        println!(
            "[{} - {}]: {}",
            to_duration_format(start_timestamp),
            to_duration_format(end_timestamp),
            text
        );

        if !text.is_empty() {
            let mut buffer = callback_buffer.lock().unwrap();
            buffer.push(TranscriptionSegment {
                start_timestamp,
                end_timestamp,
                text,
            });
        }
    });

    println!("--- Transcribing (Streaming) ---");
    state.full(params, samples)?;

    Ok(segments.lock().unwrap().clone())
}

pub fn offset_segments(
    segments: &[TranscriptionSegment],
    offset_centiseconds: u64,
) -> Vec<TranscriptionSegment> {
    segments
        .iter()
        .map(|segment| TranscriptionSegment {
            start_timestamp: segment.start_timestamp + offset_centiseconds,
            end_timestamp: segment.end_timestamp + offset_centiseconds,
            text: segment.text.clone(),
        })
        .collect()
}

pub fn chunk_samples(
    samples: &[f32],
    sample_rate: u32,
    chunk_duration_seconds: u32,
    overlap_seconds: u32,
) -> Vec<AudioChunk> {
    if samples.is_empty() {
        return Vec::new();
    }

    if sample_rate == 0 || chunk_duration_seconds == 0 {
        return vec![AudioChunk {
            start_sample_offset: 0,
            samples: samples.to_vec(),
        }];
    }

    let chunk_size = sample_rate as usize * chunk_duration_seconds as usize;
    let overlap = sample_rate as usize * overlap_seconds.min(chunk_duration_seconds) as usize;
    let step = chunk_size.saturating_sub(overlap).max(1);

    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < samples.len() {
        let end = (start + chunk_size).min(samples.len());
        chunks.push(AudioChunk {
            start_sample_offset: start,
            samples: samples[start..end].to_vec(),
        });

        if end == samples.len() {
            break;
        }

        start += step;
        if start >= samples.len() {
            let tail_start = samples.len().saturating_sub(chunk_size.min(samples.len()));
            if tail_start < samples.len() {
                chunks.push(AudioChunk {
                    start_sample_offset: tail_start,
                    samples: samples[tail_start..].to_vec(),
                });
            }
            break;
        }
    }

    if chunks.is_empty() {
        vec![AudioChunk {
            start_sample_offset: 0,
            samples: samples.to_vec(),
        }]
    } else {
        chunks
    }
}

pub fn read_wav_samples<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    if spec.sample_rate != 16000 || spec.channels != 1 {
        return Err(anyhow::anyhow!(
            "Audio file must be 16kHz mono WAV format."
        ));
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
                24 => 8_388_607.0,
                _ => i32::MAX as f32,
            };
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect()
        }
        (format, bits) => {
            return Err(anyhow::anyhow!(
                "Unsupported audio format: {:?}, {} bits",
                format,
                bits
            ))
        }
    };

    Ok(samples?)
}

pub fn to_duration_format(centiseconds: u64) -> String {
    let total_secs = centiseconds / 100;
    let cs = centiseconds % 100;

    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    format!("{:02}:{:02}:{:02}.{:02}", hours, minutes, seconds, cs)
}
