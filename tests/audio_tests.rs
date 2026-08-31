use quiet_water::audio::{chunk_samples, read_wav_samples, to_duration_format};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::{
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

fn unique_test_path(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nanos}-{}", std::process::id()))
}

fn write_pcm_wav(path: &Path, samples: &[i16], sample_rate: u32, channels: u16) {
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec).unwrap();
    for sample in samples {
        writer.write_sample(*sample).unwrap();
    }
}

#[test]
fn reads_int16_pcm_samples_from_16khz_mono_wav() {
    let path = unique_test_path("quiet-water-audio");
    let samples = vec![0_i16, 1_000, -1_000, 2_000];

    write_pcm_wav(&path, &samples, 16_000, 1);

    let decoded = read_wav_samples(&path).unwrap();
    assert_eq!(decoded.len(), samples.len());
    assert!((decoded[0] - 0.0).abs() < f32::EPSILON);
    assert!((decoded[1] - (1_000.0 / i16::MAX as f32)).abs() < 1e-6);
    assert!((decoded[2] + (1_000.0 / i16::MAX as f32)).abs() < 1e-6);
}

#[test]
fn rejects_wav_files_that_are_not_16khz_mono() {
    let path = unique_test_path("quiet-water-audio-invalid");
    let samples = vec![0_i16, 1_000];

    write_pcm_wav(&path, &samples, 8_000, 2);

    let err = read_wav_samples(&path).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("16kHz mono WAV format"));
}

#[test]
fn formats_duration_with_hours_minutes_seconds_and_centiseconds() {
    assert_eq!(to_duration_format(0), "00:00:00.00");
    assert_eq!(to_duration_format(12345), "00:02:03.45");
    assert_eq!(to_duration_format(366101), "01:01:01.01");
}

#[test]
fn chunks_samples_into_fixed_windows_with_overlap() {
    let samples: Vec<f32> = (0..(16000 * 5)).map(|value| value as f32).collect();
    let chunk_duration = 2;
    let chunk_rate = 16_000.0;
    let expected_chunk_len = chunk_duration * chunk_rate as usize;
    let chunks = chunk_samples(&samples, 16000, chunk_duration  as u32, 1);

    assert_eq!(chunks.len(), 4);
    assert_eq!(chunks[0].len(), expected_chunk_len);
    assert_eq!(chunks[1].len(), expected_chunk_len);
    assert_eq!(chunks[2].len(), expected_chunk_len);
    assert_eq!(chunks[3].len(), expected_chunk_len);
    assert_eq!(chunks[0][0], 0.0);
    assert_eq!(chunks[0][last_index(&chunks[0])], (expected_chunk_len - 1) as f32);
    assert_eq!(chunks[1][0], 16_000.0);
    assert_eq!(chunks[3][last_index(&chunks[3])],  (samples.len() - 1) as f32);
}

fn last_index<T>(slice: &[T]) -> usize {
    slice.len().saturating_sub(1)
}
