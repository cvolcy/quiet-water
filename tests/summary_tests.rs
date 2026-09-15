use quiet_water::summary::{
    load_transcription_instructions_from_path, timestamped_summary_path_in, write_summary,
};
use std::{fs, path::PathBuf, time::{SystemTime, UNIX_EPOCH}};

fn unique_test_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nanos}-{}", std::process::id()))
}

#[test]
fn loads_instructions_before_raw_transcript_marker() {
    let path = unique_test_dir("quiet-water-instructions");
    let content = "Write a summary.\n\n## Raw Transcript:\nThis is ignored.";

    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, content).unwrap();

    let instructions = load_transcription_instructions_from_path(&path).unwrap();
    assert_eq!(instructions, "Write a summary.");
}

#[test]
fn write_summary_to_dir_creates_file_and_parent_directory() {
    let output_dir = unique_test_dir("quiet-water-summary");
    let summary = "# Executive summary\n\n- Done";

    let output_path = write_summary(summary, Some(&output_dir)).unwrap();
    println!("Summary written to: {}", output_path.display());
    assert!(output_path.starts_with(&output_dir));
    assert!(output_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .starts_with("summary-"));
    assert!(output_path.is_file());
    assert_eq!(fs::read_to_string(&output_path).unwrap(), summary);
}

#[test]
fn generated_summary_name_starts_with_summary_prefix() {
    let output_dir = unique_test_dir("quiet-water-name");

    let output_path = timestamped_summary_path_in(&output_dir);

    assert!(output_path.starts_with(&output_dir));
    assert!(output_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .starts_with("summary-"));
    assert_eq!(output_path.extension().and_then(|ext| ext.to_str()), Some("md"));
}

#[test]
fn write_summary_to_dir_overwrites_the_same_summary_file() {
    let output_dir = unique_test_dir("quiet-water-refresh");

    let first = write_summary("# first", Some(&output_dir)).unwrap();
    let second = write_summary("# second", Some(&output_dir)).unwrap();

    assert_eq!(first, second);
    assert_eq!(fs::read_to_string(&second).unwrap(), "# second");
}
