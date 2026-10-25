//! End-to-end tests on small synthetic clips encoded with FFmpeg.
//! Skipped (with a note) when FFmpeg is not installed.

use fpsprobe::export::{self, ExportJob, RunSummary};
use fpsprobe::progress::Progress;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

const W: usize = 320;
const H: usize = 192;

fn ffmpeg_available() -> bool {
    let ok = Command::new("ffmpeg").arg("-version").stdout(Stdio::null()).status().is_ok_and(|s| s.success());
    if !ok {
        eprintln!("ffmpeg not found; skipping");
    }
    ok
}

/// Diagonal stripes that move every frame, so every row of tiles changes.
fn image(k: usize) -> Vec<u8> {
    (0..W * H).map(|i| (((i % W) + (i / W) + 5 * k) * 3 % 256) as u8).collect()
}

/// Encodes 60 fps gray frames to H.264 (lossy, like a real capture). `vf` filters
/// the stream, with variable frame rate output.
fn encode(name: &str, frames: impl Iterator<Item = Vec<u8>>, vf: Option<&str>) -> PathBuf {
    let path = std::env::temp_dir().join(format!("fpsprobe-test-{name}-{}.mkv", std::process::id()));
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", &format!("{W}x{H}")]);
    cmd.args(["-framerate", "60", "-i", "-"]);
    if let Some(vf) = vf {
        cmd.args(["-vf", vf, "-fps_mode", "vfr"]);
    }
    cmd.args(["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"]).arg(&path);
    let mut child = cmd.stdin(Stdio::piped()).spawn().expect("run ffmpeg");
    let mut stdin = child.stdin.take().unwrap();
    for f in frames {
        stdin.write_all(&f).unwrap();
    }
    drop(stdin);
    assert!(child.wait().unwrap().success(), "ffmpeg failed to encode the test clip");
    path
}

fn run(path: &PathBuf, configure: impl FnOnce(&mut ExportJob)) -> RunSummary {
    let mut job = ExportJob { use_gpu: false, hwaccel: None, ..ExportJob::new(path.clone()) };
    configure(&mut job);
    let report = export::run(&job, &Progress::default()).expect("export");
    let _ = std::fs::remove_file(path);
    report.stats
}

#[test]
fn thirty_fps_in_sixty() {
    if !ffmpeg_available() {
        return;
    }
    let clip = encode("thirty", (0..300).map(|i| image(i / 2)), None);
    let s = run(&clip, |_| {});
    assert!((s.fps.avg_fps - 30.0).abs() < 0.5, "{}", s.fps.avg_fps);
    assert!(!s.variable_frame_rate);
}

#[test]
fn capture_clock_repeats_are_found_and_counted() {
    if !ffmpeg_available() {
        return;
    }
    // A new image every frame, except one repeat every 150 frames (2.5 s).
    let mut k = 0;
    let frames: Vec<Vec<u8>> = (0..1200)
        .map(|i| {
            if i % 150 != 75 {
                k += 1;
            }
            image(k)
        })
        .collect();
    let clip = encode("capture", frames.clone().into_iter(), None);
    let s = run(&clip, |_| {});
    let drops = s.capture_drops.expect("capture drops found");
    assert_eq!(drops.frames, 8);
    assert!((drops.period_seconds - 2.5).abs() < 0.02, "{}", drops.period_seconds);
    assert!((s.fps.avg_fps - 60.0).abs() < 0.01, "{}", s.fps.avg_fps);

    let clip = encode("capture-kept", frames.into_iter(), None);
    let s = run(&clip, |j| j.fix_capture_drops = false);
    assert_eq!(s.fps.unique, 1192);
}

#[test]
fn variable_frame_rate_uses_timestamps() {
    if !ffmpeg_available() {
        return;
    }
    // Every 30th frame is dropped from the recording, leaving a 33 ms gap.
    let clip = encode("vfr", (0..600).map(image), Some("select='not(eq(mod(n\\,30)\\,29))'"));
    let s = run(&clip, |_| {});
    assert!(s.variable_frame_rate);
    assert_eq!(s.frames, 580);
    assert!((s.fps.avg_fps - 58.0).abs() < 0.2, "{}", s.fps.avg_fps);
}

#[test]
fn torn_frames_are_detected() {
    if !ffmpeg_available() {
        return;
    }
    // Every 20th frame shows the previous image above row 96 and the new one below.
    let frames = (0..400).map(|i| {
        let mut f = image(i);
        if i % 20 == 10 {
            f[..96 * W].copy_from_slice(&image(i - 1)[..96 * W]);
        }
        f
    });
    let clip = encode("tears", frames, None);
    let s = run(&clip, |j| j.tears = true);
    assert_eq!(s.torn_frames, Some(20));
}

#[test]
fn render_after_cancelled_render() {
    if !ffmpeg_available() {
        return;
    }
    let clip = encode("cancel", (0..1200).map(image), None);
    let dir = std::env::temp_dir().join(format!("fpsprobe-test-cancel-{}", std::process::id()));
    let (first, second) = (dir.join("first"), dir.join("second"));
    std::fs::create_dir_all(&first).unwrap();
    std::fs::create_dir_all(&second).unwrap();
    let job = |out: &PathBuf| ExportJob {
        use_gpu: false,
        hwaccel: None,
        video: Some(out.join("out.mp4")),
        ..ExportJob::new(clip.clone())
    };

    // Cancel once rendering is under way, as the GUI's Cancel button does.
    let progress = std::sync::Arc::new(Progress::default());
    let p = progress.clone();
    let first_job = job(&first);
    let handle = std::thread::spawn(move || export::run(&first_job, &p));
    while !(progress.snapshot().0 == "Rendering" && progress.snapshot().1 > 100) && !handle.is_finished() {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    progress.cancel();
    let err = handle.join().unwrap().err().expect("first export cancelled");
    assert!(format!("{err:#}").contains("cancelled"), "{err:#}");

    // Then export again somewhere else.
    let report = export::run(&job(&second), &Progress::default());
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&clip);
    report.expect("export after a cancelled one");
}
