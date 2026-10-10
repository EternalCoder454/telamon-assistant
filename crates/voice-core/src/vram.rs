//! The graphics memory cap. A full card crashed the desktop before, so
//! while Telamon's models run (whisper on Vulkan, llama-server), a watcher
//! reads amdgpu's sysfs every 2 s; two readings in a row at or over the cap
//! stop everything. Telamon starts again only 5 points below the cap. No
//! readable sysfs: the cap is off.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// The caps on offer, in percent; `None` is off.
pub const CHOICES: [Option<u32>; 5] = [None, Some(85), Some(90), Some(95), Some(98)];
pub const DEFAULT_CAP: u32 = 95;
/// How far under the cap usage must fall before Telamon starts again.
pub const RESUME_MARGIN: u32 = 5;
const INTERVAL: Duration = Duration::from_secs(2);
/// Readings in a row at or over the cap that stop the models.
const STRIKES: u32 = 2;

/// A setting's value ("off", "90", "" …) as a cap; anything else is the default.
pub fn parse_cap(value: &str) -> Option<u32> {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" | "0" => None,
        "" => Some(DEFAULT_CAP),
        v => match v.trim_end_matches('%').parse::<u32>() {
            Ok(n) if CHOICES.contains(&Some(n)) => Some(n),
            _ => Some(DEFAULT_CAP),
        },
    }
}

/// The fullest card's graphics memory use, in percent (0..100); None when
/// no card reports it.
pub fn usage() -> Option<f64> {
    usage_in(Path::new("/sys/class/drm"))
}

fn usage_in(drm: &Path) -> Option<f64> {
    let read = |p: PathBuf| std::fs::read_to_string(p).ok()?.trim().parse::<u64>().ok();
    let mut worst: Option<f64> = None;
    for card in std::fs::read_dir(drm).ok()?.flatten() {
        let name = card.file_name().to_string_lossy().into_owned();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let dev = card.path().join("device");
        let (Some(used), Some(total)) = (
            read(dev.join("mem_info_vram_used")),
            read(dev.join("mem_info_vram_total")),
        ) else {
            continue;
        };
        if total > 0 {
            let pct = used as f64 * 100.0 / total as f64;
            worst = Some(worst.map_or(pct, |w: f64| w.max(pct)));
        }
    }
    worst
}

/// Why Telamon may not start now: usage is not yet 5 points under the cap.
pub fn blocked(cap: Option<u32>) -> Option<String> {
    let cap = cap?;
    let pct = usage()?;
    let resume = cap.saturating_sub(RESUME_MARGIN) as f64;
    (pct > resume).then(|| {
        format!("Graphics memory is {pct:.0} % full; Telamon starts again below {resume:.0} %.")
    })
}

/// Watches the card until `done` is set. On the second reading in a row at
/// or over `cap`, calls `trip` once with the reason and stops watching.
/// Returns at once (and never trips) when the cap is off or unreadable.
pub fn watch(
    cap: Option<u32>,
    done: Arc<AtomicBool>,
    trip: impl FnOnce(String) + Send + 'static,
) -> Option<std::thread::JoinHandle<()>> {
    watch_in(PathBuf::from("/sys/class/drm"), INTERVAL, cap?, done, trip)
}

fn watch_in(
    drm: PathBuf,
    interval: Duration,
    cap: u32,
    done: Arc<AtomicBool>,
    trip: impl FnOnce(String) + Send + 'static,
) -> Option<std::thread::JoinHandle<()>> {
    usage_in(&drm)?;
    Some(std::thread::spawn(move || {
        let mut strikes = 0;
        while !done.load(Ordering::Relaxed) {
            match usage_in(&drm) {
                Some(pct) if pct >= cap as f64 => {
                    strikes += 1;
                    // Not once the worker it guards has ended.
                    if strikes >= STRIKES && !done.load(Ordering::Relaxed) {
                        trip(format!(
                            "Graphics memory reached {pct:.0} % (the cap is {cap} %), so Telamon stopped its models."
                        ));
                        return;
                    }
                }
                _ => strikes = 0,
            }
            std::thread::sleep(interval);
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_parse() {
        assert_eq!(parse_cap(""), Some(95));
        assert_eq!(parse_cap("off"), None);
        assert_eq!(parse_cap("90"), Some(90));
        assert_eq!(parse_cap("98%"), Some(98));
        assert_eq!(parse_cap("50"), Some(95));
    }

    fn fake_card(dir: &Path, used: u64) {
        let dev = dir.join("card0/device");
        std::fs::create_dir_all(&dev).unwrap();
        std::fs::write(dev.join("mem_info_vram_used"), used.to_string()).unwrap();
        std::fs::write(dev.join("mem_info_vram_total"), "100").unwrap();
    }

    #[test]
    fn trips_on_the_second_reading_over_the_cap() {
        let dir = std::env::temp_dir().join(format!("telamon-vram-w-{}", std::process::id()));
        let fast = Duration::from_millis(20);
        let (tx, rx) = std::sync::mpsc::channel();
        // Under the cap: never trips.
        fake_card(&dir, 90);
        let done = Arc::new(AtomicBool::new(false));
        let t = tx.clone();
        let h = watch_in(dir.clone(), fast, 95, done.clone(), move |w| {
            t.send(w).unwrap()
        })
        .unwrap();
        std::thread::sleep(Duration::from_millis(150));
        done.store(true, Ordering::Relaxed);
        h.join().unwrap();
        assert!(rx.try_recv().is_err());
        // Over it: trips once, saying why.
        fake_card(&dir, 96);
        let h = watch_in(dir.clone(), fast, 95, Arc::default(), move |w| {
            tx.send(w).unwrap()
        })
        .unwrap();
        h.join().unwrap();
        assert!(rx.recv().unwrap().contains("96 %"));
        // No card: the cap is off.
        let none = watch_in(dir.join("missing"), fast, 95, Arc::default(), |_| {});
        assert!(none.is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reads_the_fullest_card() {
        let dir = std::env::temp_dir().join(format!("telamon-vram-{}", std::process::id()));
        for (card, used, total) in [("card0", 1, 4), ("card1", 9, 10)] {
            let dev = dir.join(card).join("device");
            std::fs::create_dir_all(&dev).unwrap();
            std::fs::write(dev.join("mem_info_vram_used"), used.to_string()).unwrap();
            std::fs::write(dev.join("mem_info_vram_total"), total.to_string()).unwrap();
        }
        std::fs::create_dir_all(dir.join("card1-DP-1")).unwrap();
        assert_eq!(usage_in(&dir), Some(90.0));
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(usage_in(Path::new("/nonexistent")), None);
    }
}
