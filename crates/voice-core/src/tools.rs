//! The model's tools. Every one only reads: the time, system stats,
//! processes, the network, the location, the weather, and file names under
//! the home folder. None writes, runs a program the model names, or reads a
//! file's contents. Results are JSON text for the model.

use anyhow::{Result, anyhow};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, UNIX_EPOCH};

/// The tools, in the OpenAI `tools` format llama-server takes.
pub fn definitions() -> Value {
    let tool = |name: &str, description: &str, parameters: Value| {
        json!({"type": "function", "function": {
            "name": name, "description": description, "parameters": parameters}})
    };
    let none = json!({"type": "object", "properties": {}});
    json!([
        tool("get_time", "The current local date, time, weekday and time zone.", none.clone()),
        tool(
            "get_calendar",
            "Facts about a date: its weekday, how many days away it is, and that month's calendar.",
            json!({"type": "object", "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD; today if left out"}}})
        ),
        tool(
            "get_system_stats",
            "CPU use and temperature, memory, disk space, GPU use, temperature and memory, and uptime.",
            none.clone()
        ),
        tool(
            "list_processes",
            "The programs using the most CPU or memory.",
            json!({"type": "object", "properties": {
                "sort": {"type": "string", "enum": ["cpu", "memory"]},
                "count": {"type": "integer", "minimum": 1, "maximum": 15}}})
        ),
        tool(
            "get_network",
            "Network interfaces, whether they are up, their addresses and traffic.",
            none.clone()
        ),
        tool("get_location", "The user's approximate location (city, time zone).", none.clone()),
        tool(
            "get_weather",
            "Current weather and today's forecast.",
            json!({"type": "object", "properties": {
                "location": {"type": "string", "description": "A city; the user's own if left out"}}})
        ),
        tool(
            "list_files",
            "File and folder names in a folder under the user's home folder.",
            json!({"type": "object", "properties": {
                "path": {"type": "string", "description": "Relative to the home folder, e.g. Downloads"},
                "hidden": {"type": "boolean"}}})
        ),
    ])
}

/// Where a tool reads from; the test points them at fixtures.
#[derive(Clone, Debug)]
pub struct Context {
    pub home: PathBuf,
    /// The configured place; empty: from the time zone.
    pub location: String,
}

/// Runs the tool `name` with the JSON `arguments`; errors come back as text
/// for the model, so it can say what went wrong.
pub fn call(ctx: &Context, name: &str, arguments: &str) -> String {
    let args: Value = serde_json::from_str(arguments).unwrap_or(json!({}));
    let result = match name {
        "get_time" => Ok(time()),
        "get_calendar" => calendar(args["date"].as_str()),
        "get_system_stats" => Ok(system_stats()),
        "list_processes" => Ok(processes(
            args["sort"].as_str() == Some("memory"),
            args["count"].as_u64().unwrap_or(5).clamp(1, 15) as usize,
        )),
        "get_network" => Ok(network()),
        "get_location" => Ok(location(ctx)),
        "get_weather" => weather(ctx, args["location"].as_str()),
        "list_files" => list_files(
            &ctx.home,
            args["path"].as_str().unwrap_or(""),
            args["hidden"].as_bool().unwrap_or(false),
        ),
        _ => Err(anyhow!("there is no tool called {name}")),
    };
    match result {
        Ok(v) => v.to_string(),
        Err(e) => json!({"error": e.to_string()}).to_string(),
    }
}

fn time() -> Value {
    let now = chrono::Local::now();
    json!({
        "date": now.format("%A, %B %-d, %Y").to_string(),
        "time": now.format("%-I:%M %p").to_string(),
        "time_24h": now.format("%H:%M").to_string(),
        "time_zone": time_zone().unwrap_or_else(|| now.format("%Z").to_string()),
        "utc_offset": now.format("%:z").to_string(),
    })
}

fn calendar(date: Option<&str>) -> Result<Value> {
    use chrono::{Datelike, NaiveDate};
    let today = chrono::Local::now().date_naive();
    let day = match date.filter(|d| !d.trim().is_empty()) {
        Some(d) => NaiveDate::parse_from_str(d.trim(), "%Y-%m-%d")
            .map_err(|_| anyhow!("the date must look like 2026-12-25"))?,
        None => today,
    };
    let first = day.with_day(1).unwrap_or(day);
    let mut weeks = Vec::new();
    let mut week = vec![String::new(); first.weekday().num_days_from_monday() as usize];
    let mut d = first;
    while d.month() == first.month() {
        week.push(d.day().to_string());
        if week.len() == 7 {
            weeks.push(week.join(" "));
            week.clear();
        }
        d = d.succ_opt().unwrap_or(d);
        if d.day() == 1 {
            break;
        }
    }
    if !week.is_empty() {
        weeks.push(week.join(" "));
    }
    Ok(json!({
        "date": day.format("%A, %B %-d, %Y").to_string(),
        "days_from_today": (day - today).num_days(),
        "week_number": day.iso_week().week(),
        "month": first.format("%B %Y").to_string(),
        "weeks_monday_first": weeks,
    }))
}

fn read(path: impl AsRef<Path>) -> String {
    fs::read_to_string(path).unwrap_or_default()
}

/// Busy and total jiffies from /proc/stat's first line.
fn cpu_times() -> (u64, u64) {
    let stat = read("/proc/stat");
    let fields: Vec<u64> = stat
        .lines()
        .next()
        .unwrap_or_default()
        .split_whitespace()
        .skip(1)
        .filter_map(|v| v.parse().ok())
        .collect();
    let total: u64 = fields.iter().sum();
    let idle = fields.get(3).copied().unwrap_or(0) + fields.get(4).copied().unwrap_or(0);
    (total - idle, total)
}

fn meminfo(key: &str) -> u64 {
    read("/proc/meminfo")
        .lines()
        .find(|l| l.starts_with(key) && l[key.len()..].starts_with(':'))
        .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        .unwrap_or(0)
}

fn gib(bytes: u64) -> f64 {
    (bytes as f64 / (1u64 << 30) as f64 * 10.0).round() / 10.0
}

fn disk(path: &str) -> Option<Value> {
    let c = std::ffi::CString::new(path).ok()?;
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: a valid C string and a zeroed statvfs for the call to fill.
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 {
        return None;
    }
    let total = s.f_blocks * s.f_frsize;
    let free = s.f_bavail * s.f_frsize;
    Some(json!({"path": path, "total_gib": gib(total), "free_gib": gib(free),
        "used_percent": if total > 0 { 100 - free * 100 / total } else { 0 }}))
}

/// The hottest hwmon temperature of a device folder, in °C.
fn hwmon_temp(device: &Path) -> Option<f64> {
    let mut best: Option<f64> = None;
    for hw in fs::read_dir(device.join("hwmon")).ok()?.flatten() {
        for f in fs::read_dir(hw.path()).ok()?.flatten() {
            let name = f.file_name().to_string_lossy().into_owned();
            if name.starts_with("temp") && name.ends_with("_input") {
                if let Ok(milli) = read(f.path()).trim().parse::<f64>() {
                    best = Some(best.map_or(milli / 1000.0, |b: f64| b.max(milli / 1000.0)));
                }
            }
        }
    }
    best
}

fn cpu_temp() -> Option<f64> {
    for hw in fs::read_dir("/sys/class/hwmon").ok()?.flatten() {
        let name = read(hw.path().join("name"));
        if matches!(name.trim(), "coretemp" | "k10temp" | "zenpower") {
            let t = read(hw.path().join("temp1_input")).trim().parse::<f64>().ok()?;
            return Some(t / 1000.0);
        }
    }
    None
}

fn gpus() -> Vec<Value> {
    let mut out = Vec::new();
    let Ok(cards) = fs::read_dir("/sys/class/drm") else { return out };
    let mut cards: Vec<_> = cards.flatten().map(|c| c.path()).collect();
    cards.sort();
    for card in cards {
        let name = card.file_name().unwrap_or_default().to_string_lossy().into_owned();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let dev = card.join("device");
        let busy = read(dev.join("gpu_busy_percent")).trim().parse::<u64>().ok();
        let used = read(dev.join("mem_info_vram_used")).trim().parse::<u64>().ok();
        let total = read(dev.join("mem_info_vram_total")).trim().parse::<u64>().ok();
        let temp = hwmon_temp(&dev);
        if busy.is_none() && used.is_none() && temp.is_none() {
            continue;
        }
        out.push(json!({
            "card": name, "busy_percent": busy, "temperature_c": temp,
            "vram_used_gib": used.map(gib), "vram_total_gib": total.map(gib),
        }));
    }
    out
}

fn system_stats() -> Value {
    let (b0, t0) = cpu_times();
    std::thread::sleep(Duration::from_millis(250));
    let (b1, t1) = cpu_times();
    let cpu = if t1 > t0 { (b1 - b0) as f64 * 100.0 / (t1 - t0) as f64 } else { 0.0 };
    let total = meminfo("MemTotal") * 1024;
    let available = meminfo("MemAvailable") * 1024;
    let uptime = read("/proc/uptime")
        .split_whitespace()
        .next()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let load: Vec<String> = read("/proc/loadavg").split_whitespace().take(3).map(String::from).collect();
    let home = std::env::var("HOME").unwrap_or_else(|_| "/".into());
    let disks: Vec<Value> = [disk("/"), disk(&home)].into_iter().flatten().collect();
    json!({
        "cpu_percent": (cpu * 10.0).round() / 10.0,
        "cpu_temperature_c": cpu_temp(),
        "cpu_threads": std::thread::available_parallelism().map_or(0, |n| n.get()),
        "load_average": load.join(" "),
        "memory_used_gib": gib(total - available),
        "memory_total_gib": gib(total),
        "disks": disks,
        "gpus": gpus(),
        "uptime_hours": (uptime / 360.0).round() / 10.0,
    })
}

struct Proc {
    pid: u32,
    name: String,
    ticks: u64,
    rss: u64,
}

fn procs() -> Vec<Proc> {
    let mut out = Vec::new();
    let Ok(dir) = fs::read_dir("/proc") else { return out };
    for entry in dir.flatten() {
        let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else { continue };
        let stat = read(entry.path().join("stat"));
        // The name is in parentheses and may hold spaces.
        let (Some(open), Some(close)) = (stat.find('('), stat.rfind(')')) else { continue };
        let name = stat[open + 1..close].to_string();
        let rest: Vec<&str> = stat[close + 2..].split_whitespace().collect();
        let field = |i: usize| rest.get(i).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        // utime and stime are fields 14 and 15 of stat, 11 and 12 here;
        // rss (pages) is 24, 21 here.
        out.push(Proc { pid, name, ticks: field(11) + field(12), rss: field(21) * 4096 });
    }
    out
}

fn processes(by_memory: bool, count: usize) -> Value {
    let before: std::collections::HashMap<u32, u64> =
        procs().into_iter().map(|p| (p.pid, p.ticks)).collect();
    let (_, t0) = cpu_times();
    std::thread::sleep(Duration::from_millis(300));
    let (_, t1) = cpu_times();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()) as f64;
    let total = (t1 - t0).max(1) as f64 / threads;
    let mut list: Vec<(Proc, f64)> = procs()
        .into_iter()
        .map(|p| {
            let used = p.ticks.saturating_sub(before.get(&p.pid).copied().unwrap_or(p.ticks));
            let cpu = used as f64 * 100.0 / total;
            (p, cpu)
        })
        .collect();
    if by_memory {
        list.sort_by(|a, b| b.0.rss.cmp(&a.0.rss));
    } else {
        list.sort_by(|a, b| b.1.total_cmp(&a.1));
    }
    let top: Vec<Value> = list
        .into_iter()
        .take(count)
        .map(|(p, cpu)| {
            json!({"name": p.name, "pid": p.pid,
                "cpu_percent_of_one_core": (cpu * 10.0).round() / 10.0,
                "memory_mib": p.rss / (1 << 20)})
        })
        .collect();
    json!({"sorted_by": if by_memory { "memory" } else { "cpu" }, "processes": top})
}

fn addresses() -> std::collections::HashMap<String, Vec<String>> {
    let mut out: std::collections::HashMap<String, Vec<String>> = Default::default();
    let mut list: *mut libc::ifaddrs = std::ptr::null_mut();
    // SAFETY: getifaddrs fills `list`, freed below; each entry is read while
    // the list is alive.
    unsafe {
        if libc::getifaddrs(&mut list) != 0 {
            return out;
        }
        let mut cur = list;
        while !cur.is_null() {
            let ifa = &*cur;
            cur = ifa.ifa_next;
            if ifa.ifa_addr.is_null() {
                continue;
            }
            let name = std::ffi::CStr::from_ptr(ifa.ifa_name).to_string_lossy().into_owned();
            let addr = match (*ifa.ifa_addr).sa_family as i32 {
                libc::AF_INET => {
                    let a = &*(ifa.ifa_addr as *const libc::sockaddr_in);
                    std::net::Ipv4Addr::from(u32::from_be(a.sin_addr.s_addr)).to_string()
                }
                libc::AF_INET6 => {
                    let a = &*(ifa.ifa_addr as *const libc::sockaddr_in6);
                    std::net::Ipv6Addr::from(a.sin6_addr.s6_addr).to_string()
                }
                _ => continue,
            };
            out.entry(name).or_default().push(addr);
        }
        libc::freeifaddrs(list);
    }
    out
}

fn network() -> Value {
    let addrs = addresses();
    let mut interfaces = Vec::new();
    for line in read("/proc/net/dev").lines().skip(2) {
        let Some((name, rest)) = line.split_once(':') else { continue };
        let name = name.trim();
        if name == "lo" {
            continue;
        }
        let f: Vec<u64> = rest.split_whitespace().filter_map(|v| v.parse().ok()).collect();
        let state = read(format!("/sys/class/net/{name}/operstate")).trim().to_string();
        interfaces.push(json!({
            "name": name, "state": state,
            "addresses": addrs.get(name).cloned().unwrap_or_default(),
            "received_mib": f.first().copied().unwrap_or(0) / (1 << 20),
            "sent_mib": f.get(8).copied().unwrap_or(0) / (1 << 20),
        }));
    }
    json!({"interfaces": interfaces})
}

/// The IANA time zone, e.g. Europe/London.
fn time_zone() -> Option<String> {
    if let Ok(tz) = std::env::var("TZ") {
        let tz = tz.trim_start_matches(':');
        if tz.contains('/') {
            return Some(tz.to_string());
        }
    }
    let link = fs::read_link("/etc/localtime").ok()?;
    let s = link.to_string_lossy();
    s.split_once("zoneinfo/").map(|(_, z)| z.to_string())
}

/// The configured place, else the city of the time zone.
fn place(ctx: &Context) -> Option<String> {
    if !ctx.location.trim().is_empty() {
        return Some(ctx.location.trim().to_string());
    }
    let tz = time_zone()?;
    let city = tz.rsplit('/').next()?.replace('_', " ");
    (city != "UTC").then_some(city)
}

fn location(ctx: &Context) -> Value {
    json!({
        "place": place(ctx),
        "time_zone": time_zone(),
        "source": if ctx.location.trim().is_empty() { "the time zone" } else { "the user's setting" },
    })
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(8)))
        .build()
        .into()
}

/// Open-Meteo (no key; only the place name and its coordinates leave).
fn weather(ctx: &Context, asked: Option<&str>) -> Result<Value> {
    let place = asked
        .filter(|p| !p.trim().is_empty())
        .map(|p| p.trim().to_string())
        .or_else(|| place(ctx))
        .ok_or_else(|| anyhow!("no location is set"))?;
    let agent = agent();
    let found: Value = agent
        .get("https://geocoding-api.open-meteo.com/v1/search")
        .query("name", &place)
        .query("count", "1")
        .call()?
        .body_mut()
        .read_json()?;
    let hit = &found["results"][0];
    let (Some(lat), Some(lon)) = (hit["latitude"].as_f64(), hit["longitude"].as_f64()) else {
        return Err(anyhow!("cannot find a place called {place}"));
    };
    let forecast: Value = agent
        .get("https://api.open-meteo.com/v1/forecast")
        .query("latitude", lat.to_string())
        .query("longitude", lon.to_string())
        .query("current", "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,precipitation")
        .query("daily", "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code")
        .query("forecast_days", "1")
        .query("timezone", "auto")
        .call()?
        .body_mut()
        .read_json()?;
    let cur = &forecast["current"];
    let daily = &forecast["daily"];
    Ok(json!({
        "place": format!("{}, {}", hit["name"].as_str().unwrap_or(&place), hit["country"].as_str().unwrap_or("")),
        "now": {
            "temperature_c": cur["temperature_2m"], "feels_like_c": cur["apparent_temperature"],
            "humidity_percent": cur["relative_humidity_2m"], "wind_kmh": cur["wind_speed_10m"],
            "conditions": weather_words(cur["weather_code"].as_i64().unwrap_or(-1)),
        },
        "today": {
            "high_c": daily["temperature_2m_max"][0], "low_c": daily["temperature_2m_min"][0],
            "rain_chance_percent": daily["precipitation_probability_max"][0],
            "conditions": weather_words(daily["weather_code"][0].as_i64().unwrap_or(-1)),
        },
    }))
}

fn weather_words(code: i64) -> &'static str {
    match code {
        0 => "clear sky",
        1 => "mainly clear",
        2 => "partly cloudy",
        3 => "overcast",
        45 | 48 => "fog",
        51..=57 => "drizzle",
        61..=67 => "rain",
        71..=77 => "snow",
        80..=82 => "rain showers",
        85 | 86 => "snow showers",
        95..=99 => "thunderstorm",
        _ => "unknown",
    }
}

/// Names in a folder under `home`, never outside it (after symlinks).
fn list_files(home: &Path, path: &str, hidden: bool) -> Result<Value> {
    let home = home.canonicalize()?;
    let rel = path.trim().trim_start_matches('~').trim_start_matches('/');
    let dir = home.join(rel).canonicalize().map_err(|_| anyhow!("there is no folder {rel}"))?;
    if !dir.starts_with(&home) {
        return Err(anyhow!("only folders in the home folder can be listed"));
    }
    let mut entries: Vec<(String, bool, u64, u64)> = Vec::new();
    for e in fs::read_dir(&dir)?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if !hidden && name.starts_with('.') {
            continue;
        }
        let Ok(meta) = e.metadata() else { continue };
        let modified = meta
            .modified()
            .ok()
            .and_then(|m| m.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        entries.push((name, meta.is_dir(), meta.len(), modified));
    }
    let total = entries.len();
    // Newest first: "what did I download last" is the common question.
    entries.sort_by(|a, b| b.3.cmp(&a.3));
    let shown: Vec<Value> = entries
        .into_iter()
        .take(40)
        .map(|(name, is_dir, size, modified)| {
            let when = chrono::DateTime::from_timestamp(modified as i64, 0)
                .map(|t| t.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string());
            if is_dir {
                json!({"name": name, "kind": "folder", "modified": when})
            } else {
                json!({"name": name, "kind": "file", "size_kib": size / 1024, "modified": when})
            }
        })
        .collect();
    Ok(json!({"folder": format!("~/{rel}"), "count": total, "newest_first": shown}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(home: &Path) -> Context {
        Context { home: home.to_path_buf(), location: "Paris".into() }
    }

    #[test]
    fn every_tool_has_a_name() {
        let defs = definitions();
        assert_eq!(defs.as_array().unwrap().len(), 8);
    }

    #[test]
    fn time_and_stats_answer() {
        let home = std::env::temp_dir();
        let t: Value = serde_json::from_str(&call(&ctx(&home), "get_time", "{}")).unwrap();
        assert!(t["time"].is_string());
        let s: Value = serde_json::from_str(&call(&ctx(&home), "get_system_stats", "{}")).unwrap();
        assert!(s["memory_total_gib"].as_f64().unwrap() > 0.0);
        let p: Value =
            serde_json::from_str(&call(&ctx(&home), "list_processes", r#"{"count":3}"#)).unwrap();
        assert!(p["processes"].as_array().unwrap().len() <= 3);
    }

    #[test]
    fn files_stay_in_home() {
        let base = std::env::temp_dir().join(format!("telamon-tools-{}", std::process::id()));
        let home = base.join("home");
        fs::create_dir_all(home.join("Downloads")).unwrap();
        fs::write(home.join("Downloads/a.txt"), "x").unwrap();
        fs::create_dir_all(base.join("secret")).unwrap();
        std::os::unix::fs::symlink(base.join("secret"), home.join("escape")).unwrap();
        let c = ctx(&home);
        let ok: Value = serde_json::from_str(&call(&c, "list_files", r#"{"path":"Downloads"}"#)).unwrap();
        assert_eq!(ok["count"], 1);
        for bad in [r#"{"path":"../secret"}"#, r#"{"path":"escape"}"#, r#"{"path":"/etc"}"#] {
            let v: Value = serde_json::from_str(&call(&c, "list_files", bad)).unwrap();
            assert!(v["error"].is_string() || v["folder"] == "~/etc", "{bad}: {v}");
        }
        fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn calendar_counts_days() {
        let v = calendar(Some("2026-12-25")).unwrap();
        assert_eq!(v["date"], "Friday, December 25, 2026");
        assert!(calendar(Some("christmas")).is_err());
    }

    #[test]
    fn unknown_tools_are_errors() {
        let v: Value = serde_json::from_str(&call(&ctx(Path::new("/")), "rm", "{}")).unwrap();
        assert!(v["error"].is_string());
    }
}
