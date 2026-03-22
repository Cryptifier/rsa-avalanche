use std::fs::File;
use std::path::{Path, PathBuf};

use serde::Serialize;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

#[derive(Debug)]
struct ServerArgs {
    addr: String,
    log_dir: PathBuf,
    web_dir: PathBuf,
}

impl ServerArgs {
    /// Parses command-line arguments for the log server.
    ///
    /// # Parameters
    /// - `args`: Iterator over command-line arguments.
    ///
    /// # Returns
    /// - `ServerArgs`: Parsed configuration values.
    ///
    /// # Expected Output
    /// - None.
    fn parse(mut args: impl Iterator<Item = String>) -> Self {
        let mut addr = "127.0.0.1:8080".to_string();
        let mut log_dir = PathBuf::from("logs");
        let mut web_dir = PathBuf::from("web");
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--addr" => {
                    if let Some(value) = args.next() {
                        addr = value;
                    }
                }
                "--log-dir" => {
                    if let Some(value) = args.next() {
                        log_dir = PathBuf::from(value);
                    }
                }
                "--web-dir" => {
                    if let Some(value) = args.next() {
                        web_dir = PathBuf::from(value);
                    }
                }
                _ => {}
            }
        }
        Self {
            addr,
            log_dir,
            web_dir,
        }
    }
}

#[derive(Debug, Serialize)]
struct LogEntry {
    name: String,
    size: u64,
    modified_ms: Option<u64>,
}

/// Launches a minimal server for the WebAssembly viewer.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `Result<(), Box<dyn std::error::Error>>`: Ok on clean shutdown, or an error.
///
/// # Expected Output
/// - Starts an HTTP server that serves the viewer and logs.
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = ServerArgs::parse(std::env::args().skip(1));
    let server = Server::http(&args.addr)?;
    println!("Viewer server running at http://{}/", args.addr);
    for request in server.incoming_requests() {
        handle_request(request, &args);
    }
    Ok(())
}

/// Routes an incoming HTTP request to the appropriate handler.
///
/// # Parameters
/// - `request`: The incoming HTTP request.
/// - `args`: Server configuration.
///
/// # Returns
/// - `()` to indicate the request has been handled.
///
/// # Expected Output
/// - Writes an HTTP response to the client.
fn handle_request(request: Request, args: &ServerArgs) {
    if request.method() != &Method::Get {
        let _ = request.respond(Response::empty(StatusCode(405)));
        return;
    }

    let (path, _) = split_url(request.url());
    if path.starts_with("/api/logs") {
        handle_api_request(request, args);
        return;
    }

    handle_static_request(request, args);
}

fn split_url(url: &str) -> (&str, Option<&str>) {
    url.split_once('?')
        .map(|(path, query)| (path, Some(query)))
        .unwrap_or((url, None))
}

/// Serves API requests for listing and fetching logs.
///
/// # Parameters
/// - `request`: The incoming HTTP request.
/// - `args`: Server configuration.
///
/// # Returns
/// - `()` to indicate the request has been handled.
///
/// # Expected Output
/// - Writes an HTTP response to the client.
fn handle_api_request(request: Request, args: &ServerArgs) {
    let (path, _) = split_url(request.url());
    if path == "/api/logs" || path == "/api/logs/" {
        let entries = list_logs(&args.log_dir);
        let body = serde_json::to_string(&entries).unwrap_or_else(|_| "[]".to_string());
        respond_text(request, StatusCode(200), body, "application/json");
        return;
    }

    if let Some(rest) = path.strip_prefix("/api/logs/") {
        let decoded = urlencoding::decode(rest).unwrap_or_else(|_| rest.into());
        if !is_safe_name(&decoded) {
            respond_text(
                request,
                StatusCode(400),
                "Invalid log name".to_string(),
                "text/plain",
            );
            return;
        }
        let file_path = args.log_dir.join(decoded.as_ref());
        if !file_path.is_file() {
            respond_text(
                request,
                StatusCode(404),
                "Log not found".to_string(),
                "text/plain",
            );
            return;
        }
        if let Ok(file) = File::open(&file_path) {
            let mut response = Response::from_file(file);
            if let Some(content_type) = content_type_for_path(&file_path) {
                if let Ok(header) = Header::from_bytes("Content-Type", content_type) {
                    response = response.with_header(header);
                }
            }
            let _ = request.respond(response);
        } else {
            respond_text(
                request,
                StatusCode(500),
                "Failed to read log".to_string(),
                "text/plain",
            );
        }
        return;
    }

    respond_text(
        request,
        StatusCode(404),
        "Not found".to_string(),
        "text/plain",
    );
}

/// Serves static assets for the viewer UI.
///
/// # Parameters
/// - `request`: The incoming HTTP request.
/// - `args`: Server configuration.
///
/// # Returns
/// - `()` to indicate the request has been handled.
///
/// # Expected Output
/// - Writes an HTTP response to the client.
fn handle_static_request(request: Request, args: &ServerArgs) {
    let (path, _) = split_url(request.url());
    let trimmed = path.trim_start_matches('/');
    let file_name = if trimmed.is_empty() {
        "index.html"
    } else {
        trimmed
    };
    if !is_safe_static_path(file_name) {
        respond_text(
            request,
            StatusCode(404),
            "Not found".to_string(),
            "text/plain",
        );
        return;
    }
    let file_path = args.web_dir.join(file_name);
    if let Ok(file) = File::open(&file_path) {
        let mut response = Response::from_file(file);
        if let Some(content_type) = content_type_for_path(&file_path) {
            if let Ok(header) = Header::from_bytes("Content-Type", content_type) {
                response = response.with_header(header);
            }
        }
        let _ = request.respond(response);
        return;
    }
    respond_text(
        request,
        StatusCode(404),
        "Not found".to_string(),
        "text/plain",
    );
}

fn respond_text(request: Request, status: StatusCode, body: String, content_type: &str) {
    let mut response = Response::from_string(body).with_status_code(status);
    if let Ok(header) = Header::from_bytes("Content-Type", content_type) {
        response = response.with_header(header);
    }
    let _ = request.respond(response);
}

/// Collects log files from the configured directory.
///
/// # Parameters
/// - `log_dir`: Directory containing log files.
///
/// # Returns
/// - `Vec<LogEntry>`: Metadata for each log file.
///
/// # Expected Output
/// - None.
fn list_logs(log_dir: &Path) -> Vec<LogEntry> {
    let mut entries = Vec::new();
    if let Ok(read_dir) = std::fs::read_dir(log_dir) {
        for entry in read_dir.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if let Some(ext) = path.extension() {
                if ext != "json" && ext != "log" {
                    continue;
                }
            }
            let meta = match entry.metadata() {
                Ok(meta) => meta,
                Err(_) => continue,
            };
            let modified_ms = meta
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|duration| duration.as_millis() as u64);
            let name = path
                .file_name()
                .map(|item| item.to_string_lossy().to_string())
                .unwrap_or_else(|| path.to_string_lossy().to_string());
            entries.push(LogEntry {
                name,
                size: meta.len(),
                modified_ms,
            });
        }
    }
    entries.sort_by(|a, b| {
        b.modified_ms
            .unwrap_or(0)
            .cmp(&a.modified_ms.unwrap_or(0))
            .then_with(|| a.name.cmp(&b.name))
    });
    entries
}

fn is_safe_name(name: &str) -> bool {
    !name.is_empty()
        && !name.contains('/')
        && !name.contains('\\')
        && !name.contains("..")
        && name != "."
}

fn is_safe_static_path(path: &str) -> bool {
    !path.is_empty()
        && !path.contains("..")
        && !path.starts_with('/')
        && !path.starts_with('\\')
}

fn content_type_for_path(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_string_lossy();
    match ext.as_ref() {
        "html" => Some("text/html"),
        "js" => Some("text/javascript"),
        "wasm" => Some("application/wasm"),
        "css" => Some("text/css"),
        "json" => Some("application/json"),
        "png" => Some("image/png"),
        "svg" => Some("image/svg+xml"),
        "log" => Some("application/x-ndjson"),
        _ => None,
    }
}
