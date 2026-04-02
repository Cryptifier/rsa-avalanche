use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use clap::Parser;
use rsademo::zmq_status::{
    RouterServerBuilder, ZmqBindAddress, ZmqStatusContext, join_router, router_endpoint,
    stop_router,
};
use serde::Serialize;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

#[derive(Debug, Parser, Clone)]
#[command(
    name = "server",
    about = "Serve the viewer HTTP assets and a concurrent ZMQ ping router",
    author,
    version
)]
struct ServerArgs {
    /// HTTP bind address for the viewer server
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: String,

    /// Directory containing session log files
    #[arg(long, default_value = "logs")]
    log_dir: PathBuf,

    /// Directory containing the web viewer assets
    #[arg(long, default_value = "web")]
    web_dir: PathBuf,

    /// Host or interface for the ZMQ ROUTER socket
    #[arg(long, default_value = "127.0.0.1")]
    zmq_host: String,

    /// TCP port for the ZMQ ROUTER socket
    #[arg(long, default_value_t = 5555)]
    zmq_port: u16,

    /// Stop the ZMQ router automatically after this many pings
    #[arg(long)]
    zmq_expected_pings: Option<usize>,

    /// ZMQ linger timeout in milliseconds
    #[arg(long, default_value_t = 0)]
    zmq_linger_ms: i32,

    /// HTTP receive poll timeout in milliseconds
    #[arg(long, default_value_t = 250)]
    http_poll_timeout_ms: u64,
}

#[derive(Debug, Serialize)]
struct LogEntry {
    name: String,
    size: u64,
    modified_ms: Option<u64>,
}

/// Launches the viewer HTTP server together with a background ZMQ ROUTER server.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `Result<(), Box<dyn std::error::Error>>`: Ok on clean shutdown, or an error.
///
/// # Expected Output
/// - Starts an HTTP server, starts a ZMQ router, and prints both endpoints to stdout.
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = ServerArgs::parse();
    run_server(args)
}

/// Runs the combined HTTP and ZMQ server workflow.
///
/// # Parameters
/// - `args`: Parsed CLI arguments for HTTP and ZMQ configuration.
///
/// # Returns
/// - `Result<(), Box<dyn std::error::Error + Send + Sync>>`: Ok on clean shutdown, or an error.
///
/// # Expected Output
/// - Starts the configured servers, serves requests, and prints startup/shutdown status.
fn run_server(args: ServerArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let http_server = Server::http(&args.addr)?;
    let status_context = Arc::new(Mutex::new(ZmqStatusContext::new()));
    let router_builder = build_router_builder(&args);
    let router_handle = router_builder.build_with_shared_context(Arc::clone(&status_context))?;
    let router_endpoint_value = router_endpoint(&router_handle).to_string();
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    install_ctrlc_handler(Arc::clone(&shutdown_requested))?;

    println!("Viewer server running at http://{}/", args.addr);
    println!("ZMQ router listening at {}", router_endpoint_value);
    println!("ZMQ router will query connected pollers every 10 seconds.");

    while !shutdown_requested.load(Ordering::Relaxed) {
        match http_server.recv_timeout(Duration::from_millis(args.http_poll_timeout_ms)) {
            Ok(Some(request)) => handle_request(request, &args),
            Ok(None) => {}
            Err(err) => return Err(err.into()),
        }
    }

    println!("Shutdown requested; stopping ZMQ router.");
    let _ = stop_router(&router_endpoint_value);
    let router_stats = join_router(router_handle)?;
    println!(
        "ZMQ router stopped after {} pings, {} status queries, {} query requests, and {} query responses across {} known clients.",
        router_stats.pings,
        router_stats.status_queries,
        router_stats.query_requests_sent,
        router_stats.query_responses,
        router_stats.known_clients
    );
    Ok(())
}

/// Builds the ROUTER server configuration from CLI arguments.
///
/// # Parameters
/// - `args`: Parsed CLI arguments.
///
/// # Returns
/// - `RouterServerBuilder`: Configured builder for the ZMQ router.
///
/// # Expected Output
/// - Returns a builder value; no side effects.
fn build_router_builder(args: &ServerArgs) -> RouterServerBuilder {
    let builder = RouterServerBuilder::new()
        .bind_address(ZmqBindAddress::new(args.zmq_host.clone(), args.zmq_port))
        .linger_ms(args.zmq_linger_ms);
    match args.zmq_expected_pings {
        Some(expected_pings) => builder.expected_pings(expected_pings),
        None => builder.until_stopped(),
    }
}

/// Installs a Ctrl+C handler that requests a clean shutdown.
///
/// # Parameters
/// - `shutdown_requested`: Shared shutdown flag updated by the signal handler.
///
/// # Returns
/// - `Result<(), ctrlc::Error>`: `Ok(())` on success or a handler registration error.
///
/// # Expected Output
/// - Registers a signal handler; no immediate stdout/stderr output.
fn install_ctrlc_handler(shutdown_requested: Arc<AtomicBool>) -> Result<(), ctrlc::Error> {
    ctrlc::set_handler(move || {
        shutdown_requested.store(true, Ordering::Relaxed);
    })
}

/// Routes an incoming HTTP request to the appropriate handler.
///
/// # Parameters
/// - `request`: The incoming HTTP request.
/// - `args`: Server configuration.
///
/// # Returns
/// - `()`: This function returns nothing.
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
/// - `()`: This function returns nothing.
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
/// - `()`: This function returns nothing.
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
/// - Returns file metadata; no stdout/stderr output.
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
    !path.is_empty() && !path.contains("..") && !path.starts_with('/') && !path.starts_with('\\')
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
