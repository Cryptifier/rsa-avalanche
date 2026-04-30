/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::Parser;
use rsademo::zmq_status::{QueryResponsePayload, RouterCommand, RouterReply, ZmqConnectAddress};
use zmq::{Context, Socket};

#[derive(Debug, Parser, Clone)]
#[command(
    name = "poller",
    about = "Send ZMQ ping requests to a ROUTER endpoint and print replies",
    author,
    version
)]
struct PollerArgs {
    /// ZMQ target host name or IP address
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// ZMQ target TCP port
    #[arg(long, default_value_t = 5555)]
    port: u16,

    /// Stable identity exposed to the ROUTER as this poller client ID
    #[arg(long)]
    client_id: Option<String>,

    /// Number of ping requests to send
    #[arg(long, default_value_t = 1)]
    count: usize,

    /// Delay between ping requests in milliseconds
    #[arg(long, default_value_t = 1000)]
    interval_ms: u64,

    /// REQ send and receive timeout in milliseconds
    #[arg(long, default_value_t = 1000)]
    timeout_ms: i32,

    /// Query the router status after each ping
    #[arg(long)]
    status_after_each: bool,

    /// Send STOP after the ping loop completes
    #[arg(long)]
    stop_after: bool,
}

/// Runs the CLI poller entry point.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `Result<(), Box<dyn std::error::Error>>`: Ok on success or an error.
///
/// # Expected Output
/// - Sends ping requests and prints request/response lines to stdout.
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = PollerArgs::parse();
    run_poller(args)
}

/// Executes the configured polling loop against the ZMQ router.
///
/// # Parameters
/// - `args`: Parsed CLI arguments.
///
/// # Returns
/// - `Result<(), Box<dyn std::error::Error + Send + Sync>>`: Ok on success or an error.
///
/// # Expected Output
/// - Prints ping requests, replies, optional status queries, and optional stop output.
fn run_poller(args: PollerArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client_id = args.client_id.clone().unwrap_or_else(default_client_id);
    let endpoint = ZmqConnectAddress::new(args.host.clone(), args.port).to_endpoint();
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    install_ctrlc_handler(Arc::clone(&shutdown_requested))?;
    let socket = build_dealer_socket(&endpoint, &client_id, args.timeout_ms)?;

    println!("Connecting to {} as {}", endpoint, client_id);
    run_poller_loop(socket, &args, &client_id, shutdown_requested)
}

/// Runs the persistent poller loop for pings and server-initiated queries.
///
/// # Parameters
/// - `socket`: Connected DEALER socket.
/// - `args`: Parsed CLI arguments.
/// - `client_id`: Stable client identity exposed to the server.
/// - `shutdown_requested`: Shared shutdown flag updated by the signal handler.
///
/// # Returns
/// - `Result<(), Box<dyn std::error::Error + Send + Sync>>`: Ok on clean shutdown or an error.
///
/// # Expected Output
/// - Sends ping requests, handles query messages, and prints request/response lines to stdout.
fn run_poller_loop(
    socket: Socket,
    args: &PollerArgs,
    client_id: &str,
    shutdown_requested: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut sent_pings = 0usize;
    let mut completed_pings = 0usize;
    let mut awaiting_status_reply = false;
    let mut awaiting_stop_reply = false;
    let mut announced_idle = false;
    let mut next_ping_at = Instant::now();

    while !shutdown_requested.load(Ordering::Relaxed) {
        if sent_pings < args.count
            && completed_pings == sent_pings
            && !awaiting_status_reply
            && !awaiting_stop_reply
            && Instant::now() >= next_ping_at
        {
            let attempt = sent_pings + 1;
            println!("[{attempt}/{}] -> {}", args.count, RouterCommand::Ping);
            send_command(&socket, &RouterCommand::Ping)?;
            sent_pings += 1;
        }

        match socket.recv_msg(0) {
            Ok(message) => {
                let reply_text = message
                    .as_str()
                    .ok_or_else(|| "router reply was not valid UTF-8".to_string())?;
                let reply = RouterReply::from_text(reply_text);
                match reply {
                    RouterReply::Pong => {
                        completed_pings += 1;
                        println!(
                            "[{completed_pings}/{}] <- {}",
                            args.count,
                            RouterReply::Pong
                        );
                        if args.status_after_each {
                            println!(
                                "[{completed_pings}/{}] -> {}",
                                args.count,
                                RouterCommand::Status
                            );
                            send_command(&socket, &RouterCommand::Status)?;
                            awaiting_status_reply = true;
                        } else if completed_pings < args.count {
                            next_ping_at = Instant::now() + Duration::from_millis(args.interval_ms);
                        } else if args.stop_after {
                            println!("-> {}", RouterCommand::Stop);
                            send_command(&socket, &RouterCommand::Stop)?;
                            awaiting_stop_reply = true;
                        } else if !announced_idle {
                            println!("Ping loop complete; waiting for QUERY until Ctrl-C.");
                            announced_idle = true;
                        }
                    }
                    RouterReply::Query => {
                        println!("<- {}", RouterReply::Query);
                        let payload = build_query_response_payload(client_id)?;
                        let response = RouterCommand::QueryResponse(payload);
                        println!("-> {}", response);
                        send_command(&socket, &response)?;
                    }
                    RouterReply::Status(value) => {
                        if !awaiting_status_reply {
                            return Err(format!(
                                "unexpected status reply: {}",
                                RouterReply::Status(value)
                            )
                            .into());
                        }
                        println!(
                            "[{completed_pings}/{}] <- {}",
                            args.count,
                            RouterReply::Status(value)
                        );
                        awaiting_status_reply = false;
                        if completed_pings < args.count {
                            next_ping_at = Instant::now() + Duration::from_millis(args.interval_ms);
                        } else if args.stop_after {
                            println!("-> {}", RouterCommand::Stop);
                            send_command(&socket, &RouterCommand::Stop)?;
                            awaiting_stop_reply = true;
                        } else if !announced_idle {
                            println!("Ping loop complete; waiting for QUERY until Ctrl-C.");
                            announced_idle = true;
                        }
                    }
                    RouterReply::Clients(_) => {
                        return Err("unexpected clients reply".into());
                    }
                    RouterReply::Stopped => {
                        if !awaiting_stop_reply {
                            return Err("unexpected stop reply".into());
                        }
                        println!("<- {}", RouterReply::Stopped);
                        return Ok(());
                    }
                    RouterReply::Error(message) => {
                        return Err(format!("router returned error: {message}").into());
                    }
                }
            }
            Err(zmq::Error::EAGAIN) => continue,
            Err(err) => return Err(format!("poller receive error: {err}").into()),
        }
    }

    println!("Shutdown requested; closing poller.");
    Ok(())
}

/// Builds a connected DEALER socket for the persistent poller loop.
///
/// # Parameters
/// - `endpoint`: TCP endpoint string in `tcp://host:port` form.
/// - `client_id`: Stable socket identity used by the ROUTER.
/// - `timeout_ms`: Send and receive timeout in milliseconds.
///
/// # Returns
/// - `Result<Socket, String>`: Connected DEALER socket or an error string.
///
/// # Expected Output
/// - Creates and connects a DEALER socket; no stdout/stderr output.
fn build_dealer_socket(endpoint: &str, client_id: &str, timeout_ms: i32) -> Result<Socket, String> {
    let context = Context::new();
    let socket = context
        .socket(zmq::DEALER)
        .map_err(|err| format!("dealer socket error: {err}"))?;
    socket
        .set_identity(client_id.as_bytes())
        .map_err(|err| format!("dealer identity error: {err}"))?;
    socket
        .set_linger(0)
        .map_err(|err| format!("dealer linger error: {err}"))?;
    socket
        .set_sndtimeo(timeout_ms)
        .map_err(|err| format!("dealer send timeout error: {err}"))?;
    socket
        .set_rcvtimeo(timeout_ms)
        .map_err(|err| format!("dealer receive timeout error: {err}"))?;
    socket
        .connect(endpoint)
        .map_err(|err| format!("dealer connect error: {err}"))?;
    Ok(socket)
}

/// Sends one typed command over the DEALER socket.
///
/// # Parameters
/// - `socket`: Connected DEALER socket.
/// - `command`: Protocol message to send.
///
/// # Returns
/// - `Result<(), String>`: `Ok(())` on success or an error string.
///
/// # Expected Output
/// - Sends one UTF-8 message to the ROUTER; no stdout/stderr output.
fn send_command(socket: &Socket, command: &RouterCommand) -> Result<(), String> {
    socket
        .send(command.to_string().as_bytes(), 0)
        .map_err(|err| format!("poller send error: {err}"))?;
    Ok(())
}

/// Builds the resource snapshot returned in a `QUERY_RESPONSE`.
///
/// # Parameters
/// - `client_id`: Stable poller identity to include in the payload.
///
/// # Returns
/// - `Result<QueryResponsePayload, String>`: Snapshot payload or an error string.
///
/// # Expected Output
/// - Reads local system resource data; no stdout/stderr output.
fn build_query_response_payload(client_id: &str) -> Result<QueryResponsePayload, String> {
    Ok(QueryResponsePayload {
        client_id: client_id.to_string(),
        cpu_cores: std::thread::available_parallelism()
            .map_err(|err| format!("cpu core query error: {err}"))?
            .get(),
        available_memory_bytes: available_memory_bytes()?,
    })
}

/// Returns the available system memory in bytes for the current poller host.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `Result<u64, String>`: Available memory in bytes or an error string.
///
/// # Expected Output
/// - Queries the operating system for available memory; no stdout/stderr output.
fn available_memory_bytes() -> Result<u64, String> {
    #[cfg(target_family = "unix")]
    unsafe {
        let page_size = libc::sysconf(libc::_SC_PAGESIZE);
        let available_pages = libc::sysconf(libc::_SC_AVPHYS_PAGES);
        if page_size <= 0 || available_pages < 0 {
            return Err("available memory query returned an invalid value".to_string());
        }
        let total_bytes = (page_size as u128) * (available_pages as u128);
        return u64::try_from(total_bytes).map_err(|_| "available memory exceeds u64".to_string());
    }

    #[cfg(not(target_family = "unix"))]
    {
        Err("available memory query is unsupported on this platform".to_string())
    }
}

/// Installs a Ctrl+C handler that requests a clean poller shutdown.
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

/// Generates a stable default client identity for the current poller process.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `String`: Generated client identifier.
///
/// # Expected Output
/// - Returns a client ID string; no side effects.
fn default_client_id() -> String {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("poller-{}-{now_ms}", std::process::id())
}
