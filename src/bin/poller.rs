use std::thread;
use std::time::Duration;

use clap::Parser;
use rsademo::zmq_status::{PingClientBuilder, RouterCommand, RouterReply, ZmqConnectAddress};

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
    let client = PingClientBuilder::new()
        .connect_address(ZmqConnectAddress::new(args.host.clone(), args.port))
        .send_timeout_ms(Some(args.timeout_ms))
        .recv_timeout_ms(Some(args.timeout_ms))
        .build()?;

    println!("Connecting to {}", client.endpoint());
    for attempt in 1..=args.count {
        println!("[{attempt}/{}] -> {}", args.count, RouterCommand::Ping);
        let reply = client.send_command(RouterCommand::Ping)?;
        println!("[{attempt}/{}] <- {}", args.count, reply);
        validate_ping_reply(&reply)?;

        if args.status_after_each {
            let status = client.query_status()?;
            println!(
                "[{attempt}/{}] <- {}",
                args.count,
                RouterReply::Status(status)
            );
        }

        if attempt < args.count {
            thread::sleep(Duration::from_millis(args.interval_ms));
        }
    }

    if args.stop_after {
        println!("-> {}", RouterCommand::Stop);
        let reply = client.send_command(RouterCommand::Stop)?;
        println!("<- {}", reply);
        match reply {
            RouterReply::Stopped => {}
            other => return Err(format!("unexpected stop reply: {other}").into()),
        }
    }

    Ok(())
}

/// Validates that a poller ping round-trip returned PONG.
///
/// # Parameters
/// - `reply`: Typed router reply to validate.
///
/// # Returns
/// - `Result<(), Box<dyn std::error::Error + Send + Sync>>`: Ok on PONG or an error otherwise.
///
/// # Expected Output
/// - Returns validation status; no stdout/stderr output.
fn validate_ping_reply(
    reply: &RouterReply,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match reply {
        RouterReply::Pong => Ok(()),
        other => Err(format!("unexpected ping reply: {other}").into()),
    }
}
