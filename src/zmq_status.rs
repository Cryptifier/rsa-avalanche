use std::collections::HashMap;
use std::fmt;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
    mpsc,
};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use zmq::{Context, Socket};

static GLOBAL_STATUS: AtomicU64 = AtomicU64::new(0);

const TCP_SCHEME: &str = "tcp";
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_CLIENT_PORT: u16 = 5555;
const DEFAULT_LINGER_MS: i32 = 0;
const PING_MESSAGE: &str = "PING";
const PONG_MESSAGE: &str = "PONG";
const QUERY_MESSAGE: &str = "QUERY";
const QUERY_RESPONSE_MESSAGE: &str = "QUERY_RESPONSE";
const STATUS_MESSAGE: &str = "STATUS";
const STOP_MESSAGE: &str = "STOP";
const STOPPED_MESSAGE: &str = "STOPPED";
const UNKNOWN_MESSAGE_REPLY: &str = "ERROR";
const DEFAULT_QUERY_INTERVAL_SECS: u64 = 10;

/// Shared context for reporting r-candidate generation status.
#[derive(Debug, Clone, Default)]
pub struct ZmqStatusContext {
    status: u64,
}

/// Lifecycle mode for a ROUTER server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterRunMode {
    /// Keep processing requests until a STOP command is received.
    UntilStopped,
    /// Stop after the configured number of successful ping commands.
    ExpectedPings(usize),
}

/// Typed TCP bind address for a ROUTER server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZmqBindAddress {
    host: String,
    port: Option<u16>,
}

/// Typed TCP connect address for a REQ client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZmqConnectAddress {
    host: String,
    port: u16,
}

/// Resource snapshot returned by a poller in response to `QUERY`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryResponsePayload {
    /// Client identity used by the poller.
    pub client_id: String,
    /// Number of CPU cores available to the poller process.
    pub cpu_cores: usize,
    /// Available system memory in bytes observed by the poller.
    pub available_memory_bytes: u64,
}

/// Structured command accepted by the ROUTER server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouterCommand {
    /// Increment the ping counter and reply with PONG.
    Ping,
    /// Carry a JSON resource snapshot from a poller back to the router.
    QueryResponse(QueryResponsePayload),
    /// Return the current status mirror as a decimal string.
    Status,
    /// Stop the router after replying.
    Stop,
    /// Unsupported command text.
    Unknown(String),
}

/// Structured reply emitted by the ROUTER server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouterReply {
    /// Successful reply for a ping request.
    Pong,
    /// Query a connected poller for its resource snapshot.
    Query,
    /// Current status value.
    Status(u64),
    /// Successful reply for a stop request.
    Stopped,
    /// Error reply text.
    Error(String),
}

/// Immutable router configuration built by `RouterServerBuilder`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterServerConfig {
    bind_address: ZmqBindAddress,
    linger_ms: i32,
    query_interval: Duration,
    run_mode: RouterRunMode,
}

/// Builder for configuring and starting a ROUTER server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterServerBuilder {
    bind_address: ZmqBindAddress,
    linger_ms: i32,
    query_interval: Duration,
    run_mode: RouterRunMode,
}

/// Immutable REQ client configuration built by `PingClientBuilder`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PingClientConfig {
    connect_address: ZmqConnectAddress,
    linger_ms: i32,
    send_timeout_ms: Option<i32>,
    recv_timeout_ms: Option<i32>,
}

/// Builder for configuring a REQ client used to ping the router.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PingClientBuilder {
    connect_address: ZmqConnectAddress,
    linger_ms: i32,
    send_timeout_ms: Option<i32>,
    recv_timeout_ms: Option<i32>,
}

/// Strongly typed REQ client for the ZMQ status protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PingClient {
    config: PingClientConfig,
    endpoint: String,
}

/// Summary statistics for a router run.
#[derive(Debug, Clone)]
pub struct RouterStats {
    /// Number of PING messages processed.
    pub pings: usize,
    /// Number of QUERY messages sent to connected pollers.
    pub query_requests_sent: usize,
    /// Number of QUERY_RESPONSE messages processed.
    pub query_responses: usize,
    /// Number of STATUS messages processed.
    pub status_queries: usize,
    /// Final status value in the context.
    pub final_status: u64,
    /// Number of known client identities seen by the router.
    pub known_clients: usize,
    /// Whether the router stopped because it received a STOP message.
    pub stop_requested: bool,
}

#[derive(Debug, Clone)]
struct KnownClient {
    routing_id: Vec<u8>,
}

/// Handle to a background ROUTER thread.
///
/// # Expected Output
/// - Owns the background thread handle; no side effects.
pub struct RouterHandle {
    endpoint: String,
    join: thread::JoinHandle<Result<RouterStats, String>>,
}

impl ZmqStatusContext {
    /// Creates a new status context with a zero value.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `ZmqStatusContext`: New context instance.
    ///
    /// # Expected Output
    /// - Returns a new context; no side effects.
    pub fn new() -> Self {
        Self { status: 0 }
    }

    /// Updates the status value and global status mirror.
    ///
    /// # Parameters
    /// - `value`: Status value to store.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Updates the context and global status; no stdout/stderr output.
    pub fn set_status(&mut self, value: u64) {
        self.status = value;
        GLOBAL_STATUS.store(value, Ordering::Relaxed);
    }

    /// Returns the current status value.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `u64`: Stored status value.
    ///
    /// # Expected Output
    /// - Returns the stored value; no side effects.
    pub fn status(&self) -> u64 {
        self.status
    }

    /// Returns the global status mirror.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `u64`: Global status value.
    ///
    /// # Expected Output
    /// - Returns the stored value; no side effects.
    pub fn global_status() -> u64 {
        GLOBAL_STATUS.load(Ordering::Relaxed)
    }

    /// Resets the global status mirror to zero.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Sets the global status to zero; no stdout/stderr output.
    pub fn reset_global_status() {
        GLOBAL_STATUS.store(0, Ordering::Relaxed);
    }
}

impl Default for RouterServerBuilder {
    fn default() -> Self {
        Self {
            bind_address: ZmqBindAddress::localhost_random_port(),
            linger_ms: DEFAULT_LINGER_MS,
            query_interval: Duration::from_secs(DEFAULT_QUERY_INTERVAL_SECS),
            run_mode: RouterRunMode::UntilStopped,
        }
    }
}

impl RouterServerBuilder {
    /// Creates a new ROUTER server builder with localhost defaults.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `RouterServerBuilder`: Builder with default bind host and random port.
    ///
    /// # Expected Output
    /// - Returns a builder; no side effects.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the full bind address for the ROUTER socket.
    ///
    /// # Parameters
    /// - `bind_address`: Typed bind address to use.
    ///
    /// # Returns
    /// - `RouterServerBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn bind_address(mut self, bind_address: ZmqBindAddress) -> Self {
        self.bind_address = bind_address;
        self
    }

    /// Sets the bind host for the ROUTER socket.
    ///
    /// # Parameters
    /// - `host`: Host or interface name.
    ///
    /// # Returns
    /// - `RouterServerBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.bind_address.host = host.into();
        self
    }

    /// Sets the bind port for the ROUTER socket.
    ///
    /// # Parameters
    /// - `port`: TCP port to bind.
    ///
    /// # Returns
    /// - `RouterServerBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn port(mut self, port: u16) -> Self {
        self.bind_address.port = Some(port);
        self
    }

    /// Configures the ROUTER socket to bind an ephemeral local port.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `RouterServerBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn random_port(mut self) -> Self {
        self.bind_address.port = None;
        self
    }

    /// Sets the socket linger value in milliseconds.
    ///
    /// # Parameters
    /// - `linger_ms`: Linger timeout in milliseconds.
    ///
    /// # Returns
    /// - `RouterServerBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn linger_ms(mut self, linger_ms: i32) -> Self {
        self.linger_ms = linger_ms;
        self
    }

    /// Sets the interval between server-initiated poller queries.
    ///
    /// # Parameters
    /// - `query_interval`: Duration between `QUERY` broadcasts to known clients.
    ///
    /// # Returns
    /// - `RouterServerBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn query_interval(mut self, query_interval: Duration) -> Self {
        self.query_interval = query_interval;
        self
    }

    /// Configures the router to stop only after a STOP command.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `RouterServerBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn until_stopped(mut self) -> Self {
        self.run_mode = RouterRunMode::UntilStopped;
        self
    }

    /// Configures the router to stop after a fixed number of pings.
    ///
    /// # Parameters
    /// - `expected_pings`: Number of PING commands to process before stopping.
    ///
    /// # Returns
    /// - `RouterServerBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn expected_pings(mut self, expected_pings: usize) -> Self {
        self.run_mode = RouterRunMode::ExpectedPings(expected_pings);
        self
    }

    /// Builds an immutable router configuration.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `RouterServerConfig`: Immutable router configuration.
    ///
    /// # Expected Output
    /// - Returns a config; no side effects.
    pub fn build_config(self) -> RouterServerConfig {
        RouterServerConfig {
            bind_address: self.bind_address,
            linger_ms: self.linger_ms,
            query_interval: self.query_interval,
            run_mode: self.run_mode,
        }
    }

    /// Starts a ROUTER server backed by a shared status context.
    ///
    /// # Parameters
    /// - `context`: Shared status context updated on each ping.
    ///
    /// # Returns
    /// - `Result<RouterHandle, String>`: Handle to the running router or an error.
    ///
    /// # Expected Output
    /// - Spawns a background router thread and binds a socket.
    pub fn build_with_shared_context(
        self,
        context: Arc<Mutex<ZmqStatusContext>>,
    ) -> Result<RouterHandle, String> {
        start_router(self.build_config(), move |count| {
            if let Ok(mut guard) = context.lock() {
                guard.set_status(count as u64);
            }
        })
    }

    /// Starts a ROUTER server backed by a local status context.
    ///
    /// # Parameters
    /// - `context`: Local status context moved into the router thread.
    ///
    /// # Returns
    /// - `Result<RouterHandle, String>`: Handle to the running router or an error.
    ///
    /// # Expected Output
    /// - Spawns a background router thread and binds a socket.
    pub fn build_with_context(self, mut context: ZmqStatusContext) -> Result<RouterHandle, String> {
        start_router(self.build_config(), move |count| {
            context.set_status(count as u64);
        })
    }
}

impl Default for PingClientBuilder {
    fn default() -> Self {
        Self {
            connect_address: ZmqConnectAddress::localhost(DEFAULT_CLIENT_PORT),
            linger_ms: DEFAULT_LINGER_MS,
            send_timeout_ms: None,
            recv_timeout_ms: None,
        }
    }
}

impl PingClientBuilder {
    /// Creates a new REQ client builder with localhost defaults.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `PingClientBuilder`: Builder with default localhost configuration.
    ///
    /// # Expected Output
    /// - Returns a builder; no side effects.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the full connect address for the REQ client.
    ///
    /// # Parameters
    /// - `connect_address`: Typed connect address to use.
    ///
    /// # Returns
    /// - `PingClientBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn connect_address(mut self, connect_address: ZmqConnectAddress) -> Self {
        self.connect_address = connect_address;
        self
    }

    /// Sets the target host for the REQ client.
    ///
    /// # Parameters
    /// - `host`: Target host name or IP address.
    ///
    /// # Returns
    /// - `PingClientBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.connect_address.host = host.into();
        self
    }

    /// Sets the target port for the REQ client.
    ///
    /// # Parameters
    /// - `port`: Target TCP port.
    ///
    /// # Returns
    /// - `PingClientBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn port(mut self, port: u16) -> Self {
        self.connect_address.port = port;
        self
    }

    /// Sets the socket linger value in milliseconds.
    ///
    /// # Parameters
    /// - `linger_ms`: Linger timeout in milliseconds.
    ///
    /// # Returns
    /// - `PingClientBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn linger_ms(mut self, linger_ms: i32) -> Self {
        self.linger_ms = linger_ms;
        self
    }

    /// Sets the send timeout in milliseconds.
    ///
    /// # Parameters
    /// - `timeout_ms`: Optional timeout value.
    ///
    /// # Returns
    /// - `PingClientBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn send_timeout_ms(mut self, timeout_ms: Option<i32>) -> Self {
        self.send_timeout_ms = timeout_ms;
        self
    }

    /// Sets the receive timeout in milliseconds.
    ///
    /// # Parameters
    /// - `timeout_ms`: Optional timeout value.
    ///
    /// # Returns
    /// - `PingClientBuilder`: Updated builder.
    ///
    /// # Expected Output
    /// - Returns an updated builder; no side effects.
    pub fn recv_timeout_ms(mut self, timeout_ms: Option<i32>) -> Self {
        self.recv_timeout_ms = timeout_ms;
        self
    }

    /// Builds an immutable client configuration.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `PingClientConfig`: Immutable client configuration.
    ///
    /// # Expected Output
    /// - Returns a config; no side effects.
    pub fn build_config(self) -> PingClientConfig {
        PingClientConfig {
            connect_address: self.connect_address,
            linger_ms: self.linger_ms,
            send_timeout_ms: self.send_timeout_ms,
            recv_timeout_ms: self.recv_timeout_ms,
        }
    }

    /// Builds a reusable REQ client.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Result<PingClient, String>`: Configured client or an error string.
    ///
    /// # Expected Output
    /// - Returns a client value; no socket is opened yet.
    pub fn build(self) -> Result<PingClient, String> {
        let config = self.build_config();
        let endpoint = config.connect_address.to_endpoint();
        Ok(PingClient { config, endpoint })
    }
}

impl ZmqBindAddress {
    /// Creates a bind address for a fixed TCP port.
    ///
    /// # Parameters
    /// - `host`: Host or interface to bind.
    /// - `port`: TCP port to bind.
    ///
    /// # Returns
    /// - `ZmqBindAddress`: Typed bind address.
    ///
    /// # Expected Output
    /// - Returns an address value; no side effects.
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port: Some(port),
        }
    }

    /// Creates a localhost bind address for a fixed TCP port.
    ///
    /// # Parameters
    /// - `port`: TCP port to bind.
    ///
    /// # Returns
    /// - `ZmqBindAddress`: Typed bind address.
    ///
    /// # Expected Output
    /// - Returns an address value; no side effects.
    pub fn localhost(port: u16) -> Self {
        Self::new(DEFAULT_HOST, port)
    }

    /// Creates a localhost bind address that requests an ephemeral port.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `ZmqBindAddress`: Typed bind address using a random port.
    ///
    /// # Expected Output
    /// - Returns an address value; no side effects.
    pub fn localhost_random_port() -> Self {
        Self {
            host: DEFAULT_HOST.to_string(),
            port: None,
        }
    }

    /// Returns the bind endpoint string.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `String`: ZeroMQ bind endpoint.
    ///
    /// # Expected Output
    /// - Returns a formatted endpoint string; no side effects.
    pub fn to_endpoint(&self) -> String {
        let port = self
            .port
            .map(|value| value.to_string())
            .unwrap_or_else(|| "*".to_string());
        format!("{TCP_SCHEME}://{}:{port}", self.host)
    }
}

impl ZmqConnectAddress {
    /// Creates a connect address for a fixed TCP port.
    ///
    /// # Parameters
    /// - `host`: Host name or IP address.
    /// - `port`: Target TCP port.
    ///
    /// # Returns
    /// - `ZmqConnectAddress`: Typed connect address.
    ///
    /// # Expected Output
    /// - Returns an address value; no side effects.
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
        }
    }

    /// Creates a localhost connect address for a fixed TCP port.
    ///
    /// # Parameters
    /// - `port`: Target TCP port.
    ///
    /// # Returns
    /// - `ZmqConnectAddress`: Typed connect address.
    ///
    /// # Expected Output
    /// - Returns an address value; no side effects.
    pub fn localhost(port: u16) -> Self {
        Self::new(DEFAULT_HOST, port)
    }

    /// Returns the connect endpoint string.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `String`: ZeroMQ connect endpoint.
    ///
    /// # Expected Output
    /// - Returns a formatted endpoint string; no side effects.
    pub fn to_endpoint(&self) -> String {
        format!("{TCP_SCHEME}://{}:{}", self.host, self.port)
    }
}

impl RouterCommand {
    /// Parses a router command from the final request frame.
    ///
    /// # Parameters
    /// - `payload`: Request body bytes.
    ///
    /// # Returns
    /// - `RouterCommand`: Parsed command value.
    ///
    /// # Expected Output
    /// - Returns a command value; no side effects.
    pub fn from_payload(payload: &[u8]) -> Self {
        let text = String::from_utf8_lossy(payload);
        match text.as_ref() {
            PING_MESSAGE => Self::Ping,
            STATUS_MESSAGE => Self::Status,
            STOP_MESSAGE => Self::Stop,
            _ => match text.strip_prefix(&format!("{QUERY_RESPONSE_MESSAGE} ")) {
                Some(json) => match serde_json::from_str::<QueryResponsePayload>(json) {
                    Ok(payload) => Self::QueryResponse(payload),
                    Err(_) => Self::Unknown(text.into_owned()),
                },
                None => Self::Unknown(text.into_owned()),
            },
        }
    }

    fn payload_text(&self) -> &str {
        match self {
            Self::Ping => PING_MESSAGE,
            Self::QueryResponse(_) => QUERY_RESPONSE_MESSAGE,
            Self::Status => STATUS_MESSAGE,
            Self::Stop => STOP_MESSAGE,
            Self::Unknown(text) => text.as_str(),
        }
    }

    fn payload_string(&self) -> String {
        match self {
            Self::QueryResponse(payload) => format_query_response_payload(payload),
            _ => self.payload_text().to_string(),
        }
    }
}

impl RouterReply {
    /// Parses a reply from REQ response text.
    ///
    /// # Parameters
    /// - `text`: UTF-8 reply body.
    ///
    /// # Returns
    /// - `RouterReply`: Parsed reply value.
    ///
    /// # Expected Output
    /// - Returns a reply value; no side effects.
    pub fn from_text(text: &str) -> Self {
        match text {
            PONG_MESSAGE => Self::Pong,
            QUERY_MESSAGE => Self::Query,
            STOPPED_MESSAGE => Self::Stopped,
            UNKNOWN_MESSAGE_REPLY => Self::Error(UNKNOWN_MESSAGE_REPLY.to_string()),
            value => match value.parse::<u64>() {
                Ok(status) => Self::Status(status),
                Err(_) => Self::Error(value.to_string()),
            },
        }
    }

    fn payload_text(&self) -> String {
        match self {
            Self::Pong => PONG_MESSAGE.to_string(),
            Self::Query => QUERY_MESSAGE.to_string(),
            Self::Status(value) => value.to_string(),
            Self::Stopped => STOPPED_MESSAGE.to_string(),
            Self::Error(message) => message.clone(),
        }
    }
}

impl fmt::Display for RouterCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.payload_string())
    }
}

impl fmt::Display for RouterReply {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pong => f.write_str(PONG_MESSAGE),
            Self::Query => f.write_str(QUERY_MESSAGE),
            Self::Status(value) => write!(f, "{value}"),
            Self::Stopped => f.write_str(STOPPED_MESSAGE),
            Self::Error(message) => f.write_str(message),
        }
    }
}

impl RouterServerConfig {
    /// Returns the configured bind endpoint string.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `String`: ZeroMQ bind endpoint string.
    ///
    /// # Expected Output
    /// - Returns the endpoint string; no side effects.
    pub fn bind_endpoint(&self) -> String {
        self.bind_address.to_endpoint()
    }
}

impl PingClientConfig {
    /// Returns the configured connect endpoint string.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `String`: ZeroMQ connect endpoint string.
    ///
    /// # Expected Output
    /// - Returns the endpoint string; no side effects.
    pub fn connect_endpoint(&self) -> String {
        self.connect_address.to_endpoint()
    }
}

impl PingClient {
    /// Returns the configured endpoint string for the client.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `&str`: ZeroMQ connect endpoint.
    ///
    /// # Expected Output
    /// - Returns the endpoint string; no side effects.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Sends a command and parses the typed reply.
    ///
    /// # Parameters
    /// - `command`: Router command to send.
    ///
    /// # Returns
    /// - `Result<RouterReply, String>`: Parsed reply or an error string.
    ///
    /// # Expected Output
    /// - Opens a REQ socket, sends one command, and returns one reply.
    pub fn send_command(&self, command: RouterCommand) -> Result<RouterReply, String> {
        send_router_request(&self.config, command)
    }

    /// Sends a ping and validates that the reply is PONG.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Result<(), String>`: `Ok(())` on success or an error string.
    ///
    /// # Expected Output
    /// - Sends PING and receives PONG; no stdout/stderr output.
    pub fn ping(&self) -> Result<(), String> {
        match self.send_command(RouterCommand::Ping)? {
            RouterReply::Pong => Ok(()),
            reply => Err(format!("unexpected router reply: {reply}")),
        }
    }

    /// Queries the current status value.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Result<u64, String>`: Current status value or an error string.
    ///
    /// # Expected Output
    /// - Sends STATUS and receives a numeric reply; no stdout/stderr output.
    pub fn query_status(&self) -> Result<u64, String> {
        match self.send_command(RouterCommand::Status)? {
            RouterReply::Status(value) => Ok(value),
            reply => Err(format!("unexpected router status reply: {reply}")),
        }
    }

    /// Requests the router to stop.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `Result<(), String>`: `Ok(())` on success or an error string.
    ///
    /// # Expected Output
    /// - Sends STOP and receives STOPPED; no stdout/stderr output.
    pub fn stop(&self) -> Result<(), String> {
        match self.send_command(RouterCommand::Stop)? {
            RouterReply::Stopped => Ok(()),
            reply => Err(format!("unexpected router stop reply: {reply}")),
        }
    }
}

/// Starts a ROUTER socket that tracks pings and updates a shared context.
///
/// # Parameters
/// - `expected_pings`: Number of PING messages to wait for before stopping.
/// - `context`: Shared status context.
///
/// # Returns
/// - `Result<RouterHandle, String>`: Handle to the running router or an error.
///
/// # Expected Output
/// - Spawns a background thread and binds a ROUTER socket.
pub fn start_router_with_shared_context(
    expected_pings: usize,
    context: Arc<Mutex<ZmqStatusContext>>,
) -> Result<RouterHandle, String> {
    RouterServerBuilder::new()
        .expected_pings(expected_pings)
        .build_with_shared_context(context)
}

/// Starts a ROUTER socket that tracks pings and updates a local context.
///
/// # Parameters
/// - `expected_pings`: Number of PING messages to wait for before stopping.
/// - `context`: Mutable context to update inside the router thread.
///
/// # Returns
/// - `Result<RouterHandle, String>`: Handle to the running router or an error.
///
/// # Expected Output
/// - Spawns a background thread and binds a ROUTER socket.
pub fn start_router_with_mut_context(
    expected_pings: usize,
    context: ZmqStatusContext,
) -> Result<RouterHandle, String> {
    RouterServerBuilder::new()
        .expected_pings(expected_pings)
        .build_with_context(context)
}

/// Starts a ROUTER socket that runs until it receives a STOP message and updates a shared context.
///
/// # Parameters
/// - `context`: Shared status context.
///
/// # Returns
/// - `Result<RouterHandle, String>`: Handle to the running router or an error.
///
/// # Expected Output
/// - Spawns a background thread and binds a ROUTER socket.
pub fn start_router_until_stopped_with_shared_context(
    context: Arc<Mutex<ZmqStatusContext>>,
) -> Result<RouterHandle, String> {
    RouterServerBuilder::new()
        .until_stopped()
        .build_with_shared_context(context)
}

/// Starts a ROUTER socket that runs until it receives a STOP message and updates a local context.
///
/// # Parameters
/// - `context`: Mutable context to update inside the router thread.
///
/// # Returns
/// - `Result<RouterHandle, String>`: Handle to the running router or an error.
///
/// # Expected Output
/// - Spawns a background thread and binds a ROUTER socket.
pub fn start_router_until_stopped_with_mut_context(
    context: ZmqStatusContext,
) -> Result<RouterHandle, String> {
    RouterServerBuilder::new()
        .until_stopped()
        .build_with_context(context)
}

/// Returns the endpoint bound by the ROUTER socket.
///
/// # Parameters
/// - `handle`: Router handle returned by the builder or helper functions.
///
/// # Returns
/// - `&str`: Endpoint string.
///
/// # Expected Output
/// - Returns the endpoint; no side effects.
pub fn router_endpoint(handle: &RouterHandle) -> &str {
    &handle.endpoint
}

/// Waits for the router thread to finish and returns its stats.
///
/// # Parameters
/// - `handle`: Router handle returned from `start_router_with_*`.
///
/// # Returns
/// - `Result<RouterStats, String>`: Router stats or an error string.
///
/// # Expected Output
/// - Joins the router thread; no stdout/stderr output.
pub fn join_router(handle: RouterHandle) -> Result<RouterStats, String> {
    handle
        .join
        .join()
        .map_err(|_| "router thread panicked".to_string())?
}

/// Queries the current status value from a ROUTER endpoint.
///
/// # Parameters
/// - `endpoint`: ROUTER endpoint string.
///
/// # Returns
/// - `Result<u64, String>`: Current status value or an error string.
///
/// # Expected Output
/// - Sends STATUS and receives a numeric reply; no stdout/stderr output.
pub fn query_router_status(endpoint: &str) -> Result<u64, String> {
    let client = build_client_from_endpoint(endpoint)?;
    client.query_status()
}

/// Requests a running ROUTER endpoint to stop.
///
/// # Parameters
/// - `endpoint`: ROUTER endpoint string.
///
/// # Returns
/// - `Result<(), String>`: `Ok(())` on success or an error string.
///
/// # Expected Output
/// - Sends STOP and receives STOPPED; no stdout/stderr output.
pub fn stop_router(endpoint: &str) -> Result<(), String> {
    let client = build_client_from_endpoint(endpoint)?;
    client.stop()
}

/// Sends a PING request to a ROUTER endpoint using REQ/RESP.
///
/// # Parameters
/// - `endpoint`: ROUTER endpoint string.
///
/// # Returns
/// - `Result<(), String>`: `Ok(())` on success or an error string.
///
/// # Expected Output
/// - Sends PING and receives PONG; no stdout/stderr output.
pub fn ping_router(endpoint: &str) -> Result<(), String> {
    let client = build_client_from_endpoint(endpoint)?;
    client.ping()
}

/// Builds a client from a fully qualified TCP endpoint string.
///
/// # Parameters
/// - `endpoint`: Endpoint string in `tcp://host:port` form.
///
/// # Returns
/// - `Result<PingClient, String>`: Configured client or an error string.
///
/// # Expected Output
/// - Parses the endpoint string and returns a client value.
pub fn build_client_from_endpoint(endpoint: &str) -> Result<PingClient, String> {
    let connect_address = parse_connect_address(endpoint)?;
    PingClientBuilder::new()
        .connect_address(connect_address)
        .build()
}

fn start_router<F>(config: RouterServerConfig, mut on_ping: F) -> Result<RouterHandle, String>
where
    F: FnMut(usize) + Send + 'static,
{
    let (endpoint_tx, endpoint_rx) = mpsc::channel();
    let join = thread::spawn(move || run_router_thread(config, &mut on_ping, endpoint_tx));

    let endpoint = endpoint_rx
        .recv()
        .map_err(|_| "router endpoint channel closed".to_string())?;
    Ok(RouterHandle { endpoint, join })
}

/// Runs the background ROUTER loop using the supplied configuration.
///
/// # Parameters
/// - `config`: Router server configuration.
/// - `on_ping`: Callback invoked after each successful ping.
/// - `endpoint_tx`: Channel used to publish the bound endpoint.
///
/// # Returns
/// - `Result<RouterStats, String>`: Final router statistics or an error string.
///
/// # Expected Output
/// - Binds a ROUTER socket, processes requests, and sends replies.
fn run_router_thread<F>(
    config: RouterServerConfig,
    on_ping: &mut F,
    endpoint_tx: mpsc::Sender<String>,
) -> Result<RouterStats, String>
where
    F: FnMut(usize),
{
    let context = Context::new();
    let socket = context
        .socket(zmq::ROUTER)
        .map_err(|err| format!("router socket error: {err}"))?;
    socket
        .set_linger(config.linger_ms)
        .map_err(|err| format!("router linger error: {err}"))?;
    socket
        .set_rcvtimeo(router_poll_timeout_ms(config.query_interval))
        .map_err(|err| format!("router receive timeout error: {err}"))?;
    socket
        .bind(&config.bind_endpoint())
        .map_err(|err| format!("router bind error: {err}"))?;
    let endpoint = socket
        .get_last_endpoint()
        .map_err(|err| format!("router endpoint error: {err}"))?
        .map_err(|err| format!("router endpoint decode error: {err:?}"))?;
    endpoint_tx
        .send(endpoint)
        .map_err(|_| "router endpoint send failed".to_string())?;

    let mut count = 0usize;
    let mut known_clients = HashMap::<String, KnownClient>::new();
    let mut query_snapshots = HashMap::<String, QueryResponsePayload>::new();
    let mut query_requests_sent = 0usize;
    let mut query_responses = 0usize;
    let mut status_queries = 0usize;
    let mut stop_requested = false;
    let mut next_query_at = Instant::now() + config.query_interval;
    loop {
        let now = Instant::now();
        if now >= next_query_at {
            for client in known_clients.values() {
                send_router_message_to_client(&socket, &client.routing_id, &RouterReply::Query)?;
                query_requests_sent += 1;
            }
            next_query_at = now + config.query_interval;
        }

        let frames = match socket.recv_multipart(0) {
            Ok(frames) => frames,
            Err(zmq::Error::EAGAIN) => continue,
            Err(err) => return Err(format!("router recv error: {err}")),
        };
        if frames.is_empty() {
            continue;
        }

        let routing_id = frames.first().cloned().unwrap_or_default();
        let routing_key = String::from_utf8_lossy(&routing_id).into_owned();
        known_clients.insert(
            routing_key,
            KnownClient {
                routing_id: routing_id.clone(),
            },
        );

        let command = RouterCommand::from_payload(frames.last().unwrap());
        let reply = match command {
            RouterCommand::Ping => {
                count += 1;
                on_ping(count);
                if matches!(config.run_mode, RouterRunMode::ExpectedPings(target) if count >= target)
                {
                    send_router_reply(&socket, &frames, &RouterReply::Pong)?;
                    break;
                }
                Some(RouterReply::Pong)
            }
            RouterCommand::QueryResponse(payload) => {
                query_responses += 1;
                query_snapshots.insert(payload.client_id.clone(), payload);
                print_query_aggregation(&query_snapshots);
                None
            }
            RouterCommand::Status => {
                status_queries += 1;
                Some(RouterReply::Status(GLOBAL_STATUS.load(Ordering::Relaxed)))
            }
            RouterCommand::Stop => {
                stop_requested = true;
                send_router_reply(&socket, &frames, &RouterReply::Stopped)?;
                break;
            }
            RouterCommand::Unknown(_) => {
                Some(RouterReply::Error(UNKNOWN_MESSAGE_REPLY.to_string()))
            }
        };
        if let Some(reply) = reply {
            send_router_reply(&socket, &frames, &reply)?;
        }
    }

    Ok(RouterStats {
        pings: count,
        query_requests_sent,
        query_responses,
        status_queries,
        final_status: GLOBAL_STATUS.load(Ordering::Relaxed),
        known_clients: known_clients.len(),
        stop_requested,
    })
}

/// Sends a reply from a ROUTER socket using the incoming identity frames.
///
/// # Parameters
/// - `socket`: ROUTER socket to send from.
/// - `frames`: Incoming multipart frames containing the envelope and body.
/// - `reply`: Typed reply payload to send.
///
/// # Returns
/// - `Result<(), String>`: `Ok(())` on success or an error string.
///
/// # Expected Output
/// - Sends a multipart reply; no stdout/stderr output.
fn send_router_reply(
    socket: &Socket,
    frames: &[Vec<u8>],
    reply: &RouterReply,
) -> Result<(), String> {
    if frames.is_empty() {
        return Err("router reply requires at least one frame".to_string());
    }
    for frame in &frames[..frames.len() - 1] {
        socket
            .send(frame, zmq::SNDMORE)
            .map_err(|err| format!("router send envelope error: {err}"))?;
    }
    let payload = reply.payload_text();
    socket
        .send(payload.as_bytes(), 0)
        .map_err(|err| format!("router send payload error: {err}"))?;
    Ok(())
}

/// Sends a server-initiated message to a connected DEALER client.
///
/// # Parameters
/// - `socket`: ROUTER socket to send from.
/// - `routing_id`: Client routing identity frame.
/// - `reply`: Typed reply payload to send.
///
/// # Returns
/// - `Result<(), String>`: `Ok(())` on success or an error string.
///
/// # Expected Output
/// - Sends a multipart message to one client; no stdout/stderr output.
fn send_router_message_to_client(
    socket: &Socket,
    routing_id: &[u8],
    reply: &RouterReply,
) -> Result<(), String> {
    socket
        .send(routing_id, zmq::SNDMORE)
        .map_err(|err| format!("router send identity error: {err}"))?;
    let payload = reply.payload_text();
    socket
        .send(payload.as_bytes(), 0)
        .map_err(|err| format!("router send payload error: {err}"))?;
    Ok(())
}

/// Sends a single REQ command using the configured client settings.
///
/// # Parameters
/// - `config`: REQ client configuration.
/// - `command`: Command to send.
///
/// # Returns
/// - `Result<RouterReply, String>`: Parsed reply or an error string.
///
/// # Expected Output
/// - Opens a REQ socket, sends one command, and receives one reply.
fn send_router_request(
    config: &PingClientConfig,
    command: RouterCommand,
) -> Result<RouterReply, String> {
    let context = Context::new();
    let socket = context
        .socket(zmq::REQ)
        .map_err(|err| format!("req socket error: {err}"))?;
    socket
        .set_linger(config.linger_ms)
        .map_err(|err| format!("req linger error: {err}"))?;
    if let Some(timeout) = config.send_timeout_ms {
        socket
            .set_sndtimeo(timeout)
            .map_err(|err| format!("req send timeout error: {err}"))?;
    }
    if let Some(timeout) = config.recv_timeout_ms {
        socket
            .set_rcvtimeo(timeout)
            .map_err(|err| format!("req recv timeout error: {err}"))?;
    }
    socket
        .connect(&config.connect_endpoint())
        .map_err(|err| format!("req connect error: {err}"))?;
    socket
        .send(command.payload_string().as_bytes(), 0)
        .map_err(|err| format!("req send error: {err}"))?;
    let reply = socket
        .recv_msg(0)
        .map_err(|err| format!("req recv error: {err}"))?;
    let reply_text = reply
        .as_str()
        .ok_or_else(|| "router reply was not valid UTF-8".to_string())?;
    Ok(RouterReply::from_text(reply_text))
}

/// Parses a TCP connect endpoint into a typed address.
///
/// # Parameters
/// - `endpoint`: Endpoint string in `tcp://host:port` form.
///
/// # Returns
/// - `Result<ZmqConnectAddress, String>`: Parsed typed connect address.
///
/// # Expected Output
/// - Returns a parsed address value; no side effects.
fn parse_connect_address(endpoint: &str) -> Result<ZmqConnectAddress, String> {
    let remainder = endpoint
        .strip_prefix("tcp://")
        .ok_or_else(|| "only tcp:// endpoints are supported".to_string())?;
    let (host, port_text) = remainder
        .rsplit_once(':')
        .ok_or_else(|| "endpoint must include host and port".to_string())?;
    if host.is_empty() {
        return Err("endpoint host cannot be empty".to_string());
    }
    let port = port_text
        .parse::<u16>()
        .map_err(|err| format!("invalid endpoint port: {err}"))?;
    Ok(ZmqConnectAddress::new(host, port))
}

fn format_query_response_payload(payload: &QueryResponsePayload) -> String {
    let json = serde_json::to_string(payload).unwrap_or_else(|_| {
        "{\"client_id\":\"serialization-error\",\"cpu_cores\":0,\"available_memory_bytes\":0}"
            .to_string()
    });
    format!("{QUERY_RESPONSE_MESSAGE} {json}")
}

fn router_poll_timeout_ms(query_interval: Duration) -> i32 {
    let timeout_ms = query_interval.as_millis().clamp(1, 250) as i32;
    timeout_ms
}

fn print_query_aggregation(query_snapshots: &HashMap<String, QueryResponsePayload>) {
    let mut snapshots = query_snapshots.values().cloned().collect::<Vec<_>>();
    snapshots.sort_by(|left, right| left.client_id.cmp(&right.client_id));

    let total_cpu_cores = snapshots
        .iter()
        .map(|snapshot| snapshot.cpu_cores)
        .sum::<usize>();
    let total_available_memory = snapshots
        .iter()
        .map(|snapshot| snapshot.available_memory_bytes)
        .sum::<u64>();

    println!("Poller resource summary ({} clients):", snapshots.len());
    for snapshot in snapshots {
        println!(
            "  {}: cpu_cores={}, available_memory_bytes={}",
            snapshot.client_id, snapshot.cpu_cores, snapshot.available_memory_bytes
        );
    }
    println!(
        "  totals: cpu_cores={}, available_memory_bytes={}",
        total_cpu_cores, total_available_memory
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;
    use std::time::Duration;

    fn zmq_tests_enabled() -> bool {
        matches!(
            std::env::var("RSADEMO_ZMQ_TESTS").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    }

    const CLIENT_THREADS: usize = 4;
    const EXPECTED_PINGS: usize = CLIENT_THREADS + 1;

    fn acquire_zmq_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static ZMQ_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ZMQ_TEST_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("zmq test lock")
    }

    fn run_ping_clients(endpoint: &str) {
        let client = build_client_from_endpoint(endpoint).expect("client");
        let mut handles = Vec::with_capacity(CLIENT_THREADS);
        for _ in 0..CLIENT_THREADS {
            let client = client.clone();
            handles.push(thread::spawn(move || {
                client.ping().expect("ping failed");
            }));
        }

        client.ping().expect("main ping failed");

        for handle in handles {
            handle.join().expect("thread join");
        }
    }

    #[test]
    fn test_bind_address_formats_fixed_and_random_ports() {
        assert_eq!(
            ZmqBindAddress::localhost(5555).to_endpoint(),
            "tcp://127.0.0.1:5555"
        );
        assert_eq!(
            ZmqBindAddress::localhost_random_port().to_endpoint(),
            "tcp://127.0.0.1:*"
        );
    }

    #[test]
    fn test_parse_connect_address() {
        let address = parse_connect_address("tcp://127.0.0.1:7777").expect("parse");
        assert_eq!(address, ZmqConnectAddress::localhost(7777));
    }

    #[test]
    fn test_command_and_reply_parsing() {
        assert_eq!(RouterCommand::from_payload(b"PING"), RouterCommand::Ping);
        assert_eq!(
            RouterCommand::from_payload(b"STATUS"),
            RouterCommand::Status
        );
        assert_eq!(RouterReply::from_text("QUERY"), RouterReply::Query);
        assert_eq!(RouterReply::from_text("PONG"), RouterReply::Pong);
        assert_eq!(RouterReply::from_text("42"), RouterReply::Status(42));
        assert_eq!(
            RouterReply::from_text("ERROR"),
            RouterReply::Error("ERROR".to_string())
        );
        assert_eq!(
            RouterCommand::from_payload(
                br#"QUERY_RESPONSE {"client_id":"poller-1","cpu_cores":8,"available_memory_bytes":1024}"#
            ),
            RouterCommand::QueryResponse(QueryResponsePayload {
                client_id: "poller-1".to_string(),
                cpu_cores: 8,
                available_memory_bytes: 1024,
            })
        );
    }

    #[test]
    fn test_router_with_shared_context() {
        if !zmq_tests_enabled() {
            eprintln!("Skipping ZeroMQ tests; set RSADEMO_ZMQ_TESTS=1 to enable.");
            return;
        }
        let _guard = acquire_zmq_test_lock();
        ZmqStatusContext::reset_global_status();
        let context = Arc::new(Mutex::new(ZmqStatusContext::new()));
        let handle = RouterServerBuilder::new()
            .expected_pings(EXPECTED_PINGS)
            .build_with_shared_context(Arc::clone(&context))
            .expect("router");
        let endpoint = router_endpoint(&handle).to_string();

        run_ping_clients(&endpoint);

        let stats = join_router(handle).expect("join");
        assert_eq!(stats.pings, EXPECTED_PINGS);
        assert_eq!(stats.status_queries, 0);
        assert_eq!(ZmqStatusContext::global_status(), EXPECTED_PINGS as u64);
        let stored = context.lock().expect("lock").status();
        assert_eq!(stored, EXPECTED_PINGS as u64);
        assert_eq!(stats.final_status, EXPECTED_PINGS as u64);
        assert!(!stats.stop_requested);
    }

    #[test]
    fn test_router_with_mut_context() {
        if !zmq_tests_enabled() {
            eprintln!("Skipping ZeroMQ tests; set RSADEMO_ZMQ_TESTS=1 to enable.");
            return;
        }
        let _guard = acquire_zmq_test_lock();
        ZmqStatusContext::reset_global_status();
        let handle = RouterServerBuilder::new()
            .expected_pings(EXPECTED_PINGS)
            .build_with_context(ZmqStatusContext::new())
            .expect("router");
        let endpoint = router_endpoint(&handle).to_string();

        run_ping_clients(&endpoint);

        let stats = join_router(handle).expect("join");
        assert_eq!(stats.pings, EXPECTED_PINGS);
        assert_eq!(stats.status_queries, 0);
        assert_eq!(stats.final_status, EXPECTED_PINGS as u64);
        assert_eq!(ZmqStatusContext::global_status(), EXPECTED_PINGS as u64);
        assert!(!stats.stop_requested);
        std::thread::sleep(Duration::from_millis(10));
    }

    #[test]
    fn test_router_status_queries_and_stop_request() {
        if !zmq_tests_enabled() {
            eprintln!("Skipping ZeroMQ tests; set RSADEMO_ZMQ_TESTS=1 to enable.");
            return;
        }
        let _guard = acquire_zmq_test_lock();
        ZmqStatusContext::reset_global_status();
        let context = Arc::new(Mutex::new(ZmqStatusContext::new()));
        let handle = RouterServerBuilder::new()
            .until_stopped()
            .build_with_shared_context(Arc::clone(&context))
            .expect("router");
        let endpoint = router_endpoint(&handle).to_string();
        let client = build_client_from_endpoint(&endpoint).expect("client");

        assert_eq!(client.query_status().expect("initial status"), 0);
        assert_eq!(
            client
                .send_command(RouterCommand::Ping)
                .expect("first ping"),
            RouterReply::Pong
        );
        assert_eq!(
            client
                .send_command(RouterCommand::Ping)
                .expect("second ping"),
            RouterReply::Pong
        );
        assert_eq!(client.query_status().expect("updated status"), 2);
        client.stop().expect("stop request");

        let stats = join_router(handle).expect("join");
        assert_eq!(stats.pings, 2);
        assert_eq!(stats.status_queries, 2);
        assert_eq!(stats.final_status, 2);
        assert!(stats.stop_requested);
        assert_eq!(context.lock().expect("lock").status(), 2);
    }
}
