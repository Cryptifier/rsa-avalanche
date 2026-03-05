use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;

use zmq::{Context, Socket};

static GLOBAL_STATUS: AtomicU64 = AtomicU64::new(0);

const DEFAULT_ENDPOINT: &str = "tcp://127.0.0.1:*";
const PING_MESSAGE: &str = "PING";
const PONG_MESSAGE: &str = "PONG";

/// Shared context for reporting r-candidate generation status.
#[derive(Debug, Clone, Default)]
pub struct ZmqStatusContext {
    status: u64,
}

/// Summary statistics for a router run.
#[derive(Debug, Clone)]
pub struct RouterStats {
    /// Number of PING messages processed.
    pub pings: usize,
    /// Final status value in the context.
    pub final_status: u64,
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
    start_router(expected_pings, move |count| {
        if let Ok(mut guard) = context.lock() {
            guard.set_status(count as u64);
        }
    })
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
    mut context: ZmqStatusContext,
) -> Result<RouterHandle, String> {
    start_router(expected_pings, move |count| {
        context.set_status(count as u64);
    })
}

/// Returns the endpoint bound by the ROUTER socket.
///
/// # Parameters
/// - None.
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
    handle.join.join().map_err(|_| "router thread panicked".to_string())?
}

fn start_router<F>(expected_pings: usize, mut on_ping: F) -> Result<RouterHandle, String>
where
    F: FnMut(usize) + Send + 'static,
{
    let (endpoint_tx, endpoint_rx) = mpsc::channel();
    let join = thread::spawn(move || run_router_thread(expected_pings, &mut on_ping, endpoint_tx));

    let endpoint = endpoint_rx
        .recv()
        .map_err(|_| "router endpoint channel closed".to_string())?;
    Ok(RouterHandle { endpoint, join })
}

fn run_router_thread<F>(
    expected_pings: usize,
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
        .set_linger(0)
        .map_err(|err| format!("router linger error: {err}"))?;
    socket
        .bind(DEFAULT_ENDPOINT)
        .map_err(|err| format!("router bind error: {err}"))?;
    let endpoint = socket
        .get_last_endpoint()
        .map_err(|err| format!("router endpoint error: {err}"))?
        .map_err(|err| format!("router endpoint decode error: {err:?}"))?;
    endpoint_tx
        .send(endpoint)
        .map_err(|_| "router endpoint send failed".to_string())?;

    let mut count = 0usize;
    let mut final_status = 0u64;
    while count < expected_pings {
        let frames = socket
            .recv_multipart(0)
            .map_err(|err| format!("router recv error: {err}"))?;
        if frames.len() < 3 {
            continue;
        }
        let body = frames.last().unwrap();
        if body == PING_MESSAGE.as_bytes() {
            count += 1;
            on_ping(count);
            final_status = GLOBAL_STATUS.load(Ordering::Relaxed);
            send_router_reply(&socket, &frames, PONG_MESSAGE.as_bytes())?;
        }
    }

    Ok(RouterStats {
        pings: count,
        final_status,
    })
}

/// Sends a reply from a ROUTER socket using the incoming identity frames.
///
/// # Parameters
/// - `socket`: ROUTER socket to send from.
/// - `frames`: Incoming multipart frames containing identity + delimiter + body.
/// - `payload`: Response payload.
///
/// # Returns
/// - `Result<(), String>`: `Ok(())` on success or an error string.
///
/// # Expected Output
/// - Sends a multipart reply; no stdout/stderr output.
fn send_router_reply(socket: &Socket, frames: &[Vec<u8>], payload: &[u8]) -> Result<(), String> {
    socket
        .send(&frames[0], zmq::SNDMORE)
        .map_err(|err| format!("router send identity error: {err}"))?;
    socket
        .send("", zmq::SNDMORE)
        .map_err(|err| format!("router send delimiter error: {err}"))?;
    socket
        .send(payload, 0)
        .map_err(|err| format!("router send payload error: {err}"))?;
    Ok(())
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
    let context = Context::new();
    let socket = context
        .socket(zmq::REQ)
        .map_err(|err| format!("req socket error: {err}"))?;
    socket
        .set_linger(0)
        .map_err(|err| format!("req linger error: {err}"))?;
    socket
        .connect(endpoint)
        .map_err(|err| format!("req connect error: {err}"))?;
    socket
        .send(PING_MESSAGE, 0)
        .map_err(|err| format!("req send error: {err}"))?;
    let reply = socket
        .recv_msg(0)
        .map_err(|err| format!("req recv error: {err}"))?;
    if reply.as_str() != Some(PONG_MESSAGE) {
        return Err("unexpected router reply".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const CLIENT_THREADS: usize = 4;
    const EXPECTED_PINGS: usize = CLIENT_THREADS + 1;

    fn run_ping_clients(endpoint: &str) {
        let mut handles = Vec::with_capacity(CLIENT_THREADS);
        for _ in 0..CLIENT_THREADS {
            let endpoint = endpoint.to_string();
            handles.push(thread::spawn(move || {
                ping_router(&endpoint).expect("ping failed");
            }));
        }

        ping_router(endpoint).expect("main ping failed");

        for handle in handles {
            handle.join().expect("thread join");
        }
    }

    #[test]
    fn test_router_with_shared_context() {
        GLOBAL_STATUS.store(0, Ordering::Relaxed);
        let context = Arc::new(Mutex::new(ZmqStatusContext::new()));
        let handle =
            start_router_with_shared_context(EXPECTED_PINGS, Arc::clone(&context)).expect("router");
        let endpoint = router_endpoint(&handle).to_string();

        run_ping_clients(&endpoint);

        let stats = join_router(handle).expect("join");
        assert_eq!(stats.pings, EXPECTED_PINGS);
        assert_eq!(ZmqStatusContext::global_status(), EXPECTED_PINGS as u64);
        let stored = context.lock().expect("lock").status();
        assert_eq!(stored, EXPECTED_PINGS as u64);
        assert_eq!(stats.final_status, EXPECTED_PINGS as u64);
    }

    #[test]
    fn test_router_with_mut_context() {
        GLOBAL_STATUS.store(0, Ordering::Relaxed);
        let context = ZmqStatusContext::new();
        let handle = start_router_with_mut_context(EXPECTED_PINGS, context).expect("router");
        let endpoint = router_endpoint(&handle).to_string();

        run_ping_clients(&endpoint);

        let stats = join_router(handle).expect("join");
        assert_eq!(stats.pings, EXPECTED_PINGS);
        assert_eq!(stats.final_status, EXPECTED_PINGS as u64);
        assert_eq!(ZmqStatusContext::global_status(), EXPECTED_PINGS as u64);
        std::thread::sleep(Duration::from_millis(10));
    }
}
