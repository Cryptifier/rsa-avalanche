# ZMQ Status Protocol

This project exposes a small ZeroMQ status protocol in `src/zmq_status.rs`.

## Interfaces

The ZMQ layer uses typed interfaces and builders instead of raw string-only setup:

- `ZmqBindAddress`: typed TCP bind address for ROUTER sockets
- `ZmqConnectAddress`: typed TCP connect address for REQ sockets
- `RouterRunMode`: router lifecycle mode (`UntilStopped` or `ExpectedPings`)
- `RouterServerBuilder`: builder for starting ROUTER servers
- `PingClientBuilder`: builder for creating REQ clients
- `RouterCommand`: typed request enum
- `RouterReply`: typed response enum

## Transport

- Server socket type: `ROUTER`
- Helper client socket type: `REQ`
- Long-lived poller socket type: `DEALER`
- Default builder bind template: `tcp://127.0.0.1:*`
- Default client target: `tcp://127.0.0.1:5555`
- Replies reuse the full incoming envelope and replace only the final body frame.

## Commands

All request bodies are UTF-8 strings carried in the final message frame.

### `PING`

- Reply: `PONG`
- Effect: increments the router ping count
- Effect: updates the status context to the current ping count
- Effect: mirrors the updated value into the global atomic status

### `QUERY`

- Direction: router to connected poller
- Reply: poller sends `QUERY_RESPONSE <json>`
- Effect: requests a per-client resource snapshot from the target poller

### `QUERY_RESPONSE`

- Direction: poller to router
- Body: JSON object with `client_id`, `cpu_cores`, and `available_memory_bytes`
- Effect: updates the router's in-memory resource aggregation for that client
- Effect: causes the server to print the aggregated snapshot to stdout

### `STATUS`

- Reply: current status as a base-10 `u64` string
- Effect: does not increment the ping count
- Effect: reports the current global/status-context value

### `STOP`

- Reply: `STOPPED`
- Effect: stops the router loop after replying

### Unknown command

- Reply: `ERROR`
- Effect: router stays alive

## Router Modes

Two lifecycle modes are supported:

- Bounded mode: `start_router_with_shared_context` and `start_router_with_mut_context`
  - The router exits automatically after `expected_pings` successful `PING` requests.
- Unbounded mode: `start_router_until_stopped_with_shared_context` and `start_router_until_stopped_with_mut_context`
  - The router runs until it receives `STOP`.
- In both modes, the router sends `QUERY` to each known client at the configured interval.

## Status Semantics

- `ZmqStatusContext::set_status(value)` stores `value` locally and in the process-wide global atomic mirror.
- `PING` sets the status to the cumulative ping count seen by that router instance.
- `STATUS` returns the current mirrored value.
- `QUERY_RESPONSE` stores the latest resource snapshot for the reported `client_id`.
- `ZmqStatusContext::reset_global_status()` clears the global mirror to `0`.

## Public Client Helpers

- `ping_router(endpoint)` sends `PING` and validates `PONG`
- `query_router_status(endpoint)` sends `STATUS` and parses the numeric reply
- `stop_router(endpoint)` sends `STOP` and validates `STOPPED`
- `build_client_from_endpoint(endpoint)` parses a TCP endpoint into a typed client
- `join_router(handle)` waits for the background router thread and returns `RouterStats`

## Builder Usage

Typical ROUTER builder usage:

- `RouterServerBuilder::new()`
- `.bind_address(ZmqBindAddress::new(host, port))`
- `.linger_ms(linger_ms)`
- `.until_stopped()` or `.expected_pings(count)`
- `.build_with_shared_context(context)` or `.build_with_context(context)`

Typical REQ client builder usage:

- `PingClientBuilder::new()`
- `.connect_address(ZmqConnectAddress::new(host, port))`
- `.send_timeout_ms(Some(timeout_ms))`
- `.recv_timeout_ms(Some(timeout_ms))`
- `.build()`

## Binaries

- `src/bin/server.rs`
  - Starts the existing HTTP viewer server.
  - Starts a concurrent ZMQ ROUTER server in the background.
  - Exposes CLI options for ZMQ host, port, linger, and optional expected ping count.
- `src/bin/poller.rs`
  - Creates a long-lived ZMQ DEALER client from CLI options.
  - Sends the configured `PING` sequence to the configured host/port.
  - Remains active until `Ctrl-C`, answering server-initiated `QUERY` messages.
  - Can optionally query `STATUS` after each ping and send `STOP` at the end.

## RouterStats

`RouterStats` reports:

- `pings`: number of processed `PING` commands
- `query_requests_sent`: number of `QUERY` messages sent by the router
- `query_responses`: number of processed `QUERY_RESPONSE` messages
- `status_queries`: number of processed `STATUS` commands
- `final_status`: last mirrored status observed before exit
- `known_clients`: number of distinct client identities seen by the router
- `stop_requested`: `true` when shutdown happened via `STOP`
