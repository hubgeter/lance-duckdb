use std::future::Future;
use std::io;
use std::process;
use std::sync::OnceLock;

use tokio::runtime::{Builder, Handle, Runtime};

struct ProcessRuntime {
    pid: u32,
    runtime: Runtime,
}

static RUNTIME: OnceLock<Result<ProcessRuntime, io::Error>> = OnceLock::new();

fn configured_worker_threads() -> usize {
    // VANE_LANCE_WORKER_CPUS is an explicit per-process override.  Ray normally
    // supplies OMP_NUM_THREADS from the worker's admitted CPU capacity, so use
    // that as the fallback before consulting the host's available parallelism.
    // Refuse zero and malformed values because Tokio requires a positive count.
    for name in ["VANE_LANCE_WORKER_CPUS", "OMP_NUM_THREADS"] {
        if let Some(value) = std::env::var_os(name) {
            if let Ok(value) = value.to_string_lossy().parse::<usize>() {
                if value > 0 {
                    return value;
                }
            }
        }
    }
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .max(1)
}

fn build_runtime() -> Result<ProcessRuntime, io::Error> {
    let runtime = Builder::new_multi_thread()
        .worker_threads(configured_worker_threads())
        .thread_name("vane-lance")
        .enable_all()
        .build()?;
    Ok(ProcessRuntime {
        pid: process::id(),
        runtime,
    })
}

pub fn runtime() -> Result<&'static Runtime, io::Error> {
    match RUNTIME.get_or_init(build_runtime) {
        Ok(process_runtime) if process_runtime.pid == process::id() => Ok(&process_runtime.runtime),
        Ok(process_runtime) => Err(io::Error::other(format!(
            "the Lance Tokio runtime was initialized in process {} before fork and cannot be used in child process {}; use a spawn/exec process start method",
            process_runtime.pid,
            process::id()
        ))),
        Err(err) => Err(io::Error::new(err.kind(), err.to_string())),
    }
}

pub fn initialized_runtime() -> Option<&'static Runtime> {
    let process_runtime = RUNTIME.get()?.as_ref().ok()?;
    (process_runtime.pid == process::id()).then_some(&process_runtime.runtime)
}

pub fn handle() -> Result<Handle, io::Error> {
    Ok(runtime()?.handle().clone())
}

pub fn block_on<F: Future>(future: F) -> Result<F::Output, io::Error> {
    Ok(runtime()?.block_on(future))
}
