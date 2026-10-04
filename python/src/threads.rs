//! GIL release and rayon thread-count control for heavy calls.

use pyo3::prelude::*;
use rayon::ThreadPool;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Process default thread count; 0 = rayon's global pool (all cores, or
/// `RAYON_NUM_THREADS`).
static DEFAULT_THREADS: AtomicUsize = AtomicUsize::new(usize::MAX);

fn pools() -> &'static Mutex<HashMap<usize, Arc<ThreadPool>>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();
    POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn default_threads() -> usize {
    let v = DEFAULT_THREADS.load(Ordering::Relaxed);
    if v != usize::MAX {
        return v;
    }
    let env = std::env::var("QSIMLAB_NUM_THREADS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(0);
    DEFAULT_THREADS.store(env, Ordering::Relaxed);
    env
}

fn pool(k: usize) -> Arc<ThreadPool> {
    let mut map = pools().lock().unwrap_or_else(|p| p.into_inner());
    map.entry(k)
        .or_insert_with(|| {
            Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(k)
                    .thread_name(move |i| format!("qsimlab-{k}-{i}"))
                    .build()
                    .expect("failed to build a rayon thread pool"),
            )
        })
        .clone()
}

/// Runs `f` with the GIL released, on a `threads`-thread rayon pool
/// (`None`: the process default, see [`set_num_threads`]).
///
/// Every heavy binding goes through this. `f` must not touch Python
/// objects: extract everything you need before calling it.
pub fn heavy<T, F>(py: Python<'_>, threads: Option<usize>, f: F) -> T
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    let k = threads.unwrap_or_else(default_threads);
    py.detach(move || if k == 0 { f() } else { pool(k).install(f) })
}

/// Sets the default number of worker threads for heavy calls. ``0`` means
/// "all cores" (rayon's global pool, honouring ``RAYON_NUM_THREADS``).
#[pyfunction]
pub fn set_num_threads(n: usize) {
    DEFAULT_THREADS.store(n, Ordering::Relaxed);
}

/// The number of worker threads heavy calls use by default.
#[pyfunction]
pub fn get_num_threads() -> usize {
    match default_threads() {
        0 => rayon::current_num_threads(),
        k => k,
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    Ok(())
}
