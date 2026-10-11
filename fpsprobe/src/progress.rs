//! Progress reporting and cancellation shared between a running job and the UI.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Default)]
pub struct Progress {
    stage: Mutex<String>,
    done: AtomicUsize,
    total: AtomicUsize,
    cancel: AtomicBool,
}

impl Progress {
    /// Starts a new stage; `total` of 0 means unknown.
    pub fn stage(&self, name: &str, total: usize) {
        *self.stage.lock().unwrap() = name.to_string();
        self.done.store(0, Ordering::Relaxed);
        self.total.store(total, Ordering::Relaxed);
    }

    pub fn advance(&self) {
        self.done.fetch_add(1, Ordering::Relaxed);
    }

    /// (stage name, done, total)
    pub fn snapshot(&self) -> (String, usize, usize) {
        (self.stage.lock().unwrap().clone(), self.done.load(Ordering::Relaxed), self.total.load(Ordering::Relaxed))
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}
