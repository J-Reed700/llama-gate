//! The handshake that makes pausing a request safe.
//!
//! A dispatched request may be paused only until its output starts
//! reaching the client: before that a restart is invisible, after it the
//! client would see a seam. The request's handler and the engine race for
//! that moment, and [`Control`] settles the race atomically: the handler
//! claims it with [`Control::begin`] before relaying the first output, the
//! engine with [`Control::pause`]. Exactly one of them wins.
use crate::runtime::Grant;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};
use tokio::sync::{Notify, oneshot};

const WAITING: u8 = 0;
const RUNNING: u8 = 1;
const STARTED: u8 = 2;
const PAUSED: u8 = 3;

#[derive(Debug, Default)]
pub struct Control {
    state: AtomicU8,
    notify: Notify,
    /// Where the next grant arrives after a pause.
    regrant: Mutex<Option<oneshot::Receiver<Grant>>>,
}

impl Control {
    /// The request was dispatched and has shown nothing yet.
    pub fn dispatched(&self) {
        // A request that started before keeps its claim.
        let _ = self
            .state
            .compare_exchange(WAITING, RUNNING, Ordering::AcqRel, Ordering::Acquire);
        let _ = self
            .state
            .compare_exchange(PAUSED, RUNNING, Ordering::AcqRel, Ordering::Acquire);
    }

    /// Claims the moment output starts reaching the client. False if the
    /// request was paused instead: the handler must drop its upstream and
    /// wait for the next grant.
    pub fn begin(&self) -> bool {
        match self
            .state
            .compare_exchange(RUNNING, STARTED, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => true,
            Err(current) => current == STARTED,
        }
    }

    pub fn started(&self) -> bool {
        self.state.load(Ordering::Acquire) == STARTED
    }

    /// Pauses a request whose output has not begun, handing its handler the
    /// receiver of its next grant. False if the output has begun.
    pub fn pause(&self, regrant: oneshot::Receiver<Grant>) -> bool {
        if self
            .state
            .compare_exchange(RUNNING, PAUSED, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        if let Ok(mut slot) = self.regrant.lock() {
            *slot = Some(regrant);
        }
        self.notify.notify_one();
        true
    }

    /// Resolves once the request is paused.
    pub async fn paused(&self) {
        loop {
            let notified = self.notify.notified();
            if self.state.load(Ordering::Acquire) == PAUSED {
                return;
            }
            notified.await;
        }
    }

    /// The receiver of the grant that follows a pause.
    pub fn take_regrant(&self) -> Option<oneshot::Receiver<Grant>> {
        self.regrant.lock().ok()?.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exactly_one_side_wins_the_race() {
        let control = Control::default();
        control.dispatched();
        let (_, receiver) = oneshot::channel();
        assert!(control.pause(receiver));
        assert!(!control.begin(), "paused first: the handler must not relay");
        assert!(control.take_regrant().is_some());

        let control = Control::default();
        control.dispatched();
        assert!(control.begin());
        let (_, receiver) = oneshot::channel();
        assert!(
            !control.pause(receiver),
            "started first: the pause is declined"
        );
        assert!(control.begin(), "beginning again is harmless");
    }

    #[tokio::test]
    async fn a_pause_wakes_the_handler_even_if_it_was_not_yet_listening() {
        let control = std::sync::Arc::new(Control::default());
        control.dispatched();
        let (_, receiver) = oneshot::channel();
        assert!(control.pause(receiver));
        tokio::time::timeout(std::time::Duration::from_secs(1), control.paused())
            .await
            .unwrap();
        // Dispatched again after the pause, it can begin.
        control.dispatched();
        assert!(control.begin());
    }
}
