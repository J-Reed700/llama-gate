//! The handshake that makes pausing a request safe.
//!
//! A dispatched request may be paused only until its output starts
//! reaching the client: before that a restart is invisible, after it the
//! client would see a seam. The request's handler and the engine race for
//! that moment, and [`Control`] settles the race: the handler claims it with
//! [`Control::begin`] before relaying the first output, the engine with
//! [`Control::pause`]. Exactly one of them wins.
//!
//! The phase and the receiver of the grant that follows a pause change
//! together, under one lock: a handler that sees the request paused always
//! finds where its next grant arrives.
use crate::engine::Grant;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use tokio::sync::{Notify, oneshot};

#[derive(Debug, Default)]
pub struct Control {
    inner: Mutex<Inner>,
    /// Mirrors `Phase::Started`, which never changes back, for the relay's
    /// check on every chunk.
    started: AtomicBool,
    notify: Notify,
}

#[derive(Debug, Default)]
struct Inner {
    phase: Phase,
    /// Where the next grant arrives after a pause, until the handler takes
    /// it (it may be dispatched again before it does).
    regrant: Option<oneshot::Receiver<Grant>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Waiting,
    /// Dispatched, nothing shown yet.
    Running,
    /// Output has reached the client.
    Started,
    Paused,
}

impl Control {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        // Every change leaves the state whole, so a panic elsewhere cannot
        // leave it half-written.
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The request was dispatched (again, after a pause) and has shown
    /// nothing yet. A request that started keeps its claim.
    pub fn dispatched(&self) {
        let mut inner = self.lock();
        if matches!(inner.phase, Phase::Waiting | Phase::Paused) {
            inner.phase = Phase::Running;
        }
    }

    /// Claims the moment output starts reaching the client. False if the
    /// request was paused instead: the handler must drop its upstream and
    /// wait for the next grant.
    pub fn begin(&self) -> bool {
        let mut inner = self.lock();
        match inner.phase {
            Phase::Running => {
                inner.phase = Phase::Started;
                self.started.store(true, Ordering::Release);
                true
            }
            Phase::Started => true,
            Phase::Waiting | Phase::Paused => false,
        }
    }

    pub fn started(&self) -> bool {
        self.started.load(Ordering::Acquire)
    }

    /// Pauses a request whose output has not begun, handing its handler the
    /// receiver of its next grant. False if the output has begun.
    pub fn pause(&self, regrant: oneshot::Receiver<Grant>) -> bool {
        {
            let mut inner = self.lock();
            if inner.phase != Phase::Running {
                return false;
            }
            inner.phase = Phase::Paused;
            inner.regrant = Some(regrant);
        }
        // One handler waits; if it is not listening yet, the permit waits.
        self.notify.notify_one();
        true
    }

    /// Resolves once the request is paused.
    pub async fn paused(&self) {
        loop {
            let notified = self.notify.notified();
            if self.lock().phase == Phase::Paused {
                return;
            }
            notified.await;
        }
    }

    /// The receiver of the grant that follows a pause.
    pub fn take_regrant(&self) -> Option<oneshot::Receiver<Grant>> {
        self.lock().regrant.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

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
        let control = Arc::new(Control::default());
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

    #[test]
    fn a_new_dispatch_before_the_handler_looks_keeps_its_grant() {
        let control = Control::default();
        control.dispatched();
        let (_, receiver) = oneshot::channel();
        assert!(control.pause(receiver));
        control.dispatched();
        assert!(
            control.take_regrant().is_some(),
            "the next grant still arrives where the handler will look"
        );
    }

    /// The race the old handshake lost about once in twenty thousand pauses:
    /// a handler that saw the request paused found no receiver for its next
    /// grant, and failed the request with "paused without a way to resume".
    #[test]
    fn a_handler_that_sees_a_pause_always_finds_its_next_grant() {
        for _ in 0..20_000 {
            let control = Arc::new(Control::default());
            control.dispatched();
            let engine = {
                let control = control.clone();
                std::thread::spawn(move || {
                    let (_, receiver) = oneshot::channel();
                    control.pause(receiver)
                })
            };
            let claimed = control.begin();
            let paused = engine.join().unwrap();
            assert_ne!(claimed, paused, "exactly one side wins");
            if paused {
                assert!(control.take_regrant().is_some());
            }
        }
    }
}
