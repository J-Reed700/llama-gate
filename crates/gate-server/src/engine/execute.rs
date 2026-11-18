//! Carrying out the scheduler's commands: grants and refusals reach their
//! handlers, and slot operations run in tasks that report their outcomes
//! back as events.

use super::report::{rejection_reason, rejection_status};
use super::*;

impl Engine {
    pub(super) fn execute(&mut self, command: Command) -> Option<Event> {
        let now = self.clock.now();
        match command {
            Command::Dispatch {
                ticket,
                slot,
                generation,
                exclusive,
                reuse,
            } => {
                let Some(waiter) = self.waiting.remove(&ticket) else {
                    return Some(Event::Withdrawn {
                        ticket,
                        relayed: Some(0),
                        now,
                    });
                };
                let waited = waiter.since.elapsed();
                self.metrics
                    .observe(Timing::QueueTime, waited.as_secs_f64());
                self.metrics.count(
                    "llama_gate_requests_total",
                    labels(&[("class", &waiter.class), ("outcome", "dispatched")]),
                    1,
                );
                let backend = self.name(slot.backend);
                info!(
                    %ticket,
                    client = %waiter.client,
                    class = %waiter.class,
                    dialect = waiter.dialect,
                    backend,
                    slot = slot.index,
                    need = waiter.need,
                    generation,
                    reuse,
                    waited_ms = waited.as_millis() as u64,
                    "dispatch"
                );
                self.emit(json!({
                    "type": "dispatch", "ticket": ticket, "client": waiter.client,
                    "class": waiter.class, "model": self.model_of(slot.backend),
                    "backend": backend, "slot": slot.index,
                    "need": waiter.need, "generation": generation, "reuse": reuse,
                    "waited_ms": waited.as_millis() as u64,
                }));
                waiter.control.dispatched();
                let grant = Grant::Run {
                    slot,
                    generation,
                    exclusive,
                    reuse,
                };
                if waiter.reply.send(grant).is_err() {
                    // The client left between being scheduled and being told.
                    return Some(Event::Withdrawn {
                        ticket,
                        relayed: Some(0),
                        now,
                    });
                }
                self.running.insert(
                    ticket,
                    Runner {
                        client: waiter.client,
                        client_id: waiter.client_id,
                        class: waiter.class,
                        dialect: waiter.dialect,
                        need: waiter.need,
                        control: waiter.control,
                    },
                );
            }
            Command::Reject { ticket, rejection } => {
                if let Some(waiter) = self.waiting.remove(&ticket) {
                    let reason = rejection_reason(&rejection);
                    self.metrics.count(
                        "llama_gate_requests_total",
                        labels(&[("class", &waiter.class), ("outcome", reason)]),
                        1,
                    );
                    warn!(%ticket, client = %waiter.client, class = %waiter.class, reason, "request refused");
                    self.emit(json!({
                        "type": "reject", "ticket": ticket, "client": waiter.client,
                        "class": waiter.class, "reason": reason,
                        "status": rejection_status(&rejection),
                    }));
                    let _ = waiter.reply.send(Grant::Refused(rejection));
                }
            }
            Command::Evict { slot, save, op } => {
                let (server, store, sender, clock) = self.tools(slot.backend)?;
                let backend = self.name(slot.backend);
                tokio::spawn(async move {
                    let (outcome, detail) =
                        evict(&server, &store, &sender, &backend, slot, save, "evict").await;
                    let _ = sender.send(Message::Outcome {
                        event: Event::Evicted {
                            slot,
                            op,
                            outcome,
                            now: clock.now(),
                        },
                        detail: Some(detail),
                    });
                });
            }
            Command::Save { slot, snapshot, op } => {
                let (server, store, sender, clock) = self.tools(slot.backend)?;
                let backend = self.name(slot.backend);
                tokio::spawn(async move {
                    let started = Instant::now();
                    let outcome = match server
                        .save(slot.index, &SnapshotStore::filename(snapshot))
                        .await
                    {
                        Ok(report) if report.tokens == Some(0) => {
                            cleared_behind(&backend);
                            if let Some(store) = store {
                                tokio::task::spawn_blocking(move || store.delete(snapshot));
                            }
                            SaveOutcome::Failed
                        }
                        Ok(report) => SaveOutcome::Saved {
                            bytes: report.bytes.max(1),
                            tokens: report.tokens.unwrap_or(0),
                        },
                        Err(ServerError::Unsupported(reason)) => {
                            let _ = sender.send(Message::SwapUnsupported(slot.backend, reason));
                            SaveOutcome::Failed
                        }
                        Err(error) => {
                            warn!(backend, slot = slot.index, %error, "saving the shared context failed; its unshared part will be lost");
                            SaveOutcome::Failed
                        }
                    };
                    let millis = started.elapsed().as_millis() as u64;
                    let detail = match outcome {
                        SaveOutcome::Saved { bytes, tokens } => {
                            info!(
                                backend,
                                slot = slot.index,
                                snapshot = snapshot.0,
                                tokens,
                                bytes,
                                millis,
                                "saved a context before reusing its shared part"
                            );
                            json!({"type": "save", "backend": backend, "slot": slot.index, "outcome": "saved", "tokens": tokens, "bytes": bytes, "ms": millis})
                        }
                        SaveOutcome::Failed => {
                            json!({"type": "save", "backend": backend, "slot": slot.index, "outcome": "failed", "ms": millis})
                        }
                    };
                    let _ = sender.send(Message::Outcome {
                        event: Event::Saved {
                            slot,
                            op,
                            outcome,
                            now: clock.now(),
                        },
                        detail: Some(detail),
                    });
                });
            }
            Command::Restore { slot, snapshot, op } => {
                let (server, _, sender, clock) = self.tools(slot.backend)?;
                let backend = self.name(slot.backend);
                tokio::spawn(async move {
                    let started = Instant::now();
                    let result = server
                        .restore(slot.index, &SnapshotStore::filename(snapshot))
                        .await;
                    let millis = started.elapsed().as_millis() as u64;
                    let (outcome, detail) = match result {
                        Ok(report) if report.tokens == Some(0) => {
                            warn!(
                                backend,
                                slot = slot.index,
                                snapshot = snapshot.0,
                                "the snapshot held nothing; it is forgotten"
                            );
                            (
                                RestoreOutcome::Failed,
                                json!({"type": "restore", "backend": backend, "slot": slot.index, "outcome": "failed", "ms": millis}),
                            )
                        }
                        Ok(report) => {
                            let tokens = report.tokens.unwrap_or(0);
                            info!(
                                backend,
                                slot = slot.index,
                                snapshot = snapshot.0,
                                tokens,
                                millis,
                                "restored context"
                            );
                            (
                                RestoreOutcome::Restored { tokens },
                                json!({"type": "restore", "backend": backend, "slot": slot.index, "outcome": "restored", "tokens": tokens, "ms": millis}),
                            )
                        }
                        Err(error) => {
                            warn!(backend, slot = slot.index, snapshot = snapshot.0, %error, "restore failed; the slot will be cleared and the context recomputed");
                            (
                                RestoreOutcome::Failed,
                                json!({"type": "restore", "backend": backend, "slot": slot.index, "outcome": "failed", "ms": millis}),
                            )
                        }
                    };
                    let _ = sender.send(Message::Outcome {
                        event: Event::Restored {
                            slot,
                            op,
                            outcome,
                            now: clock.now(),
                        },
                        detail: Some(detail),
                    });
                });
            }
            Command::Preempt {
                ticket,
                slot,
                save,
                op,
            } => return self.preempt(ticket, slot, save, op),
            Command::Forget { backend, snapshot } => {
                if let Some(store) = self
                    .backends
                    .get(usize::from(backend.0))
                    .and_then(|state| state.store.clone())
                {
                    tokio::task::spawn_blocking(move || store.delete(snapshot));
                }
            }
            Command::IndexChanged { backend } => {
                self.dirty.insert(backend);
            }
            Command::Notice(Notice::PartialReuseIneffective { backend }) => {
                let message = format!(
                    "backend {}: its model did not reuse contexts that share only part of a prompt (it cannot rewind to them), so they are no longer offered there",
                    self.name(backend)
                );
                warn!("{message}");
                self.emit(json!({"type": "notice", "message": message}));
            }
            Command::Notice(Notice::Overran {
                backend,
                ticket,
                reserved,
                resident,
            }) => {
                let name = self.name(backend);
                warn!(
                    backend = name,
                    %ticket,
                    reserved,
                    resident,
                    "a context finished larger than its reservation; later reservations on this backend carry the excess"
                );
                self.metrics.count(
                    "llama_gate_overruns_total",
                    labels(&[("backend", &name)]),
                    1,
                );
                self.emit(json!({
                    "type": "notice",
                    "message": format!("backend {name}: request {ticket} held {resident} tokens against {reserved} reserved"),
                }));
            }
        }
        None
    }

    /// Pauses a running request: it goes back in line with a fresh grant
    /// channel, and a task stops it on the server and saves its progress.
    pub(super) fn preempt(
        &mut self,
        ticket: TicketId,
        slot: SlotId,
        save: Option<SnapshotId>,
        op: gate_core::OpId,
    ) -> Option<Event> {
        let now = self.clock.now();
        let backend = self.name(slot.backend);
        if let Some(runner) = self.running.get(&ticket) {
            let (reply, receiver) = oneshot::channel();
            if !runner.control.pause(receiver) {
                self.metrics.count(
                    "llama_gate_preemptions_total",
                    labels(&[("outcome", "declined")]),
                    1,
                );
                self.emit(json!({"type": "preempt", "ticket": ticket, "client": runner.client, "backend": backend, "slot": slot.index, "outcome": "declined"}));
                return Some(Event::Preempted {
                    slot,
                    op,
                    outcome: PreemptOutcome::Declined,
                    now,
                });
            }
            if let Some(runner) = self.running.remove(&ticket) {
                info!(%ticket, client = %runner.client, backend, slot = slot.index, "paused to make room for more urgent work; it resumes where it stopped");
                self.emit(json!({"type": "preempt", "ticket": ticket, "client": runner.client, "backend": backend, "slot": slot.index, "outcome": "paused"}));
                self.metrics.count(
                    "llama_gate_preemptions_total",
                    labels(&[("outcome", "paused")]),
                    1,
                );
                self.waiting.insert(ticket, runner.waiter(reply));
            }
        }
        let (server, store, sender, clock) = self.tools(slot.backend)?;
        tokio::spawn(async move {
            // The handler drops its connection; the server then stops the
            // request and runs the save, which it defers until it has.
            let (outcome, detail) =
                evict(&server, &store, &sender, &backend, slot, save, "preempt").await;
            let _ = sender.send(Message::Outcome {
                event: Event::Preempted {
                    slot,
                    op,
                    outcome: PreemptOutcome::Stopped(outcome),
                    now: clock.now(),
                },
                detail: Some(detail),
            });
        });
        None
    }

    pub(super) fn tools(
        &self,
        backend: BackendId,
    ) -> Option<(
        Server,
        Option<SnapshotStore>,
        mpsc::UnboundedSender<Message>,
        Clock,
    )> {
        let state = self.backends.get(usize::from(backend.0))?;
        Some((
            state.link.server.clone(),
            state.store.clone(),
            self.sender.clone(),
            self.clock,
        ))
    }
}

/// Saves (when asked) and erases a slot, for an eviction or a pause.
pub(super) async fn evict(
    server: &Server,
    store: &Option<SnapshotStore>,
    sender: &mpsc::UnboundedSender<Message>,
    backend: &str,
    slot: SlotId,
    save: Option<SnapshotId>,
    kind: &str,
) -> (EvictOutcome, Value) {
    let started = Instant::now();
    let mut saved = None;
    if let Some(snapshot) = save {
        match server
            .save(slot.index, &SnapshotStore::filename(snapshot))
            .await
        {
            Ok(report) if report.tokens == Some(0) => {
                cleared_behind(backend);
                if let Some(store) = store.clone() {
                    tokio::task::spawn_blocking(move || store.delete(snapshot));
                }
            }
            Ok(report) => saved = Some((snapshot, report)),
            Err(ServerError::Unsupported(reason)) => {
                let _ = sender.send(Message::SwapUnsupported(slot.backend, reason));
            }
            Err(error) => {
                warn!(backend, slot = slot.index, %error, "saving the context failed; it will be discarded")
            }
        }
    }
    let millis = || started.elapsed().as_millis() as u64;
    match server.erase(slot.index).await {
        Ok(_) => match saved {
            Some((snapshot, report)) => {
                info!(
                    backend,
                    slot = slot.index,
                    snapshot = snapshot.0,
                    tokens = report.tokens.unwrap_or(0),
                    bytes = report.bytes,
                    millis = millis(),
                    "saved and cleared context"
                );
                (
                    EvictOutcome::Saved {
                        bytes: report.bytes.max(1),
                        tokens: report.tokens.unwrap_or(0),
                    },
                    json!({"type": kind, "backend": backend, "slot": slot.index, "outcome": "saved", "tokens": report.tokens, "bytes": report.bytes, "ms": millis()}),
                )
            }
            None => {
                debug!(backend, slot = slot.index, "cleared context");
                (
                    EvictOutcome::Erased,
                    json!({"type": kind, "backend": backend, "slot": slot.index, "outcome": "erased", "ms": millis()}),
                )
            }
        },
        Err(error) => {
            warn!(backend, slot = slot.index, %error, "clearing the slot failed; it will be retried");
            if let Some((snapshot, _)) = saved
                && let Some(store) = store.clone()
            {
                tokio::task::spawn_blocking(move || store.delete(snapshot));
            }
            (
                EvictOutcome::Failed,
                json!({"type": kind, "backend": backend, "slot": slot.index, "outcome": "failed", "ms": millis()}),
            )
        }
    }
}

/// A save found empty a slot the gate believed held a context: the server
/// clears idle slots itself (llama-server's `--cache-idle-slots`, on by
/// default with its prompt cache). Said once; it recurs with every save.
pub(super) fn cleared_behind(backend: &str) {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        warn!(
            backend,
            "the server clears idle slots itself, so the gate cannot save them; start llama-server with --no-cache-idle-slots"
        );
    }
}
