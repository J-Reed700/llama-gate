//! Deterministic simulation: random workloads, failures, restarts, outages,
//! and pauses, over one to three backends serving one or two models.
//!
//! Every seed replays exactly. After every event the harness checks that no
//! pool is ever overfilled, that accounting never under-counts what a server
//! physically holds, and that no request occupies two slots; at the end it
//! checks that every request was dispatched once per pause or bounce at
//! most, that nothing is left waiting, and that nothing leaked.
#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support;

use gate_core::{Backfill, ClassId, Layout, ModelId, Tokens};
use std::collections::BTreeSet;
use support::{BackendSetup, Setup, class};

/// xorshift64*: tiny, deterministic, good enough for workloads.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound.max(1)
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

fn run(seed: u64) {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let backends = 1 + rng.below(3) as u16;
    let models = if backends > 1 && rng.chance(40) { 2 } else { 1 };
    let mut setup = Setup::unified(1, 1);
    setup.backends = (0..backends)
        .map(|backend| {
            let capacity: Tokens = 40_000 + rng.below(160_000);
            let slots = 1 + rng.below(4) as usize;
            BackendSetup {
                model: backend % models,
                layout: if rng.chance(30) {
                    Layout::Split {
                        per_slot: capacity / slots as Tokens,
                    }
                } else {
                    Layout::Unified { capacity }
                },
                slots,
                swap: rng.chance(70),
            }
        })
        .collect();
    // Servers may hold contexts from before the gate started.
    setup.holding = setup
        .backends
        .iter()
        .map(|backend| {
            let mut free = backend.layout.request_limit();
            (0..backend.slots)
                .map(|_| {
                    let held = if rng.chance(50) {
                        rng.below(free / 2 + 1)
                    } else {
                        0
                    };
                    if matches!(backend.layout, Layout::Unified { .. }) {
                        free -= held;
                    }
                    held
                })
                .collect()
        })
        .collect();
    let mut agents = class("agents", 0);
    agents.preemptible = rng.chance(60);
    setup.classes = vec![agents, class("interactive", 10)];
    setup.vtc = rng.chance(70);
    setup.eviction.partial = rng.chance(70);
    // The harness writes 1,000-byte snapshots: keep a few dozen at most, so
    // the budget's forgetting is exercised too.
    setup.eviction.snapshot_budget = 1_000 * (4 + rng.below(40));
    setup.max_preemptions = 1 + rng.below(3) as u32;
    // Each model's smallest backend bounds what is sensible to ask for.
    let limits: Vec<Tokens> = (0..models)
        .map(|model| {
            setup
                .backends
                .iter()
                .filter(|backend| backend.model == model)
                .map(|backend| backend.layout.request_limit())
                .min()
                .unwrap_or(1)
        })
        .collect();
    if rng.chance(50) {
        setup.backfill = Some(Backfill {
            max_tokens: limits[0] / 8,
            window: 2_000,
        });
    }
    let mut gate = setup.build();
    gate.auto = false;

    // Conversations grow turn by turn, some on a shared system prompt, so
    // prefixes, partial reuse, and snapshots all matter.
    let mut conversations: Vec<(u32, usize, u16)> = (0..8)
        .map(|id| (id, 0, (id % models as u32) as u16))
        .collect();
    let mut arrived = BTreeSet::new();
    let mut granted = std::collections::BTreeMap::new();

    for _ in 0..500 {
        match rng.below(12) {
            0..=3 => {
                let pick = rng.below(conversations.len() as u64) as usize;
                let (id, length, model) = conversations[pick];
                let limit = limits[usize::from(model)];
                let grown = (length + 200 + rng.below(limit / 3) as usize).min(limit as usize);
                let length = if rng.chance(15) { grown / 2 } else { grown };
                conversations[pick].1 = length;
                let generation = 1 + rng.below(limit / 6);
                let system = (length / 3).min(8_000);
                let prefix = if id % 2 == 0 {
                    support::forked(1_000 + u32::from(model), system, id, length - system)
                } else {
                    support::prefix(id, length)
                };
                let class = ClassId(u16::from(rng.chance(25)));
                let ticket = if rng.chance(10) {
                    // An embedding: no generation, a slot of the server's
                    // choosing.
                    gate.arrive_unpinned(
                        ModelId(model),
                        id % 3,
                        (length / 4).max(1) as Tokens,
                        prefix,
                    )
                } else {
                    gate.arrive_for(
                        ModelId(model),
                        id % 3,
                        class,
                        length as Tokens,
                        generation,
                        prefix,
                        false,
                    )
                };
                arrived.insert(ticket);
            }
            4..=5 => {
                if let Some(&ticket) = gate.running.keys().nth(rng.below(4) as usize) {
                    let reserved = gate
                        .was_dispatched(ticket)
                        .map_or(0, |dispatch| dispatch.generation);
                    gate.cached = if rng.chance(50) {
                        Some(rng.below(gate.running[&ticket].prompt + 1))
                    } else {
                        None
                    };
                    gate.finish(ticket, rng.below(500).min(reserved));
                }
            }
            6 => {
                let pick = rng.below(arrived.len() as u64) as usize;
                if let Some(&ticket) = arrived.iter().nth(pick)
                    && rng.chance(20)
                {
                    gate.withdraw(ticket);
                }
            }
            7..=8 => {
                let succeed = !rng.chance(10);
                if let Some(gate_core::Command::Dispatch {
                    ticket, generation, ..
                }) = gate.step(succeed)
                {
                    granted.insert(ticket, generation);
                }
            }
            9 => {
                if let Some(&ticket) = gate.running.keys().nth(rng.below(4) as usize) {
                    gate.start(ticket);
                }
            }
            _ => {
                let backend = rng.below(u64::from(backends)) as u16;
                match rng.below(100) {
                    0..=2 => gate.restart(backend),
                    3..=4 => gate.down(backend),
                    5..=9 => gate.up(backend),
                    10..=12 => gate.bounce_next = true,
                    13..=24 => {
                        // A client that bypasses the gate; the server reports
                        // only the prompt it read, not the answer.
                        let shape = gate.shapes[usize::from(backend)];
                        let index = rng.below(shape.slots as u64) as usize;
                        let tokens = 1 + rng.below(shape.layout.request_limit() / 2);
                        let reported = 1 + rng.below(tokens);
                        gate.intrude(backend, index, tokens, reported);
                    }
                    _ => gate.advance(1 + rng.below(3_000)),
                }
            }
        }
    }

    // Quiesce: every server back, every operation succeeding, every request
    // finishing.
    gate.bounce_next = false;
    for backend in 0..backends {
        gate.up(backend);
    }
    gate.auto = true;
    gate.drain();
    for _ in 0..2_000 {
        let Some(&ticket) = gate.running.keys().next() else {
            let settled = gate
                .scheduler
                .status(gate.now)
                .backends
                .iter()
                .all(|backend| {
                    backend
                        .slots
                        .iter()
                        .all(|slot| slot.state == "idle" || slot.state == "empty")
                });
            if gate.scheduler.waiting() == 0 && settled {
                break;
            }
            // Time drives queue timeouts and the retry of faulted slots.
            gate.advance(10_000);
            continue;
        };
        gate.finish(ticket, 1);
    }

    for ticket in &arrived {
        let dispatches = gate.dispatches(*ticket);
        let allowed = 1
            + gate
                .preempted
                .iter()
                .filter(|paused| *paused == ticket)
                .count()
            + gate
                .bounced
                .iter()
                .filter(|(bounced, _)| bounced == ticket)
                .count();
        assert!(
            dispatches <= allowed,
            "seed {seed}: ticket {ticket} dispatched {dispatches} times, allowed {allowed}"
        );
        let rejected = gate.rejected.iter().any(|(refused, _)| refused == ticket);
        assert!(
            !(dispatches > 0 && rejected && gate.preempted.iter().all(|paused| paused != ticket)),
            "seed {seed}: ticket {ticket} both ran and was refused"
        );
    }
    assert_eq!(
        gate.scheduler.waiting(),
        0,
        "seed {seed}: requests left waiting: {:?}",
        gate.scheduler.status(gate.now).waiting
    );
    let status = gate.scheduler.status(gate.now);
    for backend in &status.backends {
        assert!(
            backend
                .slots
                .iter()
                .all(|slot| slot.state == "idle" || slot.state == "empty"),
            "seed {seed}: slots left busy: {:?}",
            backend.slots
        );
    }
}

#[test]
fn random_workloads_never_overfill_and_never_lose_a_request() {
    // SIM_SEEDS widens the search; SIM_SEED replays one seed (add SIM_TRACE
    // to print every event).
    if let Some(seed) = std::env::var("SIM_SEED")
        .ok()
        .and_then(|seed| seed.parse().ok())
    {
        run(seed);
        return;
    }
    let seeds: u64 = std::env::var("SIM_SEEDS")
        .ok()
        .and_then(|count| count.parse().ok())
        .unwrap_or(2_000);
    // SIM_THREADS sets the parallelism; by default half the machine, at
    // most eight threads, so a run never starves the host.
    let workers = std::env::var("SIM_THREADS")
        .ok()
        .and_then(|count| count.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(2, |count| (count.get() / 2).clamp(1, 8))
                as u64
        });
    std::thread::scope(|scope| {
        for worker in 0..workers {
            scope.spawn(move || {
                let mut seed = worker;
                while seed < seeds {
                    let outcome = std::panic::catch_unwind(|| run(seed));
                    if let Err(panic) = outcome {
                        eprintln!("simulation failed at seed {seed}; replay with SIM_SEED={seed} SIM_TRACE=1");
                        std::panic::resume_unwind(panic);
                    }
                    seed += workers;
                }
            });
        }
    });
}
