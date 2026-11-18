# Architecture

llama-gate is three crates with one dependency direction:

```text
gate-server   HTTP data plane, runtime actor, config, identity, metrics, launcher
   │
   ├──▶ gate-llama   llama.cpp control-plane client, request dialects, measurement
   │        │
   └────────┴──▶ gate-core   the scheduler: pure, deterministic, no I/O
```

`gate-core` knows nothing about HTTP, llama.cpp, or time. `gate-llama` knows
llama.cpp's API but nothing about scheduling. `gate-server` composes them.
A different backend that exposes slots would replace `gate-llama` only.

## The core: events in, commands out

`Scheduler::handle(Event) -> Vec<Command>` is the whole engine. Every event
carries the time it happened; the scheduler never reads a clock and never
performs I/O, so any sequence of events replays identically.

| Event | From |
| --- | --- |
| `Arrived` | a measured request |
| `Withdrawn` | its client left (before or after dispatch) |
| `Started` | its output began reaching the client: it can no longer be paused |
| `Finished` | its response completed, with the tokens the slot holds and the cache reuse the server reported |
| `Bounced` | its server refused the connection before anything reached it |
| `Evicted`, `Saved`, `Restored`, `Preempted` | a slot operation finished (carrying its `OpId`) |
| `Observed` | what a server reports it holds (adoption at startup) |
| `BackendUp`, `BackendDown` | a server came up (first time, back, restarted, or reshaped) or went away |
| `Tick`, `Shutdown` | time, and stopping |

| Command | The caller must |
| --- | --- |
| `Dispatch` | forward the request pinned to `slot`, limited to `generation` new tokens |
| `Evict` | save the slot to a snapshot (optional), then erase it; report `Evicted` |
| `Save` | save the slot and leave it; report `Saved` |
| `Restore` | load a snapshot into the slot; report `Restored` |
| `Preempt` | stop the request (its client keeps waiting), save the slot (optional), erase it; report `Preempted`, or `Declined` if its output had begun |
| `Reject` | answer the request with the refusal |
| `Forget` | delete a snapshot file |
| `IndexChanged` | persist a backend's snapshot index |
| `Notice` | tell an operator (say, partial reuse turned itself off) |

### Backends and models

The scheduler manages any number of **backends**, each a `llama-server`
with its own slots, its own KV pool (`Layout::Unified { capacity }` or
`Layout::Split { per_slot }`), and its own snapshot index. Backends are
announced with their `Shape` at runtime and may come and go. Each serves one
`ModelId`; backends serving the same model are interchangeable, and a request
may only be placed on a backend of its model. Models are independent: a
request waiting for model A never holds up model B.

### Slots and the accounting rule

Each slot is in exactly one state, and each state has a *usage*: the most
KV cells the slot can occupy in it.

```text
          ┌─────────── Finished / Withdrawn ─────────────┐
          ▼                                              │
 Empty ──▶ Idle ──(claimed, content kept)──────────▶ Running ──(paused)──▶ Preempting ──▶ Empty
  ▲  ▲      │                                        ▲    │
  │  │      └─(claimed, content replaced)─▶ Preparing ┘    └─(bounced)──▶ Faulted
  │  │      └─(room needed)──▶ Evicting ──▶ Empty
  │  └─────────────── Faulted ◀──(an operation failed; erase retried)
```

| State | Usage |
| --- | --- |
| Empty | 0 |
| Idle | the resident context's tokens |
| Evicting | the resident context's tokens, until the erase completes |
| Preparing | max(reservation, what the slot still physically holds) |
| Running | the reservation: prompt + granted generation (+ any restored surplus) |
| Preempting | the paused request's reservation, until the slot is erased |
| Faulted | what it held, until an erase succeeds |

A backend's usage is the sum over its slots, plus any *stray* cells (below).
The scheduler admits work only while that stays within a unified pool's
capacity, or each slot within its own share (split). Because the caller
enforces each request's granted generation as its limit, physical use can
never exceed usage, so no server's pool can overflow.

### Placing a request

1. **Order.** Waiting requests are sorted by class priority, then requests
   that were paused (admitted once already), then the `QueueOrder` key (VTC
   counter by default), then arrival. Each model's first request is its
   *head*.
2. **Plan** the head over every available slot (Empty or Idle) on every
   online backend of its model:
   - an Idle slot whose context is a prefix of the request is *reused* in
     place;
   - with partial reuse, one sharing only the prompt's start is reused in
     place too, after *saving* the context if the unshared part is worth
     keeping;
   - otherwise its context is *cleared* (saved first if worth keeping);
   - a snapshot covering the request's start may be *restored*;
   - the best option fits now, reuses an appreciably larger part of the
     prompt, runs on a less busy backend, reuses more, destroys the least,
     and runs on a less full backend, in that order.
3. If no option fits:
   - idle contexts on that backend that would make room are **evicted**
     (`EvictionOrder` picks which), and the head waits for them;
   - else, lower-priority requests of a *preemptible* class that have shown
     no output are **paused** (fewest first, least work lost), with idle
     contexts evicted as needed;
   - else the head **waits** for running work; meanwhile small requests of
     its model may **backfill** what is free, until the head has waited for
     the backfill window.
4. A request that could never fit any backend of its model is refused at
   once; one whose model has had no server online for `upstream_wait` is
   refused when that passes.

### Pausing

A running request may be paused only until its output begins: before that,
stopping and later restarting it is invisible to the client; after it, the
client would see a seam. llama.cpp offers no way to pause a slot, so the gate
pauses from outside: the handler drops its upstream connection (the server
stops the task), the slot is saved (the server defers the save until the
task has stopped) and erased, and the request goes back in line, ahead of
others of its priority. When it runs again, its snapshot is restored and the
server continues from where it stopped: a paused prefill resumes its prompt
instead of recomputing it.

The handler and the engine race for the moment output begins, and
`Control` settles it atomically: the handler claims it with `begin()` before
relaying the first event, the engine with `pause()` when it executes
`Preempt`. Exactly one wins; a lost pause is reported as `Declined` and the
request runs on. A request is paused at most `max_preemptions` times, and a
request started in the same scheduling pass is never paused in it.

### Partial reuse, learned

Reusing a context that shares only part of a prompt needs the server to
rewind to the shared point, which some models (hybrid recurrent, sliding
window) cannot do without a checkpoint there. Each dispatch carries the
reuse the scheduler expects; each finish reports the reuse the server
achieved (`timings.cache_n`). After three partial expectations missed without
one met, partial reuse is turned off for that backend, with a `Notice`.

### Requests the server places itself

Embeddings and reranking occupy slots, but llama.cpp ignores `id_slot` for
them and picks any idle slot. On a unified pool such a request runs alone
(the backend otherwise empty), so wherever it lands it fits. When it
finishes its cells may be in any slot, so they are counted once as *stray*
usage and every slot that may hold them is erased; when the last erase
completes the stray count is cleared. On a split pool each slot's share
bounds them anyway.

Content the gate never placed is handled the same way. llama-server reports
the prompt a slot last read but not the answer that followed, so it cannot
say how large a context really is. When the gate first meets a server, every
slot is erased before anything runs in it (contexts saved to disk are
kept). Later, a slot the ledger holds empty that the server reports holding a
prompt (a client went around the gate) is erased too, the rest of the pool
counted as taken until it is. A report never lowers what the ledger holds.

### Robustness details

- Every slot operation has an `OpId`; a completion that does not match the
  slot's current operation (a late report from before a restart) is ignored.
- A failed erase faults the slot; it is retried after `retry` until it
  succeeds, while its backend is online. A failed restore, or one that
  restores nothing, forgets the snapshot and clears the slot, and the
  request chooses again, keeping its place: another snapshot may hold its
  context.
- A save that finds its slot empty is discarded rather than indexed: the
  server cleared the slot itself (llama-server's `--cache-idle-slots`).
- A request bounced by a dead server is placed again (on another backend of
  its model if one is up); the dead backend is offline until its watcher
  sees it ready again.
- After a restart, running requests stay accounted until they report back,
  since a dispatch not yet sent will still run on the new server.
- A snapshot holds its conversation's prompt and answer; restoring it
  reserves whichever is larger, the snapshot or the request.

## Content addressing

`Prefix` hashes a token sequence in fixed blocks, each hash chained to the
previous one and seeded with a namespace (the model, plus the client when
`clients.isolate` is on), as vLLM's automatic prefix caching and llm-d's
prefix scorer do. Two prompts share their first *k* blocks exactly when
their first *k* hashes match, so prefix tests never need the tokens and
snapshots never match across models (or clients). A paused prefill's
snapshot is addressed by the truncated prefix it actually holds.

## Measurement

Each generative request is measured with the server's own template
(`/apply-template`) and tokenizer (`/tokenize`), so its prompt count matches
the server's exactly. A long conversation's prompt repeats its previous turn,
so `TokenCache` keeps the tokens of conversation heads and tokenizes only the
new text. Text can only be cut where tokenization composes: llama.cpp splits
a prompt at special tokens before tokenizing the plain text between them, so
the tokens of `a + b` are those of `a` then `b` when `b` starts with a
special token. The gate learns the template's special tokens from the server
(the tokens that only appear when special tokens are parsed) and cuts only
right before one, never after whitespace. It checks itself against full
tokenizations, the first few and then one in `verify_every`; a single
disagreement turns the cache off for that model.

## The runtime

`gate-server` runs one actor task that owns the scheduler. HTTP handlers send
it `Arrive`, `Withdraw`, `Started`, `Finish`, and `Bounce`; slot operations
and pauses run in spawned tasks that send their outcomes back. All scheduling
state changes on that one task, in the same serialized order the simulation
tests exercise. A watcher per backend polls health and shape; it reports a
**restart** only on evidence (task ids going backwards, or `/props`
changing), not on a single missed probe. The actor also keeps per-client
totals, the recent requests, and a broadcast of events for `/gate/events`
and the dashboard. A reload (SIGHUP or `POST /gate/reload`) swaps classes,
clients, keys, and policy settings in place, remapping waiting requests'
classes by name.

## The data plane

```text
request ─▶ authenticate ─▶ dialect? ─no─▶ pass through (to a server of the named model)
                              │yes
                              ▼
                 identify ─▶ measure ─▶ Arrive ─▶ session
                                                     │
    ┌──────────────────── grant ◀───────────────────┘
    ▼
  attempt: pin id_slot, set the limit ─▶ forward ─┬─ response ─▶ relay ─▶ Finish
                                                  ├─ paused ───▶ wait for the next grant
                                                  └─ refused ──▶ Bounce ─▶ next grant
```

A generative request is a **session** that may span several attempts. The
client sees one response: a streaming response is answered through a
channel that outlives each attempt, mirroring the first server's headers (or
committing to `200 text/event-stream` with keepalive comments once the
request has waited `keepalive_after`); between attempts the client receives
pings, and a refusal or error after commitment arrives in-band as an `error`
event, as llama.cpp reports mid-stream errors. However the session ends,
the engine hears exactly once: `Finish` with the usage read from the
response's tail, or `Withdraw`.

## Extension points

| Change | Implement | Register |
| --- | --- | --- |
| Queue order | `gate_core::QueueOrder` | `Gate::queue_order` |
| Eviction victims | `gate_core::EvictionOrder` | `Gate::eviction_order` |
| Client identity, class, and admission | `gate_server::clients::Identify` | `Gate::identify` |
| A new endpoint | `gate_llama::Dialect` | `Gate::dialects(Dialects::standard().with(..))` |

The traits are deliberately small: each sees only what its decision needs,
and the engine guarantees the rest.

## Testing

| Layer | What it proves |
| --- | --- |
| Unit tests | each module's rules (hashing, VTC, budgets, parsing, dialects, keys, the pause handshake, incremental tokenization) |
| Scenarios (`gate-core/tests/scenarios.rs`) | each guarantee as a story |
| Simulation (`gate-core/tests/simulation.rs`) | the invariants under random workloads over one to three backends and two models, with failed operations, restarts, outages, bounces, pauses, and server-placed requests |
| End to end (`gate-server/tests/gate.rs`) | the real gate against fake servers that fail like llama.cpp |

After every simulated event: no pool exceeds its capacity, accounting never
under-counts what a server holds, and no request occupies two slots. At the
end: nothing waits, nothing leaked, and no request ran more often than once
plus once per pause or bounce.

## References

- Y. Sheng et al., *Fairness in Serving Large Language Models*, OSDI 2024 (VTC).
- vLLM automatic prefix caching: block hashes chained to their parent;
  preemption of running requests under memory pressure.
- llm-d inference scheduler: priority bands, weighted fair share, 64-token
  prefix blocks seeded per model.
- NVIDIA Dynamo and SGLang's router: routing that weighs cache overlap
  against load.
- A. Mu'alem and D. Feitelson, *Utilization, Predictability, Workloads, and
  User Runtime Estimates in Scheduling the IBM SP2 with Backfilling* (EASY).
