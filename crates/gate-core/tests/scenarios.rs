//! Behavioural scenarios for the scheduler, each one a guarantee.
#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support;

use gate_core::{
    BackendId, Backfill, ClassId, Command, CostBenefit, Demand, Event, EvictionPolicy, Layout,
    ModelId, Notice, Prefix, Rejection, Scheduler, SchedulerConfig, Shape, TicketId, Vtc,
};
use support::{BackendSetup, Setup, class, forked, prefix, slot};

const POOL: u64 = 196_608;

/// The incident that motivated the gate: two agents with ~115k-token
/// contexts on a 196k unified pool. Unguarded, llama.cpp overfilled the pool
/// and failed both requests on every retry. The gate must run them one at a
/// time, keep the first agent's context on disk while the second runs, and
/// never exceed the pool.
#[test]
fn two_large_agents_take_turns_instead_of_overfilling_the_pool() {
    let mut gate = Setup::unified(POOL, 2).build();
    let first = gate.arrive(1, 115_000, 8_192, prefix(1, 115_000));
    let second = gate.arrive(2, 112_000, 8_192, prefix(2, 112_000));
    assert!(gate.was_dispatched(first).is_some());
    assert!(
        gate.was_dispatched(second).is_none(),
        "the second agent waits while the first could still grow into the pool"
    );

    gate.finish(first, 4_000);
    assert!(
        gate.was_dispatched(second).is_some(),
        "runs once the first finishes"
    );
    assert_eq!(
        gate.files[0].values().copied().collect::<Vec<_>>(),
        vec![119_000],
        "the idle first context was saved to disk, not thrown away"
    );

    // The first agent comes back with its conversation extended: it waits for
    // the second, then its context is restored from disk instead of being
    // recomputed.
    let returning = gate.arrive(1, 117_000, 8_192, forked(1, 115_000, 9, 2_000));
    assert!(gate.was_dispatched(returning).is_none());
    gate.finish(second, 3_000);
    let dispatch = gate.was_dispatched(returning).expect("dispatched");
    assert_eq!(dispatch.reuse, 119_000, "the restored context is reused");
    assert_eq!(gate.occupied(0), 1);
}

#[test]
fn small_requests_run_beside_a_large_one() {
    let mut gate = Setup::unified(POOL, 2).build();
    let large = gate.arrive(1, 150_000, 8_192, prefix(1, 150_000));
    let small = gate.arrive(2, 4_000, 2_048, prefix(2, 4_000));
    assert!(gate.was_dispatched(large).is_some());
    assert_eq!(
        gate.was_dispatched(small).map(|dispatch| dispatch.slot),
        Some(slot(0, 1))
    );
}

#[test]
fn a_continuing_conversation_returns_to_the_slot_that_holds_it() {
    let mut gate = Setup::unified(POOL, 2).build();
    let turn = gate.arrive(1, 6_000, 1_000, prefix(1, 6_000));
    gate.finish(turn, 500);
    let other = gate.arrive(2, 3_000, 1_000, prefix(2, 3_000));
    gate.finish(other, 200);

    let next = gate.arrive(1, 6_900, 1_000, forked(1, 6_000, 5, 900));
    let dispatch = gate.was_dispatched(next).expect("dispatched");
    assert_eq!(dispatch.slot, gate.was_dispatched(turn).unwrap().slot);
    assert_eq!(dispatch.reuse, 6_000);
    assert!(
        gate.files[0].is_empty(),
        "nothing needed saving: the slot was reused in place"
    );
}

#[test]
fn a_light_client_is_served_between_a_heavy_clients_jobs() {
    let mut gate = Setup::unified(POOL, 1).swap(false).build();
    let heavy: Vec<_> = (0..4)
        .map(|index| gate.arrive(1, 8_000, 4_000, prefix(100 + index, 8_000)))
        .collect();
    let light = gate.arrive(2, 1_000, 500, prefix(7, 1_000));
    gate.finish(heavy[0], 4_000);
    assert!(
        gate.was_dispatched(light).is_some(),
        "fair queueing serves the light client next, not after every heavy job"
    );
    assert!(gate.was_dispatched(heavy[1]).is_none());
}

#[test]
fn higher_priority_classes_go_first() {
    let mut setup = Setup::unified(POOL, 1);
    setup.classes = vec![class("batch", 0), class("interactive", 10)];
    let mut gate = setup.build();
    let running = gate.arrive_with(1, ClassId(0), 2_000, 500, prefix(1, 2_000), false);
    let batch = gate.arrive_with(1, ClassId(0), 2_000, 500, prefix(2, 2_000), false);
    let interactive = gate.arrive_with(2, ClassId(1), 2_000, 500, prefix(3, 2_000), false);
    gate.finish(running, 100);
    assert!(gate.was_dispatched(interactive).is_some());
    assert!(gate.was_dispatched(batch).is_none());
}

#[test]
fn a_prompt_that_can_never_fit_is_refused_at_once() {
    let mut gate = Setup::unified(10_000, 2).build();
    let ticket = gate.arrive(1, 12_000, 100, prefix(1, 12_000));
    assert_eq!(
        gate.rejected,
        vec![(
            ticket,
            Rejection::TooLarge {
                prompt: 12_000,
                limit: 10_000
            }
        )]
    );
}

#[test]
fn generation_is_clamped_to_what_the_pool_can_hold() {
    let mut gate = Setup::unified(10_000, 1).build();
    let ticket = gate.arrive(1, 9_000, 4_000, prefix(1, 9_000));
    assert_eq!(
        gate.was_dispatched(ticket).map(|d| d.generation),
        Some(1_000)
    );
}

#[test]
fn requests_that_wait_too_long_are_refused() {
    let mut setup = Setup::unified(POOL, 1);
    setup.classes[0].max_wait = Some(1_000);
    let mut gate = setup.build();
    let running = gate.arrive(1, 1_000, 100, prefix(1, 1_000));
    let waiting = gate.arrive(2, 1_000, 100, prefix(2, 1_000));
    gate.advance(1_001);
    assert!(gate.was_dispatched(running).is_some());
    assert!(matches!(
        gate.rejected.as_slice(),
        [(ticket, Rejection::Timeout { .. })] if *ticket == waiting
    ));
}

#[test]
fn a_client_cannot_flood_the_queue() {
    let mut setup = Setup::unified(POOL, 1);
    setup.classes[0].max_queued_per_client = Some(2);
    let mut gate = setup.build();
    gate.arrive(1, 1_000, 100, prefix(1, 1_000));
    gate.arrive(1, 1_000, 100, prefix(2, 1_000));
    gate.arrive(1, 1_000, 100, prefix(3, 1_000));
    let refused = gate.arrive(1, 1_000, 100, prefix(4, 1_000));
    assert!(matches!(
        gate.rejected.as_slice(),
        [(ticket, Rejection::QueueFull { queued: 2 })] if *ticket == refused
    ));
}

#[test]
fn a_withdrawn_request_frees_its_place() {
    let mut gate = Setup::unified(POOL, 1).build();
    let first = gate.arrive(1, 1_000, 100, prefix(1, 1_000));
    let second = gate.arrive(2, 1_000, 100, prefix(2, 1_000));
    let third = gate.arrive(3, 1_000, 100, prefix(3, 1_000));
    gate.withdraw(second);
    gate.withdraw(first);
    assert!(gate.was_dispatched(second).is_none());
    assert!(gate.was_dispatched(third).is_some());
}

#[test]
fn a_failed_restore_is_forgotten_and_the_request_still_runs() {
    let mut gate = Setup::unified(POOL, 1).build();
    let first = gate.arrive(1, 50_000, 1_000, prefix(1, 50_000));
    gate.finish(first, 100);
    let other = gate.arrive(2, 50_000, 1_000, prefix(2, 50_000));
    gate.finish(other, 100);
    assert_eq!(
        gate.files[0].len(),
        1,
        "the first context was saved to make room"
    );

    gate.auto = false;
    let returning = gate.arrive(1, 50_500, 1_000, forked(1, 50_000, 3, 500));
    // Clearing the slot (saving the second context) succeeds; the restore fails.
    assert!(matches!(gate.step(true), Some(Command::Evict { .. })));
    assert!(matches!(gate.step(false), Some(Command::Restore { .. })));
    gate.drain();
    assert!(gate.was_dispatched(returning).is_some());
    assert_eq!(
        gate.scheduler.snapshots(BackendId(0)).unwrap().len(),
        1,
        "the broken snapshot was forgotten; only the second context remains saved"
    );
}

#[test]
fn a_failed_restore_falls_back_to_another_snapshot() {
    let mut gate = Setup::unified(POOL, 1).build();
    // One conversation is saved twice as it grows, at 64k and at 89.6k.
    for (conversation, length) in [(1, 64_000), (2, 64_000), (1, 89_600), (3, 64_000)] {
        let ticket = gate.arrive(
            conversation,
            length,
            1_000,
            prefix(conversation, length as usize),
        );
        gate.finish(ticket, 100);
    }
    assert_eq!(gate.files[0].len(), 3);

    gate.auto = false;
    let returning = gate.arrive(1, 96_000, 1_000, prefix(1, 96_000));
    assert!(matches!(gate.step(true), Some(Command::Evict { .. })));
    assert!(
        matches!(gate.step(false), Some(Command::Restore { .. })),
        "the longer snapshot fails to restore"
    );
    gate.drain();
    let reuse = gate.was_dispatched(returning).unwrap().reuse;
    assert!(
        (64_000..89_600).contains(&reuse),
        "the request chose again and restored the shorter snapshot ({reuse})"
    );
}

#[test]
fn a_slot_that_cannot_be_erased_is_retried_until_it_can() {
    let mut gate = Setup::unified(POOL, 1).build();
    let first = gate.arrive(1, 100_000, 1_000, prefix(1, 100_000));
    gate.finish(first, 100);
    gate.auto = false;
    let other = gate.arrive(2, 100_000, 1_000, prefix(2, 100_000));
    assert!(matches!(gate.step(false), Some(Command::Evict { .. })));
    assert!(gate.was_dispatched(other).is_none());
    gate.auto = true;
    gate.advance(EvictionPolicy::default().retry_after);
    assert!(
        gate.was_dispatched(other).is_some(),
        "the retried erase made room"
    );
}

#[test]
fn a_server_restart_puts_preparing_requests_back_in_line() {
    let mut gate = Setup::unified(POOL, 1).build();
    let first = gate.arrive(1, 10_000, 1_000, prefix(1, 10_000));
    gate.finish(first, 100);
    gate.auto = false;
    let second = gate.arrive(2, 10_000, 1_000, prefix(2, 10_000));
    assert!(matches!(gate.pending.front(), Some(Command::Evict { .. })));
    gate.restart(0);
    gate.auto = true;
    gate.drain();
    assert!(gate.was_dispatched(second).is_some());
}

#[test]
fn split_pools_never_couple_slots() {
    let mut setup = Setup::unified(POOL, 2);
    setup.backends[0].layout = Layout::Split { per_slot: 98_304 };
    let mut gate = setup.build();
    let first = gate.arrive(1, 90_000, 8_000, prefix(1, 90_000));
    let second = gate.arrive(2, 90_000, 8_000, prefix(2, 90_000));
    assert!(gate.was_dispatched(first).is_some());
    assert!(gate.was_dispatched(second).is_some());
    let too_big = gate.arrive(3, 99_000, 100, prefix(3, 99_000));
    assert!(matches!(
        gate.rejected.as_slice(),
        [(ticket, Rejection::TooLarge { limit: 98_304, .. })] if *ticket == too_big
    ));
}

#[test]
fn small_requests_backfill_while_a_large_one_waits_but_not_forever() {
    let mut setup = Setup::unified(100_000, 2).swap(false);
    setup.backfill = Some(Backfill {
        max_tokens: 10_000,
        window: 5_000,
    });
    let mut gate = setup.build();
    let running = gate.arrive(1, 60_000, 1_000, prefix(1, 60_000));
    let large = gate.arrive(2, 50_000, 1_000, prefix(2, 50_000));
    let small = gate.arrive(3, 5_000, 1_000, prefix(3, 5_000));
    assert!(gate.was_dispatched(running).is_some());
    assert!(gate.was_dispatched(large).is_none());
    assert!(
        gate.was_dispatched(small).is_some(),
        "fits beside the running request"
    );

    gate.finish(small, 100);
    gate.advance(5_000);
    let late = gate.arrive(4, 5_000, 1_000, prefix(4, 5_000));
    assert!(
        gate.was_dispatched(late).is_none(),
        "after the window, room is held for the waiting large request"
    );
    gate.finish(running, 100);
    assert!(gate.was_dispatched(large).is_some());
}

#[test]
fn exclusive_requests_wait_for_an_idle_server() {
    let mut gate = Setup::unified(POOL, 2).build();
    let running = gate.arrive(1, 10_000, 1_000, prefix(1, 10_000));
    let exclusive = gate.arrive_with(2, ClassId(0), 1_000, 1_000, Prefix::opaque(1_000), true);
    assert!(gate.was_dispatched(exclusive).is_none());
    gate.finish(running, 100);
    assert!(gate.was_dispatched(exclusive).is_some());
}

// llama-server reports the prompt a slot last read, but not the answer that
// followed it, so it can never tell the gate how much a slot really holds.

#[test]
fn contexts_found_at_startup_are_erased_before_use() {
    let mut gate = Setup::unified(100_000, 2).holding(&[80_000, 0]).build();
    assert_eq!(gate.physical[0], vec![0, 0], "every slot is erased at once");
    assert!(
        gate.files[0].is_empty(),
        "an unknown context cannot be matched later, so it is erased, not saved"
    );
    let ticket = gate.arrive(1, 90_000, 1_000, prefix(1, 90_000));
    assert!(
        gate.was_dispatched(ticket).is_some(),
        "the whole pool is free"
    );
}

#[test]
fn contexts_placed_behind_the_gates_back_are_erased() {
    let mut gate = Setup::unified(100_000, 2).build();
    let running = gate.arrive(1, 30_000, 1_000, prefix(1, 30_000));
    let other = 1 - gate.was_dispatched(running).unwrap().slot.index;
    gate.auto = false;
    gate.intrude(0, other, 50_000, 40_000);
    assert_eq!(
        gate.scheduler.status(gate.now).backends[0].usage,
        100_000,
        "until the slot is erased, the rest of the pool counts as taken"
    );
    assert!(
        gate.pending.iter().any(|command| matches!(
            command,
            gate_core::Command::Evict { slot, save: None, .. } if slot.index == other
        )),
        "the slot is erased, not saved"
    );
    let waiting = gate.arrive(2, 20_000, 1_000, prefix(2, 20_000));
    assert!(gate.was_dispatched(waiting).is_none());
    gate.drain();
    assert!(gate.was_dispatched(waiting).is_some());
}

#[test]
fn a_report_never_lowers_what_the_gate_knows() {
    let mut gate = Setup::unified(100_000, 2).build();
    let ticket = gate.arrive(1, 30_000, 2_000, prefix(1, 30_000));
    let index = gate.was_dispatched(ticket).unwrap().slot.index;
    gate.finish(ticket, 2_000);
    let held =
        |gate: &support::Harness| gate.scheduler.status(gate.now).backends[0].slots[index].tokens;
    let before = held(&gate);
    let mut reports = [(false, Some(0)); 2];
    reports[index] = (false, Some(30_000));
    gate.observe(0, &reports);
    assert_eq!(held(&gate), before, "the report leaves out the answer");
}

#[test]
fn slots_a_server_does_not_report_are_trusted_once_erased() {
    let mut gate = Setup::unified(100_000, 2).build();
    gate.auto = false;
    gate.observe(0, &[(false, None), (false, None)]);
    assert!(
        gate.pending.is_empty(),
        "both were erased when the gate met the server"
    );
}

// ------------------------------------------------------------ several servers

#[test]
fn work_spreads_over_backends_and_conversations_return_to_theirs() {
    let mut gate = Setup::unified(POOL, 2).replicate().build();
    let first = gate.arrive(1, 20_000, 1_000, prefix(1, 20_000));
    let second = gate.arrive(2, 20_000, 1_000, prefix(2, 20_000));
    let backends =
        |gate: &support::Harness, ticket| gate.was_dispatched(ticket).unwrap().slot.backend;
    assert_ne!(
        backends(&gate, first),
        backends(&gate, second),
        "each runs on its own server rather than sharing one"
    );
    gate.finish(first, 500);
    gate.finish(second, 500);
    let continuing = gate.arrive(2, 21_000, 1_000, forked(2, 20_000, 8, 1_000));
    assert_eq!(backends(&gate, continuing), backends(&gate, second));
}

#[test]
fn requests_only_run_on_backends_of_their_model() {
    let mut setup = Setup::unified(POOL, 1);
    setup.backends.push(BackendSetup {
        model: 1,
        layout: Layout::Unified { capacity: 50_000 },
        slots: 1,
        swap: true,
    });
    let mut gate = setup.build();
    let large = gate.arrive_for(
        ModelId(1),
        1,
        ClassId(0),
        40_000,
        1_000,
        prefix(1, 40_000),
        false,
    );
    let other = gate.arrive_for(
        ModelId(1),
        2,
        ClassId(0),
        30_000,
        1_000,
        prefix(2, 30_000),
        false,
    );
    let first_model = gate.arrive(3, 1_000, 100, prefix(3, 1_000));
    assert_eq!(gate.was_dispatched(large).unwrap().slot, slot(1, 0));
    assert!(
        gate.was_dispatched(other).is_none(),
        "model 1's only server is busy, and model 0's server is not an option"
    );
    assert_eq!(
        gate.was_dispatched(first_model).unwrap().slot,
        slot(0, 0),
        "a model waiting does not hold up another model"
    );
    let unknown = gate.arrive_for(
        ModelId(7),
        4,
        ClassId(0),
        1_000,
        100,
        prefix(4, 1_000),
        false,
    );
    assert!(gate.was_dispatched(unknown).is_none());
    assert!(
        gate.rejected.is_empty(),
        "a model without servers waits for one"
    );
    let too_big = gate.arrive_for(
        ModelId(1),
        5,
        ClassId(0),
        60_000,
        100,
        prefix(5, 60_000),
        false,
    );
    assert!(matches!(
        gate.rejected.as_slice(),
        [(ticket, Rejection::TooLarge { limit: 50_000, .. })] if *ticket == too_big
    ));
}

#[test]
fn a_request_bounced_by_a_dead_server_runs_on_another() {
    let mut gate = Setup::unified(POOL, 1).replicate().build();
    gate.bounce_next = true;
    let ticket = gate.arrive(1, 5_000, 500, prefix(1, 5_000));
    let [(bounced, refused)] = gate.bounced[..] else {
        panic!("one bounce expected: {:?}", gate.bounced);
    };
    assert_eq!(bounced, ticket);
    let dispatch = gate.was_dispatched(ticket).expect("placed again");
    assert_ne!(dispatch.slot.backend, refused.backend);
    assert!(!gate.scheduler.is_online(refused.backend));
}

#[test]
fn with_its_only_server_down_a_request_waits_and_runs_when_it_returns() {
    let mut gate = Setup::unified(POOL, 1).build();
    gate.bounce_next = true;
    let ticket = gate.arrive(1, 5_000, 500, prefix(1, 5_000));
    assert!(gate.was_dispatched(ticket).is_none());
    assert!(gate.rejected.is_empty());
    gate.up(0);
    // The slot it bounced off is erased before reuse, as its content is no
    // longer known.
    gate.advance(EvictionPolicy::default().retry_after);
    assert!(gate.was_dispatched(ticket).is_some());
}

// ------------------------------------------------------------------ pausing

fn priorities() -> Setup {
    let mut setup = Setup::unified(100_000, 2);
    let mut agents = class("agents", 0);
    agents.preemptible = true;
    setup.classes = vec![agents, class("interactive", 10)];
    setup
}

#[test]
fn an_urgent_request_pauses_work_that_has_shown_nothing_and_it_resumes_where_it_stopped() {
    let mut gate = priorities().build();
    let agent = gate.arrive_with(1, ClassId(0), 80_000, 8_000, prefix(1, 80_000), false);
    assert!(gate.was_dispatched(agent).is_some());
    let chat = gate.arrive_with(2, ClassId(1), 30_000, 4_000, prefix(2, 30_000), false);
    assert_eq!(
        gate.preempted,
        vec![agent],
        "the agent is paused, not failed"
    );
    assert!(gate.was_dispatched(chat).is_some());
    assert_eq!(
        gate.files[0].values().copied().collect::<Vec<_>>(),
        vec![80_000],
        "its progress was saved"
    );
    assert_eq!(gate.dispatches(agent), 1, "it waits for the chat");
    gate.finish(chat, 1_000);
    let resumed = gate.was_dispatched(agent).expect("resumed");
    assert_eq!(gate.dispatches(agent), 2);
    assert_eq!(resumed.reuse, 80_000, "from its saved progress");
    assert!(gate.rejected.is_empty());
}

#[test]
fn work_whose_output_has_begun_is_never_paused() {
    let mut gate = priorities().build();
    let agent = gate.arrive_with(1, ClassId(0), 80_000, 8_000, prefix(1, 80_000), false);
    gate.start(agent);
    let chat = gate.arrive_with(2, ClassId(1), 30_000, 4_000, prefix(2, 30_000), false);
    assert!(gate.preempted.is_empty());
    assert!(gate.was_dispatched(chat).is_none(), "it waits instead");
    gate.finish(agent, 2_000);
    assert!(gate.was_dispatched(chat).is_some());
}

#[test]
fn a_pause_that_loses_the_race_to_the_first_output_is_declined() {
    let mut gate = priorities().build();
    let agent = gate.arrive_with(1, ClassId(0), 80_000, 8_000, prefix(1, 80_000), false);
    gate.start_unannounced(agent);
    let chat = gate.arrive_with(2, ClassId(1), 30_000, 4_000, prefix(2, 30_000), false);
    assert!(gate.preempted.is_empty());
    assert!(gate.was_dispatched(chat).is_none());
    gate.start(agent);
    gate.finish(agent, 2_000);
    assert!(gate.was_dispatched(chat).is_some());
    assert_eq!(gate.dispatches(agent), 1);
}

#[test]
fn equal_priorities_and_protected_classes_are_never_paused() {
    let mut setup = priorities();
    setup.classes[0].preemptible = false;
    let mut gate = setup.build();
    let agent = gate.arrive_with(1, ClassId(0), 80_000, 8_000, prefix(1, 80_000), false);
    let chat = gate.arrive_with(2, ClassId(1), 30_000, 4_000, prefix(2, 30_000), false);
    assert!(gate.preempted.is_empty());
    assert!(gate.was_dispatched(chat).is_none());
    gate.finish(agent, 100);
    assert!(gate.was_dispatched(chat).is_some());
}

#[test]
fn a_request_is_paused_at_most_max_preemptions_times() {
    let mut setup = priorities();
    setup.max_preemptions = 1;
    let mut gate = setup.build();
    let agent = gate.arrive_with(1, ClassId(0), 80_000, 8_000, prefix(1, 80_000), false);
    let chat = gate.arrive_with(2, ClassId(1), 30_000, 4_000, prefix(2, 30_000), false);
    gate.finish(chat, 100);
    assert_eq!(gate.dispatches(agent), 2);
    let again = gate.arrive_with(3, ClassId(1), 30_000, 4_000, prefix(3, 30_000), false);
    assert_eq!(gate.preempted, vec![agent], "paused once only");
    assert!(gate.was_dispatched(again).is_none());
}

// ----------------------------------------------------------- partial reuse

#[test]
fn a_new_conversation_reuses_a_shared_system_prompt_and_the_old_context_is_kept() {
    let mut gate = Setup::unified(POOL, 1).build();
    // Conversation 1: a 20k system prompt (conversation 50) and 15k of its own.
    let first = gate.arrive(1, 35_000, 1_000, forked(50, 20_000, 1, 15_000));
    gate.finish(first, 500);
    // Conversation 2 starts with the same system prompt.
    let second = gate.arrive(2, 22_000, 1_000, forked(50, 20_000, 2, 2_000));
    let dispatch = gate.was_dispatched(second).expect("dispatched");
    assert_eq!(dispatch.slot, gate.was_dispatched(first).unwrap().slot);
    assert_eq!(
        dispatch.reuse, 19_968,
        "the shared blocks are reused in place"
    );
    assert_eq!(
        gate.files[0].values().copied().collect::<Vec<_>>(),
        vec![35_500],
        "conversation 1's context was saved before it was truncated"
    );
}

#[test]
fn a_snapshot_sharing_the_start_of_a_prompt_is_restored() {
    let mut gate = Setup::unified(POOL, 1).build();
    let first = gate.arrive(1, 60_000, 1_000, forked(50, 40_000, 1, 20_000));
    gate.finish(first, 100);
    let other = gate.arrive(2, 100_000, 1_000, prefix(2, 100_000));
    gate.finish(other, 100);
    // Conversation 3 shares the 40k head of conversation 1, now on disk.
    let third = gate.arrive(3, 45_000, 1_000, forked(50, 40_000, 3, 5_000));
    let dispatch = gate.was_dispatched(third).expect("dispatched");
    assert_eq!(dispatch.reuse, 40_000, "the shared head, 625 whole blocks");
}

#[test]
fn partial_reuse_is_given_up_on_a_server_that_cannot_rewind() {
    let mut gate = Setup::unified(POOL, 1).build();
    gate.cached = Some(0);
    for conversation in 0..4 {
        let head = gate.arrive(
            conversation,
            30_000,
            100,
            forked(50, 20_000, conversation + 100, 10_000),
        );
        gate.finish(head, 10);
    }
    assert_eq!(
        gate.notices,
        vec![Notice::PartialReuseIneffective {
            backend: BackendId(0)
        }]
    );
    let after = gate.arrive(9, 30_000, 100, forked(50, 20_000, 9, 10_000));
    assert_eq!(gate.was_dispatched(after).unwrap().reuse, 0);
}

// ------------------------------------------------- server-placed requests

#[test]
fn an_embedding_runs_alone_and_its_cells_are_swept_wherever_the_server_put_them() {
    let mut gate = Setup::unified(POOL, 2).build();
    let chat = gate.arrive(1, 10_000, 1_000, prefix(1, 10_000));
    let embedding = gate.arrive_unpinned(ModelId(0), 2, 5_000, Prefix::opaque(5_000));
    assert!(
        gate.was_dispatched(embedding).is_none(),
        "the server may put it in any idle slot, so it waits for an otherwise empty server"
    );
    gate.finish(chat, 100);
    let dispatch = gate.was_dispatched(embedding).expect("dispatched");
    assert_eq!(dispatch.generation, 0);
    assert_eq!(
        gate.files[0].len(),
        1,
        "the chat's context was saved to make way"
    );

    gate.auto = false;
    gate.finish(embedding, 0);
    let status = gate.scheduler.status(gate.now);
    assert_eq!(
        status.backends[0].stray, 5_000,
        "counted once, wherever it is"
    );
    let erases = gate
        .pending
        .iter()
        .filter(|command| matches!(command, Command::Evict { save: None, .. }))
        .count();
    assert_eq!(erases, 2, "every slot that may hold it is erased");
    gate.auto = true;
    gate.drain();
    assert_eq!(gate.scheduler.status(gate.now).backends[0].stray, 0);
    assert_eq!(gate.occupied(0), 0);
}

#[test]
fn requests_for_a_model_whose_servers_stay_down_are_refused_in_time() {
    let mut setup = Setup::unified(POOL, 1);
    setup.unavailable_after = Some(10_000);
    let mut gate = setup.build();
    gate.down(0);
    let ticket = gate.arrive(1, 1_000, 100, prefix(1, 1_000));
    gate.advance(5_000);
    assert!(gate.rejected.is_empty(), "it waits a while for the server");
    gate.advance(5_001);
    assert!(matches!(
        gate.rejected.as_slice(),
        [(refused, Rejection::Unavailable { .. })] if *refused == ticket
    ));
}

#[test]
fn a_new_configuration_keeps_everyone_in_line() {
    let mut setup = Setup::unified(POOL, 1);
    setup.classes = vec![class("batch", 0), class("interactive", 10)];
    let mut gate = setup.build();
    let running = gate.arrive_with(1, ClassId(0), 2_000, 500, prefix(1, 2_000), false);
    let batch = gate.arrive_with(1, ClassId(0), 2_000, 500, prefix(2, 2_000), false);
    let chat = gate.arrive_with(2, ClassId(1), 2_000, 500, prefix(3, 2_000), false);
    // The new configuration lists the classes the other way round and makes
    // batch the urgent one.
    gate.scheduler.reconfigure(
        gate_core::SchedulerConfig {
            classes: vec![class("interactive", 0), class("batch", 10)],
            backfill: None,
            eviction: EvictionPolicy::default(),
            max_preemptions: 3,
            unavailable_after: None,
        },
        |old| ClassId(1 - old.0),
    );
    gate.finish(running, 100);
    assert!(gate.was_dispatched(batch).is_some());
    assert!(gate.was_dispatched(chat).is_none());
}

/// A request for several answers at once (`n` = 2) is granted one slot, but
/// the server runs each answer in a slot of its own. The ledger must account
/// for every copy until those slots are erased, or a large request admitted
/// next overfills the pool (the copies are invisible to a one-slot booking).
#[test]
fn several_answers_at_once_leave_no_hidden_context() {
    let mut gate = Setup::unified(100_000, 2).build();
    let pair = gate.arrive_with(1, ClassId(0), 40_000, 1_000, prefix(1, 40_000), true);
    assert!(gate.was_dispatched(pair).is_some());
    // Erasing takes time: the next request arrives while it is under way.
    gate.auto = false;
    gate.finish_spread(pair, 1_000, &[0, 1]);
    let large = gate.arrive(2, 50_000, 10_000, prefix(2, 50_000));
    let granted = |gate: &support::Harness| {
        gate.pending
            .iter()
            .any(|command| matches!(command, Command::Dispatch { ticket, .. } if *ticket == large))
    };
    assert!(
        !granted(&gate),
        "waits while the server's copies may still fill the pool"
    );
    gate.drain();
    assert!(
        gate.was_dispatched(large).is_some(),
        "runs once they are erased"
    );
}

/// A request spread over several slots cannot be paused: stopping it frees
/// no single slot the waiting request could use.
#[test]
fn several_answers_at_once_are_not_paused() {
    let mut setup = Setup::unified(100_000, 2);
    let mut batch = class("batch", 0);
    batch.preemptible = true;
    setup.classes = vec![batch, class("interactive", 10)];
    let mut gate = setup.build();
    let pair = gate.arrive_with(1, ClassId(0), 40_000, 20_000, prefix(1, 40_000), true);
    assert!(gate.was_dispatched(pair).is_some());
    let urgent = gate.arrive_with(2, ClassId(1), 50_000, 5_000, prefix(2, 50_000), false);
    gate.drain();
    assert!(
        gate.was_dispatched(urgent).is_none(),
        "the spread request keeps running"
    );
    assert_eq!(gate.dispatches(pair), 1);
}

/// A context that finishes larger than its reservation (a prompt measured
/// short) means the pool may have overflowed while it ran. The gate says so
/// and reserves the excess on top of every later request.
#[test]
fn an_overrun_is_reported_and_reserved_against() {
    let mut gate = Setup::unified(200_000, 2).build();
    let first = gate.arrive(1, 50_000, 10_000, prefix(1, 50_000));
    // The server held 17,000 tokens more than was reserved.
    gate.finish(first, 27_000);
    assert!(gate.notices.iter().any(|notice| matches!(
        notice,
        Notice::Overran {
            reserved: 60_000,
            resident: 77_000,
            ..
        }
    )));
    let second = gate.arrive(2, 50_000, 10_000, prefix(2, 50_000));
    assert!(gate.was_dispatched(second).is_some());
    let usage = gate.scheduler.status(gate.now).backends[0].usage;
    assert_eq!(
        usage,
        77_000 + 77_000,
        "the idle first context plus a padded reservation"
    );
}

#[test]
fn hot_vram_beats_cold_disk_restore_when_token_gap_is_small() {
    // Pool holds 100k across 2 slots. Put a 3,136-token prefix in slot 0 and
    // a 2,880-token prefix in slot 1 (both sharing the same conversation head).
    // Evict slot 0 to disk via a large filler request, leaving slot 1 in hot
    // VRAM. A new 4,000-token continuation must choose hot VRAM (2,880 reuse)
    // over cold disk restore (3,136 reuse) because TTFT is clearly lower.
    let mut gate = Setup::unified(100_000, 2).build();
    let disk_seed = gate.arrive(1, 3_136, 100, prefix(99, 3_136));
    gate.finish(disk_seed, 64);
    gate.advance(10);
    let hot_seed = gate.arrive(2, 2_880, 100, prefix(99, 2_880));
    gate.finish(hot_seed, 64);
    let hot_slot = gate.was_dispatched(hot_seed).unwrap().slot;

    // Large filler evicts the older slot (disk_seed) and saves it to disk.
    let filler = gate.arrive(3, 95_000, 1_000, prefix(50, 95_000));
    gate.finish(filler, 100);
    assert_eq!(gate.files[0].len(), 1, "disk_seed was saved to disk");

    // 4,000-token continuation: 3,136 on disk vs 2,880 in hot VRAM.
    let query = gate.arrive(4, 4_000, 500, forked(99, 3_136, 7, 864));
    let dispatch = gate.was_dispatched(query).expect("dispatched");
    assert_eq!(dispatch.slot, hot_slot, "hot VRAM beats cold disk restore");
    assert_eq!(dispatch.reuse, 2_880);
}

#[test]
fn heterogeneous_replicas_route_to_the_faster_backend_and_reset_on_shape_change() {
    let mut gate = Setup::unified(POOL, 2).replicate().build();
    let r0 = gate.arrive(1, 5_000, 100, prefix(1, 5_000));
    let r1 = gate.arrive(2, 5_000, 100, prefix(2, 5_000));
    let fast_backend = gate.was_dispatched(r0).unwrap().slot.backend;
    let slow_backend = gate.was_dispatched(r1).unwrap().slot.backend;
    assert_ne!(fast_backend, slow_backend);

    gate.advance(500); // fast prefill: 5000 in 500ms = 10,000 tok/s
    gate.start(r0);
    gate.advance(500);
    gate.finish(r0, 100); // fast decode: 100 in 500ms = 200 tok/s

    gate.advance(9_000); // slow prefill: 5000 in 10,000ms = 500 tok/s
    gate.start(r1);
    gate.advance(5_000);
    gate.finish(r1, 100); // slow decode: 100 in 5,000ms = 20 tok/s

    // Fresh request with 4,000 tokens goes to fast_backend even though both are idle.
    let probe = gate.arrive(3, 4_000, 100, prefix(3, 4_000));
    assert_eq!(
        gate.was_dispatched(probe).unwrap().slot.backend,
        fast_backend
    );
    gate.finish(probe, 10);

    // Restarting fast_backend while a request is in-flight does not corrupt
    // telemetry; changing its shape resets its learned speed to the prior.
    let inflight = gate.arrive(4, 4_000, 100, prefix(4, 4_000));
    assert_eq!(
        gate.was_dispatched(inflight).unwrap().slot.backend,
        fast_backend
    );
    gate.advance(60_000);
    gate.restart(fast_backend.0);

    // Shape change resets fast_backend back to prior (2,000 tok/s); if we make
    // slow_backend fast (20,000 tok/s) and change fast_backend's shape,
    // slow_backend now wins.
    let changed_shape = Shape {
        model: ModelId(0),
        layout: Layout::Unified { capacity: POOL },
        slots: 3, // changed slot count
        swap: true,
    };
    gate.shapes[usize::from(fast_backend.0)] = changed_shape;
    gate.physical[usize::from(fast_backend.0)].resize(3, 0);
    gate.feed(Event::BackendUp {
        backend: fast_backend,
        shape: changed_shape,
        restarted: false,
        snapshots: None,
        now: gate.now,
    });

    // Re-calibrate slow_backend to 20,000 tok/s while fast_backend is full.
    let blocker = gate.arrive(5, POOL - 2_000, 1_000, prefix(5, (POOL - 2_000) as usize));
    let on_other = gate.arrive(6, 10_000, 100, prefix(6, 10_000));
    let other_backend = gate.was_dispatched(on_other).unwrap().slot.backend;
    gate.finish(blocker, 10);
    assert_eq!(other_backend, slow_backend);
    gate.advance(500); // 10,000 in 500ms = 20,000 tok/s
    gate.start(on_other);
    gate.advance(500);
    gate.finish(on_other, 100);
    let after = gate.arrive(7, 4_000, 100, prefix(7, 4_000));
    assert_eq!(
        gate.was_dispatched(after).unwrap().slot.backend,
        slow_backend
    );
}

#[test]
fn cache_aware_vtc_charges_cached_prefix_at_discounted_rate() {
    let mut gate = Setup::unified(150_000, 2).swap(true).build();
    let c1_t1 = gate.arrive(1, 50_000, 100, prefix(1, 50_000));
    gate.finish(c1_t1, 0); // c1 counter = 50,000
    let c2_t1 = gate.arrive(2, 50_000, 100, prefix(2, 50_000));
    gate.finish(c2_t1, 0); // c2 counter = 50,000

    // Client 1 turn 2: 51,000 prompt reusing 50,000 in slot 0 -> charged
    // 1,000 * 1.0 + 50,000 * 0.1 = 6,000 (total 56,000).
    let c1_t2 = gate.arrive(1, 51_000, 100, forked(1, 50_000, 10, 1_000));
    assert_eq!(gate.was_dispatched(c1_t2).unwrap().reuse, 50_000);
    // Queue Client 2's heavy uncached job (100k) and Client 1's next job (80k)
    // while c1_t2 runs so neither client goes idle.
    let c2_t2 = gate.arrive(2, 100_000, 100, prefix(20, 100_000));
    let c1_t3 = gate.arrive(1, 80_000, 100, prefix(11, 80_000));
    gate.finish(c1_t2, 0);

    // c2_t2 had counter 50,000 < 56,000 so it runs next and accrues 100,000
    // (total 150,000); c2_t3 arrives before c2_t2 finishes.
    assert_eq!(
        gate.scheduler.service(gate_core::ClientId(1)),
        Some(56_000.0)
    );
    assert!(gate.was_dispatched(c2_t2).is_some());
    let c2_t3 = gate.arrive(2, 80_000, 100, prefix(21, 80_000));
    gate.finish(c2_t2, 0);

    assert_eq!(
        gate.scheduler.service(gate_core::ClientId(2)),
        Some(150_000.0)
    );
    assert!(
        gate.was_dispatched(c1_t3).is_some(),
        "cached client has lower VTC counter and goes before heavy uncached client"
    );
    assert!(gate.was_dispatched(c2_t3).is_none());
    assert_eq!(
        gate.scheduler.service(gate_core::ClientId(1)),
        Some(136_000.0)
    );
}

#[test]
fn cost_benefit_eviction_protects_large_warm_prefix_over_stale_one() {
    let mut scheduler = Scheduler::new(
        SchedulerConfig {
            classes: vec![class("standard", 0)],
            backfill: None,
            eviction: EvictionPolicy::default(),
            max_preemptions: 3,
            unavailable_after: None,
        },
        Box::new(Vtc::default()),
        Box::new(CostBenefit),
    );
    let mut now = 0;
    let out = scheduler.handle(Event::BackendUp {
        backend: BackendId(0),
        shape: Shape {
            model: ModelId(0),
            layout: Layout::Unified { capacity: 100_000 },
            slots: 2,
            swap: false,
        },
        restarted: true,
        snapshots: None,
        now,
    });
    assert!(out.is_empty());

    let demand = |client: u32, prompt: u64, generation: u64, pfx: Prefix| Demand {
        client: gate_core::ClientId(client),
        class: ClassId(0),
        model: ModelId(0),
        prompt,
        work: prompt,
        generation,
        min_generation: 1,
        prefix: pfx,
        exclusive: false,
        pinned: true,
    };

    // Place 40k context in slot 0 and 20k context in slot 1 at t = 0.
    let _ = scheduler.handle(Event::Arrived {
        ticket: TicketId(1),
        demand: demand(1, 40_000, 100, prefix(1, 40_000)),
        now,
    });
    let _ = scheduler.handle(Event::Finished {
        ticket: TicketId(1),
        resident: Some(40_000),
        generated: Some(0),
        cached: None,
        now,
    });
    let _ = scheduler.handle(Event::Arrived {
        ticket: TicketId(2),
        demand: demand(2, 20_000, 100, prefix(2, 20_000)),
        now,
    });
    let _ = scheduler.handle(Event::Finished {
        ticket: TicketId(2),
        resident: Some(20_000),
        generated: Some(0),
        cached: None,
        now,
    });

    // Advance 10 minutes and refresh slot 1 so slot 0 is stale (40k / 11 ≈ 3.6k)
    // while slot 1 is warm (20k / 1 = 20k).
    now += 600_000;
    let _ = scheduler.handle(Event::Arrived {
        ticket: TicketId(3),
        demand: demand(2, 20_000, 100, prefix(2, 20_000)),
        now,
    });
    let _ = scheduler.handle(Event::Finished {
        ticket: TicketId(3),
        resident: Some(20_000),
        generated: Some(0),
        cached: None,
        now,
    });

    // Request needing 70k fits only by clearing/evicting one slot: stale slot 0
    // (40k) must be evicted while warm slot 1 (20k) stays!
    let cmds = scheduler.handle(Event::Arrived {
        ticket: TicketId(4),
        demand: demand(3, 70_000, 100, prefix(3, 70_000)),
        now,
    });
    assert!(
        cmds.iter().any(|cmd| matches!(
            cmd,
            Command::Evict { slot: s, .. } if *s == slot(0, 0)
        )),
        "stale slot 0 should be evicted before warm slot 1: {cmds:?}"
    );
}

#[test]
fn time_aware_easy_backfill_blocks_slow_candidates_near_window_expiry() {
    let mut setup = Setup::unified(100_000, 2).swap(false);
    setup.backfill = Some(Backfill {
        max_tokens: 20_000,
        window: 5_000,
    });
    let mut gate = setup.build();

    // Calibrate prefill_tps = 2,000 tok/s, decode_tps = 100 tok/s,
    // expected_generation = 200 tokens.
    let cal = gate.arrive(1, 4_000, 1_000, prefix(1, 4_000));
    gate.advance(2_000);
    gate.start(cal);
    gate.advance(2_000);
    gate.finish(cal, 200);

    // Running request occupies 61,000 tokens; head arrives while running is
    // still prefilling, then running starts decode 200ms before head's window
    // expires so at waited=4,800ms: remaining_window=200ms, shadow_ms=500ms
    // (200 expected tokens @ 100 tok/s = 2,000ms decode, 1,500ms elapsed).
    let running = gate.arrive(2, 60_000, 1_000, prefix(2, 60_000));
    gate.advance(25_200);
    let head = gate.arrive(3, 50_000, 1_000, prefix(3, 50_000));
    assert!(gate.was_dispatched(head).is_none());
    gate.advance(3_300);
    gate.start(running);
    gate.advance(1_500); // waited = 4,800ms (remaining_window = 200ms), shadow_ms = 500ms

    // Slow backfill (10,000 tokens, est runtime >> 500ms) must NOT stall head.
    let slow = gate.arrive(4, 10_000, 500, prefix(4, 10_000));
    assert!(
        gate.was_dispatched(slow).is_none(),
        "slow backfill near window expiry must wait behind head"
    );

    // Medium backfill (~250ms prefill*2 + 100ms decode? No, 100 prompt + 10 gen:
    // est = 100*1000/2000 * 2 + 10*1000/100 = 100 + 100 = 200ms <= 500ms)
    // and specifically a 150-token, 20-gen backfill (~150 + 200 = 350ms):
    // 200ms < 350ms <= 500ms proves shadow_ms.max(remaining_window) is used!
    let within_shadow = gate.arrive(5, 150, 20, prefix(5, 150));
    assert!(
        gate.was_dispatched(within_shadow).is_some(),
        "backfill within shadow_ms (350ms <= 500ms) should run"
    );

    // Hard KV-pool safety invariant: even though expected_generation learned 200,
    // a subsequent request with generation=8,000 still reserves full 8,000!
    gate.finish(within_shadow, 20);
    gate.finish(running, 200);
    assert!(gate.was_dispatched(head).is_some());
    gate.finish(head, 100);
    gate.finish(slow, 100);

    let full_reserve = gate.arrive(6, 50_000, 8_000, prefix(6, 50_000));
    let dispatch = gate.was_dispatched(full_reserve).unwrap();
    assert_eq!(dispatch.generation, 8_000);
    let slot_usage = gate.scheduler.status(gate.now).backends[0]
        .slots
        .iter()
        .find(|s| s.ticket == Some(full_reserve))
        .unwrap()
        .tokens;
    assert_eq!(slot_usage, 58_000);
}
