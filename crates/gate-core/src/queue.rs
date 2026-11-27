//! Ordering policies for waiting requests.
//!
//! The scheduler applies class priority first; a [`QueueOrder`] only orders
//! requests of equal priority. Implement the trait to plug in another policy.
use crate::{Class, ClientId, Tokens};
use std::collections::HashMap;

/// A waiting request, as an ordering policy sees it.
#[derive(Clone, Copy, Debug)]
pub struct Candidate<'a> {
    pub client: ClientId,
    pub class: &'a Class,
    /// Prompt plus reserved generation.
    pub need: Tokens,
    /// How long it has waited so far.
    pub waited: crate::Millis,
}

/// Decides which waiting request of equal priority is offered a slot next.
///
/// The scheduler sorts by `(priority descending, key ascending, arrival)`.
/// Implementations must be deterministic: the same sequence of calls must
/// always produce the same keys.
pub trait QueueOrder: Send + std::fmt::Debug {
    /// Sort key for a candidate; lower is served first.
    fn key(&self, candidate: Candidate<'_>) -> f64;

    /// `client` just went from no waiting requests to at least one, while
    /// `backlogged` clients (excluding `client`) were already waiting.
    fn activated(&mut self, _client: ClientId, _backlogged: &[ClientId]) {}

    /// `client` no longer has any waiting requests.
    fn deactivated(&mut self, _client: ClientId) {}

    /// A request of `client` started running with `tokens` of prompt.
    fn served_prompt(&mut self, _client: ClientId, _class: &Class, _tokens: Tokens) {}

    /// A request of `client` started running with `tokens` of cached prompt
    /// reused from a slot or snapshot.
    fn served_cached_prompt(&mut self, client: ClientId, class: &Class, tokens: Tokens) {
        self.served_prompt(client, class, tokens);
    }

    /// A request of `client` finished after generating `tokens`.
    fn served_generation(&mut self, _client: ClientId, _class: &Class, _tokens: Tokens) {}

    /// The service `client` has received so far as this policy counts it,
    /// for status pages. `None` when the policy keeps no such count.
    fn service(&self, _client: ClientId) -> Option<f64> {
        None
    }
}

/// First come, first served (within a priority).
#[derive(Clone, Copy, Debug, Default)]
pub struct Fifo;

impl QueueOrder for Fifo {
    fn key(&self, _candidate: Candidate<'_>) -> f64 {
        0.0
    }
}

/// Virtual Token Counter fair queueing (Sheng et al., "Fairness in Serving
/// Large Language Models", OSDI 2024).
///
/// Every client accrues a counter of the service it has received: prompt
/// tokens weighted by `input`, generated tokens by `output`, both divided by
/// its class weight. The waiting client with the smallest counter goes next.
/// When a client starts waiting again after being idle, its counter is lifted
/// to the smallest counter among clients already waiting (or, if nobody is
/// waiting, to the counter of the last client to leave the queue), so idle
/// time cannot be banked and spent later to starve others. llm-d's
/// `weighted-fair-share` flow control uses the same rule.
#[derive(Clone, Debug)]
pub struct Vtc {
    input: f64,
    cached_input: f64,
    output: f64,
    counters: HashMap<ClientId, f64>,
    last_departed: Option<f64>,
}

impl Vtc {
    /// The paper's defaults weigh a generated token twice a prompt token,
    /// reflecting that decode costs more GPU time per token than prefill.
    /// Cached prompt tokens default to 10% of `input`.
    pub fn new(input: f64, output: f64) -> Self {
        Self::with_cached(input, input * 0.1, output)
    }

    /// Like [`Vtc::new`], with an explicit weight for cached prompt tokens.
    pub fn with_cached(input: f64, cached_input: f64, output: f64) -> Self {
        Self {
            input,
            cached_input,
            output,
            counters: HashMap::new(),
            last_departed: None,
        }
    }

    /// The service counter of `client`.
    pub fn counter(&self, client: ClientId) -> f64 {
        self.counters.get(&client).copied().unwrap_or(0.0)
    }

    fn charge(&mut self, client: ClientId, class: &Class, cost: f64) {
        let weight = if class.weight > 0.0 {
            class.weight
        } else {
            1.0
        };
        *self.counters.entry(client).or_insert(0.0) += cost / weight;
    }
}

impl Default for Vtc {
    fn default() -> Self {
        Self::new(1.0, 2.0)
    }
}

impl QueueOrder for Vtc {
    fn key(&self, candidate: Candidate<'_>) -> f64 {
        self.counter(candidate.client)
    }

    fn activated(&mut self, client: ClientId, backlogged: &[ClientId]) {
        let floor = if backlogged.is_empty() {
            self.last_departed
        } else {
            Some(
                backlogged
                    .iter()
                    .map(|other| self.counter(*other))
                    .fold(f64::INFINITY, f64::min),
            )
        };
        let counter = self.counters.entry(client).or_insert(0.0);
        if let Some(floor) = floor.filter(|floor| floor.is_finite())
            && *counter < floor
        {
            *counter = floor;
        }
    }

    fn deactivated(&mut self, client: ClientId) {
        self.last_departed = Some(self.counter(client));
    }

    fn served_prompt(&mut self, client: ClientId, class: &Class, tokens: Tokens) {
        self.charge(client, class, self.input * tokens as f64);
    }

    fn served_cached_prompt(&mut self, client: ClientId, class: &Class, tokens: Tokens) {
        self.charge(client, class, self.cached_input * tokens as f64);
    }

    fn served_generation(&mut self, client: ClientId, class: &Class, tokens: Tokens) {
        self.charge(client, class, self.output * tokens as f64);
    }

    fn service(&self, client: ClientId) -> Option<f64> {
        Some(self.counter(client))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_least_served_client_goes_first() {
        let class = Class::new("standard");
        let mut vtc = Vtc::default();
        vtc.served_prompt(ClientId(1), &class, 10_000);
        vtc.served_generation(ClientId(1), &class, 1_000);
        let candidate = |client| Candidate {
            client: ClientId(client),
            class: &class,
            need: 0,
            waited: 0,
        };
        let heavy = vtc.key(candidate(1));
        let light = vtc.key(candidate(2));
        assert_eq!(heavy, 12_000.0);
        assert!(light < heavy);
    }

    #[test]
    fn idle_time_cannot_be_banked() {
        let class = Class::new("standard");
        let mut vtc = Vtc::default();
        vtc.served_prompt(ClientId(1), &class, 50_000);
        // Client 2 was idle with a zero counter; when it returns while client
        // 1 is waiting, it starts level with client 1, not 50k tokens ahead.
        vtc.activated(ClientId(2), &[ClientId(1)]);
        assert_eq!(vtc.counter(ClientId(2)), 50_000.0);
        // A client already ahead is never pulled back.
        vtc.served_prompt(ClientId(3), &class, 90_000);
        vtc.activated(ClientId(3), &[ClientId(1)]);
        assert_eq!(vtc.counter(ClientId(3)), 90_000.0);
    }

    #[test]
    fn a_client_returning_to_an_empty_queue_starts_level_with_the_last_to_leave() {
        let class = Class::new("standard");
        let mut vtc = Vtc::default();
        vtc.served_prompt(ClientId(1), &class, 30_000);
        vtc.deactivated(ClientId(1));
        vtc.activated(ClientId(2), &[]);
        assert_eq!(vtc.counter(ClientId(2)), 30_000.0);
    }

    #[test]
    fn class_weight_scales_the_charge() {
        let mut heavy = Class::new("interactive");
        heavy.weight = 4.0;
        let mut vtc = Vtc::default();
        vtc.served_prompt(ClientId(1), &heavy, 4_000);
        assert_eq!(vtc.counter(ClientId(1)), 1_000.0);
    }

    #[test]
    fn cached_prompt_charges_discounted_weight_and_custom_policies_delegate() {
        let class = Class::new("standard");
        let mut vtc = Vtc::new(1.0, 2.0);
        vtc.served_prompt(ClientId(1), &class, 1_000);
        vtc.served_cached_prompt(ClientId(1), &class, 99_000);
        assert!((vtc.counter(ClientId(1)) - 10_900.0).abs() < 1e-6);

        let mut custom = Vtc::with_cached(1.0, 0.25, 2.0);
        custom.served_cached_prompt(ClientId(2), &class, 4_000);
        assert!((custom.counter(ClientId(2)) - 1_000.0).abs() < 1e-6);

        #[derive(Debug, Default)]
        struct LegacyOrder {
            charged: Tokens,
        }
        impl QueueOrder for LegacyOrder {
            fn key(&self, _candidate: Candidate<'_>) -> f64 {
                0.0
            }
            fn served_prompt(&mut self, _client: ClientId, _class: &Class, tokens: Tokens) {
                self.charged += tokens;
            }
        }
        let mut legacy = LegacyOrder::default();
        legacy.served_prompt(ClientId(1), &class, 500);
        legacy.served_cached_prompt(ClientId(1), &class, 1_500);
        assert_eq!(legacy.charged, 2_000);
    }
}
