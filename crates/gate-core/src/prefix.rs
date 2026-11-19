//! Content addresses for token sequences.
use crate::Tokens;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// The content address of a token sequence: one hash per full block of
/// tokens, each chained to the previous block's hash. This is the scheme
/// vLLM uses for automatic prefix caching. Two sequences share their first
/// `k` blocks exactly when their first `k` hashes are equal, so prefixes can
/// be compared (and persisted) without keeping the tokens.
///
/// A trailing partial block is counted in [`Prefix::tokens`] but not hashed;
/// the server re-checks those few tokens itself when it reuses a cache.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prefix {
    block: u32,
    tokens: Tokens,
    hashes: Arc<[u64]>,
}

/// Separates this hashing scheme from any future one in persisted indexes.
const DOMAIN: &[u8] = b"llama-gate/prefix/v1";

impl Prefix {
    /// Hashes `tokens` in blocks of `block` tokens.
    pub fn from_tokens(tokens: &[i32], block: usize) -> Self {
        Self::from_tokens_in("", tokens, block)
    }

    /// Hashes `tokens` in blocks of `block` tokens within `namespace`, such
    /// as the model's identity: sequences from different namespaces never
    /// match, so a context saved under one model is never offered to
    /// another (llm-d seeds its block hashes the same way).
    pub fn from_tokens_in(namespace: &str, tokens: &[i32], block: usize) -> Self {
        let block = block.clamp(1, u32::MAX as usize);
        let mut parent: [u8; 32] = Sha256::new()
            .chain_update(DOMAIN)
            .chain_update((block as u32).to_le_bytes())
            .chain_update((namespace.len() as u64).to_le_bytes())
            .chain_update(namespace.as_bytes())
            .finalize()
            .into();
        let mut hashes = Vec::with_capacity(tokens.len() / block);
        for chunk in tokens.chunks_exact(block) {
            let mut hasher = Sha256::new();
            hasher.update(parent);
            for token in chunk {
                hasher.update(token.to_le_bytes());
            }
            parent = hasher.finalize().into();
            let mut head = [0u8; 8];
            head.copy_from_slice(&parent[..8]);
            hashes.push(u64::from_le_bytes(head));
        }
        Self {
            block: block as u32,
            tokens: tokens.len() as Tokens,
            hashes: hashes.into(),
        }
    }

    /// A sequence of known length but unknown content, such as a context the
    /// server already held when the gate started. It matches nothing.
    pub fn opaque(tokens: Tokens) -> Self {
        Self {
            block: 0,
            tokens,
            hashes: Arc::from([]),
        }
    }

    /// Length of the sequence in tokens.
    pub fn tokens(&self) -> Tokens {
        self.tokens
    }

    /// Whether the content of this sequence is unknown.
    pub fn is_opaque(&self) -> bool {
        self.block == 0
    }

    /// Tokens covered by the full blocks the two sequences have in common.
    ///
    /// Each hash covers its block and everything before it, so the blocks in
    /// common are a run from the start, found by binary search: O(log n)
    /// rather than a walk over every block of a long context.
    pub fn shared(&self, other: &Prefix) -> Tokens {
        if self.is_opaque() || self.block != other.block {
            return 0;
        }
        // The largest k whose k-th hashes agree (agreement is monotone: once
        // two chains differ, every later block differs too).
        let (mut low, mut high) = (0, self.hashes.len().min(other.hashes.len()));
        while low < high {
            let middle = low + (high - low).div_ceil(2);
            if self.hashes[middle - 1] == other.hashes[middle - 1] {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        low as Tokens * Tokens::from(self.block)
    }

    /// Whether `self` is a prefix of `other`: every full block of `self`
    /// matches, and `other` is at least as long. The last block's hash
    /// covers all the others, so one comparison decides.
    pub fn is_prefix_of(&self, other: &Prefix) -> bool {
        if self.is_opaque() || self.block != other.block || self.tokens > other.tokens {
            return false;
        }
        match self.hashes.len().checked_sub(1) {
            None => true,
            Some(last) => other.hashes.get(last) == Some(&self.hashes[last]),
        }
    }

    /// The first `tokens` tokens of this sequence, such as the part of a
    /// prompt a server processed before it was stopped.
    pub fn truncate(&self, tokens: Tokens) -> Prefix {
        if tokens >= self.tokens {
            return self.clone();
        }
        if self.is_opaque() {
            return Prefix::opaque(tokens);
        }
        let blocks = (tokens / Tokens::from(self.block)) as usize;
        Self {
            block: self.block,
            tokens,
            hashes: self.hashes[..blocks.min(self.hashes.len())].into(),
        }
    }

    /// A short stable fingerprint, for file names and logs.
    pub fn fingerprint(&self) -> String {
        match self.hashes.last() {
            Some(hash) => format!("{hash:016x}"),
            None => format!("{:016x}", self.tokens),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sequence(length: usize, seed: i32) -> Vec<i32> {
        (0..length as i32).map(|value| value * 31 + seed).collect()
    }

    #[test]
    fn shared_blocks_end_at_the_first_difference() {
        let base = sequence(1000, 7);
        let mut fork = base.clone();
        fork[530] += 1;
        let left = Prefix::from_tokens(&base, 128);
        let right = Prefix::from_tokens(&fork, 128);
        assert_eq!(
            left.shared(&right),
            512,
            "blocks 0..4 match, block 4 differs"
        );
        assert_eq!(left.shared(&left), 896, "only full blocks are hashed");
    }

    #[test]
    fn shared_and_is_prefix_of_agree_with_the_tokens_everywhere() {
        // Every pair of lengths and every point of difference, against the
        // definition: full blocks before the first differing token.
        let block = 4;
        let base = sequence(41, 11);
        for left_length in 0..=base.len() {
            for right_length in 0..=base.len() {
                for differ in 0..=right_length {
                    let left = &base[..left_length];
                    let mut right = base[..right_length].to_vec();
                    if let Some(token) = right.get_mut(differ) {
                        *token += 1;
                    }
                    let common = left.iter().zip(&right).take_while(|(a, b)| a == b).count();
                    let (lp, rp) = (
                        Prefix::from_tokens(left, block),
                        Prefix::from_tokens(&right, block),
                    );
                    assert_eq!(
                        lp.shared(&rp),
                        (common / block * block) as Tokens,
                        "lengths {left_length}/{right_length}, differing at {differ}"
                    );
                    assert_eq!(
                        lp.is_prefix_of(&rp),
                        left_length <= right_length && common >= left_length / block * block,
                        "lengths {left_length}/{right_length}, differing at {differ}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_conversation_is_a_prefix_of_its_continuation() {
        let turn = sequence(700, 3);
        let mut next = turn.clone();
        next.extend(sequence(300, 99));
        let turn = Prefix::from_tokens(&turn, 128);
        let next = Prefix::from_tokens(&next, 128);
        assert!(turn.is_prefix_of(&next));
        assert!(!next.is_prefix_of(&turn));
        assert!(turn.is_prefix_of(&turn));
    }

    #[test]
    fn opaque_and_differently_blocked_prefixes_match_nothing() {
        let tokens = sequence(512, 1);
        let opaque = Prefix::opaque(512);
        let coarse = Prefix::from_tokens(&tokens, 256);
        let fine = Prefix::from_tokens(&tokens, 128);
        assert_eq!(opaque.shared(&fine), 0);
        assert!(!opaque.is_prefix_of(&fine));
        assert_eq!(coarse.shared(&fine), 0);
        assert!(!coarse.is_prefix_of(&fine));
    }

    #[test]
    fn namespaces_never_match_each_other() {
        let tokens = sequence(256, 5);
        let left = Prefix::from_tokens_in("model-a", &tokens, 64);
        let right = Prefix::from_tokens_in("model-b", &tokens, 64);
        assert_eq!(left.shared(&right), 0);
        assert_eq!(left.shared(&left), 256);
    }

    #[test]
    fn a_truncated_sequence_is_a_prefix_of_the_whole() {
        let tokens = sequence(1000, 4);
        let whole = Prefix::from_tokens(&tokens, 128);
        let part = whole.truncate(600);
        assert_eq!(part.tokens(), 600);
        assert_eq!(part, Prefix::from_tokens(&tokens[..600], 128));
        assert!(part.is_prefix_of(&whole));
        assert_eq!(whole.truncate(5000), whole);
        assert!(Prefix::opaque(900).truncate(300).is_opaque());
    }

    #[test]
    fn hashing_is_stable_across_runs() {
        let prefix = Prefix::from_tokens(&[1, 2, 3, 4], 2);
        assert_eq!(
            prefix.fingerprint(),
            Prefix::from_tokens(&[1, 2, 3, 4], 2).fingerprint()
        );
        let round_trip: Prefix =
            serde_json::from_str(&serde_json::to_string(&prefix).unwrap()).unwrap();
        assert_eq!(round_trip, prefix);
    }
}
