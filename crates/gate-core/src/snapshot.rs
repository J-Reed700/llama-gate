//! The index of slot contexts saved to disk.
//!
//! The index holds metadata only; the caller owns the files. It is
//! serializable so the caller can persist it and keep snapshots across
//! restarts. Recency is a logical clock rather than wall time, so ordering
//! survives a restart of the caller's monotonic clock.
use crate::{Prefix, SnapshotId, Tokens};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotMeta {
    pub id: SnapshotId,
    pub prefix: Prefix,
    pub tokens: Tokens,
    pub bytes: u64,
    /// Logical time of the last save or restore.
    pub used: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotIndex {
    next_id: u64,
    clock: u64,
    entries: BTreeMap<SnapshotId, SnapshotMeta>,
}

impl SnapshotIndex {
    /// Allocates an id for a snapshot about to be written.
    pub fn allocate(&mut self) -> SnapshotId {
        self.next_id += 1;
        SnapshotId(self.next_id)
    }

    /// Records a completed snapshot.
    pub fn insert(&mut self, id: SnapshotId, prefix: Prefix, tokens: Tokens, bytes: u64) {
        self.clock += 1;
        self.next_id = self.next_id.max(id.0);
        self.entries.insert(
            id,
            SnapshotMeta {
                id,
                prefix,
                tokens,
                bytes,
                used: self.clock,
            },
        );
    }

    pub fn remove(&mut self, id: SnapshotId) -> Option<SnapshotMeta> {
        self.entries.remove(&id)
    }

    /// Never allocates `id` or any id below it: a file by that name exists.
    pub fn skip_past(&mut self, id: SnapshotId) {
        self.next_id = self.next_id.max(id.0);
    }

    /// Marks a snapshot as just used.
    pub fn touch(&mut self, id: SnapshotId) {
        self.clock += 1;
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.used = self.clock;
        }
    }

    pub fn get(&self, id: SnapshotId) -> Option<&SnapshotMeta> {
        self.entries.get(&id)
    }

    /// The snapshot that lets the server reuse the most of `prefix`, with the
    /// tokens it covers. A snapshot that prefixes the request covers its whole
    /// context, the earlier answer included, which a continuing conversation
    /// repeats. With `partial`, a snapshot sharing only the start of the
    /// request (a common system prompt, a rewound conversation) qualifies too,
    /// covering the blocks they share.
    pub fn best_for(&self, prefix: &Prefix, partial: bool) -> Option<(&SnapshotMeta, Tokens)> {
        self.entries
            .values()
            .filter_map(|entry| {
                if entry.prefix.is_prefix_of(prefix) {
                    return Some((entry, entry.tokens));
                }
                let shared = if partial {
                    entry.prefix.shared(prefix)
                } else {
                    0
                };
                (shared > 0).then_some((entry, shared))
            })
            // Among equals, the smallest restores fastest.
            .max_by_key(|(entry, covered)| (*covered, std::cmp::Reverse(entry.tokens), entry.used))
    }

    /// Least recently used snapshots to delete so the total fits `budget`.
    pub fn over_budget(&self, budget: u64) -> Vec<SnapshotId> {
        let mut total = self.bytes();
        let mut by_age: Vec<&SnapshotMeta> = self.entries.values().collect();
        by_age.sort_by_key(|entry| entry.used);
        let mut doomed = Vec::new();
        for entry in by_age {
            if total <= budget {
                break;
            }
            total = total.saturating_sub(entry.bytes);
            doomed.push(entry.id);
        }
        doomed
    }

    pub fn bytes(&self) -> u64 {
        self.entries.values().map(|entry| entry.bytes).sum()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &SnapshotMeta> {
        self.entries.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefix(length: usize) -> Prefix {
        Prefix::from_tokens(&(0..length as i32).collect::<Vec<_>>(), 64)
    }

    #[test]
    fn the_longest_snapshot_that_prefixes_the_request_wins() {
        let mut index = SnapshotIndex::default();
        let short = index.allocate();
        index.insert(short, prefix(256), 256, 10);
        let long = index.allocate();
        index.insert(long, prefix(640), 640, 10);
        let unrelated = index.allocate();
        index.insert(
            unrelated,
            Prefix::from_tokens(&(1000..1900).collect::<Vec<_>>(), 64),
            900,
            10,
        );
        let best =
            |prefix: &Prefix, partial| index.best_for(prefix, partial).map(|(e, n)| (e.id, n));
        assert_eq!(best(&prefix(1024), false), Some((long, 640)));
        assert_eq!(best(&prefix(300), false), Some((short, 256)));
        assert_eq!(best(&prefix(100), false), None);
        // Partially, the long snapshot covers the same blocks as the short
        // one; the smaller is cheaper to restore.
        assert_eq!(best(&prefix(300), true), Some((short, 256)));
        assert_eq!(best(&prefix(100), true), Some((short, 64)));
        assert_eq!(best(&prefix(1024), true), Some((long, 640)));
    }

    #[test]
    fn the_budget_evicts_least_recently_used_first() {
        let mut index = SnapshotIndex::default();
        let ids: Vec<_> = (0..3)
            .map(|step| {
                let id = index.allocate();
                index.insert(id, prefix(64 * (step + 1)), 64, 100);
                id
            })
            .collect();
        index.touch(ids[0]);
        assert_eq!(index.over_budget(200), vec![ids[1]]);
        assert!(index.over_budget(300).is_empty());
    }

    #[test]
    fn the_index_round_trips_through_json() {
        let mut index = SnapshotIndex::default();
        let id = index.allocate();
        index.insert(id, prefix(128), 128, 42);
        let restored: SnapshotIndex =
            serde_json::from_str(&serde_json::to_string(&index).unwrap()).unwrap();
        assert_eq!(restored, index);
        let mut restored = restored;
        assert!(
            restored.allocate() > id,
            "ids are never reused after a reload"
        );
    }
}
