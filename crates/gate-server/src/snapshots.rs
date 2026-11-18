//! Snapshot files: naming, the persisted index, and its integrity.
//!
//! The llama.cpp server writes and reads the files (in its
//! `--slot-save-path`); the gate names them, keeps their index, and deletes
//! them. Three rules keep the store whole whatever happens to the process:
//!
//! - One gate owns a directory at a time (an advisory lock held for the
//!   gate's life), and every change the gate makes to it is serialized.
//! - The index is replaced atomically and durably (write a temporary file,
//!   sync it, rename it over the index, sync the directory), so a crash at
//!   any moment leaves either the previous index or the new one.
//! - The index decides what exists. Loading it drops entries whose files are
//!   gone and removes the gate's files it does not list (a save still in
//!   flight when the gate stopped, a server that died mid-write): no file
//!   the gate can never restore is left taking space.
//!
//! The index records the model identity it belongs to, so snapshots from
//! another model are never offered for restore.
use gate_core::{SnapshotId, SnapshotIndex};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;
use tokio::sync::watch;
use tracing::warn;

const INDEX: &str = "llama-gate-index.json";
const LOCK: &str = "llama-gate.lock";
const PREFIX: &str = "llama-gate-";
const SUFFIX: &str = ".slot";
const VERSION: u32 = 1;

/// One backend's snapshot directory, owned by this gate.
#[derive(Clone, Debug)]
pub struct SnapshotStore {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    directory: PathBuf,
    /// Serializes every change the gate makes to the directory.
    io: Mutex<()>,
    /// Held for the gate's life: the advisory lock is released with it.
    _lock: std::fs::File,
}

/// Why a directory cannot be used for snapshots.
#[derive(Debug)]
pub enum OpenError {
    NotADirectory,
    /// Another process (another gate) holds the directory's lock.
    Locked,
    Io(std::io::Error),
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotADirectory => formatter.write_str("it is not an accessible directory"),
            Self::Locked => formatter.write_str("another llama-gate is using it"),
            Self::Io(error) => write!(formatter, "{error}"),
        }
    }
}

/// An index as loaded, with what loading it had to repair.
#[derive(Debug, Default)]
pub struct Loaded {
    pub index: SnapshotIndex,
    /// The index could not be used as it was, and why: the gate starts over.
    pub started_over: Option<StartOver>,
    /// Entries dropped because their files are gone.
    pub dropped: usize,
    /// The gate's files removed because no entry listed them, and their size.
    pub removed: usize,
    pub removed_bytes: u64,
}

/// Why a loaded index was not used.
#[derive(Debug, PartialEq, Eq)]
pub enum StartOver {
    /// There was none: a new directory, or the first run of this gate.
    Missing,
    /// It belongs to another model; its snapshots are useless here.
    OtherModel(String),
    /// Written by an incompatible version of the gate.
    OtherVersion(u32),
    /// It does not parse. It is kept aside, as `*.corrupt`, for inspection.
    Corrupt,
    /// It could not be read; the files are left alone until it can be.
    Unreadable(String),
}

#[derive(Serialize, Deserialize)]
struct IndexFile {
    version: u32,
    namespace: String,
    index: SnapshotIndex,
}

impl SnapshotStore {
    /// Takes ownership of `directory` for this gate's life.
    pub fn open(directory: &Path) -> Result<Self, OpenError> {
        if !directory.is_dir() {
            return Err(OpenError::NotADirectory);
        }
        let lock = std::fs::File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(directory.join(LOCK))
            .map_err(OpenError::Io)?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => return Err(OpenError::Locked),
            Err(std::fs::TryLockError::Error(error)) => return Err(OpenError::Io(error)),
        }
        Ok(Self {
            inner: Arc::new(Inner {
                directory: directory.to_owned(),
                io: Mutex::new(()),
                _lock: lock,
            }),
        })
    }

    pub fn directory(&self) -> &Path {
        &self.inner.directory
    }

    /// The file name the server is asked to use for a snapshot. Only
    /// characters llama.cpp's filename validation accepts.
    pub fn filename(id: SnapshotId) -> String {
        format!("{PREFIX}{:016}{SUFFIX}", id.0)
    }

    /// Loads the index for `namespace` and reconciles it with the files.
    pub fn load(&self, namespace: &str) -> Loaded {
        let _io = self.io();
        let directory = &self.inner.directory;
        let mut loaded = Loaded::default();
        match std::fs::read(directory.join(INDEX)) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                loaded.started_over = Some(StartOver::Missing);
            }
            Err(error) => {
                // Possibly transient: start empty, delete nothing. The files
                // are reconciled at a later load that can read the index.
                loaded.started_over = Some(StartOver::Unreadable(error.to_string()));
                return loaded;
            }
            Ok(bytes) => match serde_json::from_slice::<IndexFile>(&bytes) {
                Err(_) => {
                    let aside = directory.join(format!("{INDEX}.corrupt"));
                    if let Err(error) = std::fs::rename(directory.join(INDEX), &aside) {
                        warn!(%error, "could not set the unreadable snapshot index aside");
                    }
                    loaded.started_over = Some(StartOver::Corrupt);
                }
                Ok(file) if file.version != VERSION => {
                    loaded.started_over = Some(StartOver::OtherVersion(file.version));
                }
                Ok(file) if file.namespace != namespace => {
                    loaded.started_over = Some(StartOver::OtherModel(file.namespace));
                }
                Ok(file) => loaded.index = file.index,
            },
        }
        let listed: BTreeSet<String> = loaded
            .index
            .iter()
            .map(|meta| Self::filename(meta.id))
            .collect();
        let mut present = BTreeSet::new();
        for (name, path, size) in gate_files(directory) {
            if is_temporary(&name) || (is_snapshot(&name) && !listed.contains(&name)) {
                if remove(&path) {
                    if is_snapshot(&name) {
                        loaded.removed += 1;
                        loaded.removed_bytes += size;
                    }
                } else if let Some(id) = snapshot_id(&name) {
                    // Still there: never hand out its id again.
                    loaded.index.skip_past(id);
                }
            } else if is_snapshot(&name) {
                present.insert(name);
            }
        }
        let missing: Vec<SnapshotId> = loaded
            .index
            .iter()
            .filter(|meta| !present.contains(&Self::filename(meta.id)))
            .map(|meta| meta.id)
            .collect();
        loaded.dropped = missing.len();
        for id in missing {
            loaded.index.remove(id);
        }
        loaded
    }

    /// Replaces the index atomically and durably: after a crash at any
    /// point, the directory holds either the previous index or this one.
    pub fn persist(&self, namespace: &str, index: &SnapshotIndex) -> std::io::Result<()> {
        let bytes = serde_json::to_vec(&IndexFile {
            version: VERSION,
            namespace: namespace.to_owned(),
            index: index.clone(),
        })
        .map_err(std::io::Error::other)?;
        let _io = self.io();
        let directory = &self.inner.directory;
        let temporary = directory.join(format!("{INDEX}.{}.tmp", std::process::id()));
        let written = (|| {
            let mut file = std::fs::File::create(&temporary)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            std::fs::rename(&temporary, directory.join(INDEX))?;
            sync_directory(directory)
        })();
        if written.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        written
    }

    /// Deletes a snapshot's file. A file already gone is not an error.
    pub fn delete(&self, id: SnapshotId) {
        let _io = self.io();
        remove(&self.inner.directory.join(Self::filename(id)));
    }

    fn io(&self) -> MutexGuard<'_, ()> {
        // The guard protects no data, so a panic elsewhere cannot leave
        // anything half-done behind it.
        self.inner
            .io
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Writes one backend's index in the background, one write at a time, the
/// latest version only: indexes that change faster than the disk can take
/// them are coalesced, never interleaved.
#[derive(Debug)]
pub struct Persister {
    pending: watch::Sender<Option<Arc<Pending>>>,
    written: watch::Receiver<u64>,
    submitted: u64,
}

#[derive(Debug)]
struct Pending {
    sequence: u64,
    namespace: String,
    index: SnapshotIndex,
}

impl Persister {
    pub fn spawn(store: SnapshotStore, backend: String) -> Self {
        let (pending, mut next) = watch::channel(None::<Arc<Pending>>);
        let (written_sender, written) = watch::channel(0);
        tokio::spawn(async move {
            while next.changed().await.is_ok() {
                let Some(work) = next.borrow_and_update().clone() else {
                    continue;
                };
                let store = store.clone();
                let sequence = work.sequence;
                let result = tokio::task::spawn_blocking(move || {
                    store.persist(&work.namespace, &work.index)
                })
                .await;
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        warn!(backend, %error, "could not persist the snapshot index")
                    }
                    Err(error) => {
                        warn!(backend, %error, "persisting the snapshot index failed")
                    }
                }
                // Written or failed, this version is settled.
                written_sender.send_replace(sequence);
            }
        });
        Self {
            pending,
            written,
            submitted: 0,
        }
    }

    /// Queues `index` to be written, replacing any version not yet written.
    pub fn submit(&mut self, namespace: String, index: SnapshotIndex) {
        self.submitted += 1;
        self.pending.send_replace(Some(Arc::new(Pending {
            sequence: self.submitted,
            namespace,
            index,
        })));
    }

    /// Waits, up to `limit`, until the last submitted version is settled.
    pub async fn flush(&mut self, limit: Duration) -> bool {
        let target = self.submitted;
        tokio::time::timeout(limit, self.written.wait_for(|written| *written >= target))
            .await
            .is_ok_and(|settled| settled.is_ok())
    }
}

/// Makes a rename in `directory` durable: on Unix the new name is on disk
/// only once the directory itself is synced. (Windows offers no handle to a
/// directory for this; its renames are journaled by the filesystem.)
fn sync_directory(directory: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    std::fs::File::open(directory)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = directory;
    Ok(())
}

/// The gate's files in `directory`: name, path, and size.
fn gate_files(directory: &Path) -> Vec<(String, PathBuf, u64)> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.starts_with(PREFIX).then(|| {
                let size = entry.metadata().map_or(0, |metadata| metadata.len());
                (name, entry.path(), size)
            })
        })
        .collect()
}

fn is_snapshot(name: &str) -> bool {
    snapshot_id(name).is_some()
}

fn snapshot_id(name: &str) -> Option<SnapshotId> {
    let digits = name.strip_prefix(PREFIX)?.strip_suffix(SUFFIX)?;
    (!digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| digits.parse().ok().map(SnapshotId))
        .flatten()
}

/// An index write that never finished (this process's, or a crashed one's).
fn is_temporary(name: &str) -> bool {
    name.starts_with(INDEX) && name.ends_with(".tmp")
}

/// Removes a file; true if it is gone (or was never there).
fn remove(path: &Path) -> bool {
    match std::fs::remove_file(path) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
        Err(error) => {
            warn!(path = %path.display(), %error, "could not remove a snapshot file");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gate_core::Prefix;

    fn index_with(ids: usize) -> (SnapshotIndex, Vec<SnapshotId>) {
        let mut index = SnapshotIndex::default();
        let ids = (0..ids)
            .map(|n| {
                let id = index.allocate();
                index.insert(id, Prefix::from_tokens(&[n as i32, 2, 3, 4], 2), 4, 100);
                id
            })
            .collect();
        (index, ids)
    }

    fn touch(directory: &Path, id: SnapshotId) -> PathBuf {
        let path = directory.join(SnapshotStore::filename(id));
        std::fs::write(&path, b"state").unwrap();
        path
    }

    #[test]
    fn the_index_survives_a_restart_and_drops_missing_files() {
        let directory = tempfile::tempdir().unwrap();
        let store = SnapshotStore::open(directory.path()).unwrap();
        let (index, ids) = index_with(2);
        touch(directory.path(), ids[0]);
        store.persist("model-a", &index).unwrap();

        let loaded = store.load("model-a");
        assert_eq!(loaded.started_over, None);
        assert!(loaded.index.get(ids[0]).is_some());
        assert!(loaded.index.get(ids[1]).is_none(), "its file is gone");
        assert_eq!(loaded.dropped, 1);
    }

    #[test]
    fn files_the_index_does_not_list_are_removed() {
        let directory = tempfile::tempdir().unwrap();
        let store = SnapshotStore::open(directory.path()).unwrap();
        let (index, ids) = index_with(1);
        let kept = touch(directory.path(), ids[0]);
        let orphan = touch(directory.path(), SnapshotId(77));
        let unfinished = directory.path().join(format!("{INDEX}.4242.tmp"));
        std::fs::write(&unfinished, b"{").unwrap();
        store.persist("model-a", &index).unwrap();

        let loaded = store.load("model-a");
        assert!(kept.exists());
        assert!(!orphan.exists(), "it can never be restored");
        assert!(!unfinished.exists());
        assert_eq!((loaded.removed, loaded.removed_bytes), (1, 5));
    }

    #[test]
    fn another_models_snapshots_are_discarded_but_foreign_files_are_kept() {
        let directory = tempfile::tempdir().unwrap();
        let store = SnapshotStore::open(directory.path()).unwrap();
        let (index, ids) = index_with(1);
        let ours = touch(directory.path(), ids[0]);
        let theirs = directory.path().join("someone-else.bin");
        std::fs::write(&theirs, b"x").unwrap();
        store.persist("model-a", &index).unwrap();

        let loaded = store.load("model-b");
        assert_eq!(
            loaded.started_over,
            Some(StartOver::OtherModel("model-a".into()))
        );
        assert!(loaded.index.is_empty());
        assert!(!ours.exists());
        assert!(theirs.exists());
    }

    #[test]
    fn an_unreadable_index_is_set_aside_not_trusted() {
        let directory = tempfile::tempdir().unwrap();
        let store = SnapshotStore::open(directory.path()).unwrap();
        std::fs::write(directory.path().join(INDEX), b"{\"version\":1,\"nam").unwrap();
        let file = touch(directory.path(), SnapshotId(3));

        let loaded = store.load("model-a");
        assert_eq!(loaded.started_over, Some(StartOver::Corrupt));
        assert!(directory.path().join(format!("{INDEX}.corrupt")).exists());
        assert!(!file.exists(), "unlisted, it can never be restored");
    }

    #[test]
    fn one_gate_owns_a_directory_at_a_time() {
        let directory = tempfile::tempdir().unwrap();
        let first = SnapshotStore::open(directory.path()).unwrap();
        assert!(matches!(
            SnapshotStore::open(directory.path()),
            Err(OpenError::Locked)
        ));
        drop(first);
        assert!(
            SnapshotStore::open(directory.path()).is_ok(),
            "released with the gate"
        );
    }

    #[test]
    fn concurrent_writes_and_reads_never_see_a_partial_index() {
        let directory = tempfile::tempdir().unwrap();
        let store = SnapshotStore::open(directory.path()).unwrap();
        let (small, _) = index_with(1);
        let (large, _) = index_with(64);
        std::thread::scope(|scope| {
            for writer in 0..4 {
                let (store, small, large) = (&store, &small, &large);
                scope.spawn(move || {
                    for round in 0..200 {
                        let index = if (writer + round) % 2 == 0 {
                            small
                        } else {
                            large
                        };
                        store.persist("model-a", index).unwrap();
                    }
                });
            }
            let store = &store;
            scope.spawn(move || {
                for _ in 0..400 {
                    let bytes = {
                        let _io = store.io();
                        std::fs::read(store.directory().join(INDEX)).ok()
                    };
                    if let Some(bytes) = bytes {
                        assert!(serde_json::from_slice::<IndexFile>(&bytes).is_ok());
                    }
                }
            });
        });
        assert_eq!(store.load("model-a").started_over, None);
    }

    #[tokio::test]
    async fn the_persister_writes_the_latest_version_and_flushes() {
        let directory = tempfile::tempdir().unwrap();
        let store = SnapshotStore::open(directory.path()).unwrap();
        let mut persister = Persister::spawn(store.clone(), "test".into());
        let mut ids = Vec::new();
        for count in 1..=50 {
            let (index, written) = index_with(count);
            ids = written;
            persister.submit("model-a".into(), index);
        }
        assert!(persister.flush(Duration::from_secs(5)).await);
        for id in &ids {
            touch(directory.path(), *id);
        }
        assert_eq!(
            store.load("model-a").index.len(),
            50,
            "the last version won"
        );
    }

    #[test]
    fn filenames_pass_llama_cpp_validation_rules() {
        let name = SnapshotStore::filename(SnapshotId(42));
        assert!(name.len() <= 255);
        assert!(!name.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|']));
        assert!(!name.contains(".."));
        assert_eq!(snapshot_id(&name), Some(SnapshotId(42)));
        assert_eq!(snapshot_id("llama-gate-index.json.1.tmp"), None);
    }
}
