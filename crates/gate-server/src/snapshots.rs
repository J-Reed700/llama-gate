//! Snapshot files: naming, deletion, and the persisted index.
//!
//! The llama.cpp server writes and reads the files (in its
//! `--slot-save-path`); the gate names them, keeps their index, and deletes
//! them. The index records the model identity it belongs to, so snapshots
//! from another model are never offered for restore.
use gate_core::{SnapshotId, SnapshotIndex};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const INDEX: &str = "llama-gate-index.json";
const PREFIX: &str = "llama-gate-";
const SUFFIX: &str = ".slot";

#[derive(Clone, Debug)]
pub struct SnapshotStore {
    directory: Option<PathBuf>,
}

#[derive(Serialize, Deserialize)]
struct IndexFile {
    version: u32,
    namespace: String,
    index: SnapshotIndex,
}

impl SnapshotStore {
    pub fn new(directory: Option<PathBuf>) -> Self {
        Self { directory }
    }

    /// Whether the gate can manage snapshot files at all.
    pub fn enabled(&self) -> bool {
        self.directory.as_deref().is_some_and(Path::is_dir)
    }

    /// The file name the server is asked to use for a snapshot. Only
    /// characters llama.cpp's filename validation accepts.
    pub fn filename(id: SnapshotId) -> String {
        format!("{PREFIX}{:016}{SUFFIX}", id.0)
    }

    /// Loads the index for `namespace`. A missing, unreadable, or foreign
    /// index starts empty and the gate's own stale files are removed; entries
    /// whose files are gone are dropped.
    pub fn load(&self, namespace: &str) -> SnapshotIndex {
        let Some(directory) = self.directory.as_deref().filter(|path| path.is_dir()) else {
            return SnapshotIndex::default();
        };
        let parsed = std::fs::read(directory.join(INDEX))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<IndexFile>(&bytes).ok())
            .filter(|file| file.version == 1 && file.namespace == namespace);
        let Some(file) = parsed else {
            self.remove_all(directory);
            return SnapshotIndex::default();
        };
        let mut index = file.index;
        let missing: Vec<SnapshotId> = index
            .iter()
            .filter(|meta| !directory.join(Self::filename(meta.id)).is_file())
            .map(|meta| meta.id)
            .collect();
        for id in missing {
            index.remove(id);
        }
        index
    }

    /// Writes the index atomically (write, sync, rename).
    pub fn persist(&self, namespace: &str, index: &SnapshotIndex) -> std::io::Result<()> {
        let Some(directory) = self.directory.as_deref() else {
            return Ok(());
        };
        let file = IndexFile {
            version: 1,
            namespace: namespace.to_owned(),
            index: index.clone(),
        };
        let bytes = serde_json::to_vec(&file).map_err(std::io::Error::other)?;
        let temporary = directory.join(format!("{INDEX}.tmp"));
        {
            use std::io::Write as _;
            let mut handle = std::fs::File::create(&temporary)?;
            handle.write_all(&bytes)?;
            handle.sync_all()?;
        }
        std::fs::rename(temporary, directory.join(INDEX))
    }

    pub fn delete(&self, id: SnapshotId) {
        if let Some(directory) = self.directory.as_deref() {
            let _ = std::fs::remove_file(directory.join(Self::filename(id)));
        }
    }

    /// Removes every file this gate could have written; leaves others alone.
    fn remove_all(&self, directory: &Path) {
        let Ok(entries) = std::fs::read_dir(directory) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(PREFIX) && name.ends_with(SUFFIX) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gate_core::Prefix;

    #[test]
    fn the_index_survives_a_restart_and_drops_missing_files() {
        let directory = tempfile::tempdir().unwrap();
        let store = SnapshotStore::new(Some(directory.path().into()));
        let mut index = SnapshotIndex::default();
        let kept = index.allocate();
        index.insert(kept, Prefix::from_tokens(&[1, 2, 3, 4], 2), 4, 100);
        let lost = index.allocate();
        index.insert(lost, Prefix::from_tokens(&[5, 6, 7, 8], 2), 4, 100);
        std::fs::write(directory.path().join(SnapshotStore::filename(kept)), b"x").unwrap();
        store.persist("model-a", &index).unwrap();

        let loaded = store.load("model-a");
        assert!(loaded.get(kept).is_some());
        assert!(loaded.get(lost).is_none(), "its file is gone");
    }

    #[test]
    fn another_models_snapshots_are_discarded_but_foreign_files_are_kept() {
        let directory = tempfile::tempdir().unwrap();
        let store = SnapshotStore::new(Some(directory.path().into()));
        let mut index = SnapshotIndex::default();
        let id = index.allocate();
        index.insert(id, Prefix::from_tokens(&[1, 2], 2), 2, 10);
        let ours = directory.path().join(SnapshotStore::filename(id));
        let theirs = directory.path().join("someone-else.bin");
        std::fs::write(&ours, b"x").unwrap();
        std::fs::write(&theirs, b"x").unwrap();
        store.persist("model-a", &index).unwrap();

        assert!(store.load("model-b").is_empty());
        assert!(!ours.exists());
        assert!(theirs.exists());
    }

    #[test]
    fn filenames_pass_llama_cpp_validation_rules() {
        let name = SnapshotStore::filename(SnapshotId(42));
        assert!(name.len() <= 255);
        assert!(!name.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|']));
        assert!(!name.contains(".."));
    }
}
