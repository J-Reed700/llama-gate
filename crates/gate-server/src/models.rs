//! The models the gate serves, and which one a request asks for.
//!
//! Each backend serves one model, named in its configuration (or after the
//! backend). Backends naming the same model are interchangeable replicas.
//! A request's `model` field picks a model by name or alias; requests that
//! name none, or one the gate does not know, go to the default model, so
//! clients that send any name to a single-model server keep working.
use gate_core::ModelId;
use serde::Serialize;
use std::sync::RwLock;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelEntry {
    pub id: ModelId,
    pub name: String,
    /// Other names: configured aliases and what the servers report.
    pub aliases: Vec<String>,
}

#[derive(Debug, Default)]
pub struct Models {
    inner: RwLock<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    entries: Vec<ModelEntry>,
    default: Option<ModelId>,
}

impl Models {
    /// Registers `name` (if new) with `aliases`, returning its id. The first
    /// model registered is the default until [`Models::set_default`].
    pub fn register(&self, name: &str, aliases: &[String]) -> ModelId {
        let Ok(mut inner) = self.inner.write() else {
            return ModelId(0);
        };
        let id = match inner.entries.iter().position(|entry| entry.name == name) {
            Some(index) => ModelId(index as u16),
            None => {
                let id = ModelId(inner.entries.len() as u16);
                inner.entries.push(ModelEntry {
                    id,
                    name: name.to_owned(),
                    aliases: Vec::new(),
                });
                id
            }
        };
        let entry = &mut inner.entries[usize::from(id.0)];
        for alias in aliases {
            if !alias.is_empty() && alias != name && !entry.aliases.contains(alias) {
                entry.aliases.push(alias.clone());
            }
        }
        inner.default.get_or_insert(id);
        id
    }

    pub fn set_default(&self, name: &str) {
        let id = self.find(name);
        if let (Some(id), Ok(mut inner)) = (id, self.inner.write()) {
            inner.default = Some(id);
        }
    }

    /// The model a name or alias refers to.
    pub fn find(&self, name: &str) -> Option<ModelId> {
        let inner = self.inner.read().ok()?;
        inner
            .entries
            .iter()
            .find(|entry| entry.name == name)
            .or_else(|| {
                inner
                    .entries
                    .iter()
                    .find(|entry| entry.aliases.iter().any(|alias| alias == name))
            })
            .map(|entry| entry.id)
    }

    /// The model a request asking for `requested` goes to.
    pub fn resolve(&self, requested: Option<&str>) -> Option<ModelId> {
        requested
            .and_then(|name| self.find(name))
            .or_else(|| self.inner.read().ok()?.default)
    }

    pub fn name(&self, id: ModelId) -> String {
        self.inner
            .read()
            .ok()
            .and_then(|inner| {
                inner
                    .entries
                    .get(usize::from(id.0))
                    .map(|entry| entry.name.clone())
            })
            .unwrap_or_else(|| id.to_string())
    }

    pub fn list(&self) -> Vec<ModelEntry> {
        self.inner
            .read()
            .map(|inner| inner.entries.clone())
            .unwrap_or_default()
    }
}

/// Names a server's reported model goes by: the reported string itself and
/// its file name without extension, so `/models/Qwen3-8B-Q4_K_M.gguf` also
/// answers to `Qwen3-8B-Q4_K_M`.
pub fn reported_names(reported: &str) -> Vec<String> {
    let mut names = vec![reported.to_owned()];
    let file = reported.rsplit(['/', '\\']).next().unwrap_or(reported);
    for name in [file, file.strip_suffix(".gguf").unwrap_or(file)] {
        if !names.iter().any(|known| known == name) {
            names.push(name.to_owned());
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_find_their_model_by_name_or_alias_and_fall_back_to_the_default() {
        let models = Models::default();
        let qwen = models.register("qwen", &["qwen3.8".into()]);
        let gemma = models.register("gemma", &reported_names("/models/gemma-3-4b.gguf"));
        assert_eq!(models.register("qwen", &[]), qwen, "replicas share the id");
        assert_eq!(models.resolve(Some("qwen3.8")), Some(qwen));
        assert_eq!(models.resolve(Some("gemma-3-4b")), Some(gemma));
        assert_eq!(
            models.resolve(Some("gpt-4o")),
            Some(qwen),
            "unknown: the default"
        );
        assert_eq!(models.resolve(None), Some(qwen));
        models.set_default("gemma");
        assert_eq!(models.resolve(None), Some(gemma));
        assert_eq!(models.name(gemma), "gemma");
    }
}
