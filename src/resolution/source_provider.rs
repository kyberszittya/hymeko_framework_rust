use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub trait SourceProvider {
    fn canonicalize(&self, p: &Path) -> Result<PathBuf, String>;
    fn read(&mut self, p: &Path) -> Result<std::sync::Arc<str>, String>;
    fn cwd(&self) -> Option<PathBuf> { None }
    fn version(&self, path: &Path) -> Result<u64, String>;
}

pub struct StdFsProvider {
    cache: std::collections::HashMap<std::path::PathBuf, std::sync::Arc<str>>,
}

impl StdFsProvider {
    pub fn new() -> Self { Self { cache: Default::default() } }

    pub fn cwd(&self) -> Option<PathBuf> {
        std::env::current_dir().ok()
    }
}

impl SourceProvider for StdFsProvider {
    fn canonicalize(&self, p: &Path) -> Result<PathBuf, String> {
        std::fs::canonicalize(p).map_err(|e| e.to_string())
    }

    fn read(&mut self, p: &Path) -> Result<std::sync::Arc<str>, String> {
        if let Some(s) = self.cache.get(p) { return Ok(s.clone()); }
        let s = std::fs::read_to_string(p).map_err(|e| e.to_string())?;
        let a: std::sync::Arc<str> = s.into();
        self.cache.insert(p.to_path_buf(), a.clone());
        Ok(a)
    }

    fn version(&self, path: &Path) -> Result<u64, String> {
        todo!()
    }
}

#[derive(Clone, Default)]
pub struct MemProvider {
    files: HashMap<PathBuf, Arc<str>>,
    reads: Arc<Mutex<HashMap<PathBuf, usize>>>,
}

impl MemProvider {
    pub fn with_file(mut self, path: impl Into<PathBuf>, content: impl Into<Arc<str>>) -> Self {
        self.files.insert(path.into(), content.into());
        self
    }

    pub fn read_count(&self, p: impl AsRef<Path>) -> usize {
        let m = self.reads.lock().unwrap();
        *m.get(p.as_ref()).unwrap_or(&0)
    }
}

impl SourceProvider for MemProvider {
    fn canonicalize(&self, p: &Path) -> Result<PathBuf, String> {
        // Tesztben elég az identity. (Ha akarsz, itt normalizálhatsz.)
        Ok(p.to_path_buf())
    }

    fn read(&mut self, p: &Path) -> Result<Arc<str>, String> {
        {
            let mut m = self.reads.lock().unwrap();
            *m.entry(p.to_path_buf()).or_insert(0) += 1;
        }
        self.files
            .get(p)
            .cloned()
            .ok_or_else(|| format!("file not found: {}", p.display()))
    }

    fn version(&self, path: &Path) -> Result<u64, String> {
        todo!()
    }
}