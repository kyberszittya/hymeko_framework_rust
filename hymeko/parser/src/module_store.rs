use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use std::sync::Arc;
use crate::{ast::*, interner::Interner};
use crate::common::ids::SymId;
use crate::ir::hash_pass::compute_merkle_hashes;
use crate::ir::ir::Ir;
use crate::ir::lower::lower_program_to_ir;
use crate::resolve::{build_index_sym_with_prefix, validate_all_refs_sym_with_prefix, Index};
use crate::source_provider::SourceProvider;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModuleKey(pub PathBuf);

#[derive(Debug)]
pub struct ModuleEntry {
    pub key: ModuleKey,
    pub ast: AstSym<'static>,
    pub deps: Vec<ModuleKey>,
}

pub struct CompiledProgram {
    pub root: ModuleKey,
    pub idx: Index,
    pub ir: Ir,
    // extra debug/teszt kényelmesség:
    pub imports: Vec<(SymId, ModuleKey)>, // (namespace, module key)
}

#[derive(Debug)]
pub enum ModuleLoadError {
    Io(IoDiag),
    Parse(String),
    Cycle { stack: Vec<PathBuf> },
}

#[derive(Debug)]
pub struct IoDiag {
    pub op: &'static str,
    pub path: PathBuf,
    pub cwd: Option<PathBuf>,
    pub err: std::io::Error,
}



/// A parse-t is kiszervezzük, mert nincs még "main"/egységes entrypoint.
/// Tesztben adsz neki egy implementációt, ami a LALRPOP parsereidet hívja.
pub trait HymekoParser {
    fn parse<'a>(&self, src: &'a str) -> Result<AstStr<'a>, String>;
}

pub struct CompiledProgramCache {
    // később ide jöhet fingerprint/hash
    pub idx: Index,
    pub ir: Ir,
    pub imports: Vec<(SymId, ModuleKey)>,
}

pub struct ModuleStore<P: SourceProvider, R: HymekoParser> {
    pub it: Interner,
    fs: P,
    parser: R,
    entries: HashMap<ModuleKey, ModuleEntry>,
    in_progress: Vec<ModuleKey>,
    program_cache: HashMap<ModuleKey, CompiledProgramCache>,
    last_compiled: Option<Arc<CompiledProgram>>,
}

impl<'a, P: SourceProvider, R: HymekoParser> ModuleStore<P, R> {
    pub fn new(fs: P, parser: R) -> Self {
        Self {
            it: Interner::new(),
            fs,
            parser,
            entries: HashMap::new(),
            in_progress: Vec::new(),
            program_cache: HashMap::new(),
            last_compiled: None,
        }
    }

    pub fn get(&self, k: &ModuleKey) -> Option<&ModuleEntry> {
        self.entries.get(k)
    }

    pub fn load_recursive(&mut self, root: &Path) -> Result<ModuleKey, ModuleLoadError> {
        let cwd = std::env::current_dir().ok();

        let root_can = self.fs.canonicalize(root).map_err(|e| {
            ModuleLoadError::Io(IoDiag {
                op: "canonicalize",
                path: root.to_path_buf(),
                cwd: cwd.clone(),
                err: std::io::Error::new(std::io::ErrorKind::Other, e),
            })
        })?;

        let key = ModuleKey(root_can);
        self.ensure_loaded(&key)?;
        Ok(key)
    }

    pub fn canonicalize_path(&self, p: &Path) -> Result<PathBuf, String> {
        self.fs.canonicalize(p)
    }

    fn ensure_loaded(&mut self, key: &ModuleKey) -> Result<(), ModuleLoadError> {
        if self.entries.contains_key(key) {
            return Ok(());
        }
        if self.in_progress.contains(key) {
            return Err(ModuleLoadError::Cycle {
                stack: self.in_progress.iter().map(|k| k.0.clone()).collect(),
            });
        }

        self.in_progress.push(key.clone());

        // 1) read (provider cache-elhet)
        let src = self
            .fs
            .read(&key.0)
            .map_err(|e| io_diag("read", &key.0, e))?;

        // 2) parse
        let ast_str = self
            .parser
            .parse(src.as_ref())
            .map_err(ModuleLoadError::Parse)?;
        // 3) intern (közös internerbe!)
        let ast_sym: AstSym<'static> =
            crate::intern_pass::intern_ast_into_owned(&ast_str, &mut self.it);
        // 4) deps from imports
        let base_dir = key.0.parent().unwrap_or(Path::new("."));
        let mut deps = Vec::new();

        for imp in &ast_sym.imports {
            let resolved = resolve_import_path(base_dir, imp.path.as_ref());
            let canon = self
                .fs
                .canonicalize(&resolved)
                .map_err(|e| io_diag("canonicalize", &resolved, e))?;
            deps.push(ModuleKey(canon));
        }

        // 5) recursively load deps
        for d in &deps {
            self.ensure_loaded(d)?;
        }

        // 6) commit
        let entry = ModuleEntry {
            key: key.clone(),
            ast: ast_sym,
            deps,
        };
        self.entries.insert(key.clone(), entry);

        self.in_progress.pop();
        Ok(())
    }

    pub fn compile(&mut self, root_path: &Path) -> Result<Arc<CompiledProgram>, ModuleLoadError> {
        // 1) load modules
        let root = self.load_recursive(root_path)?;

        // 2) cache (ultra-minimál): ha ugyanaz a root, add vissza
        if let Some(c) = &self.last_compiled {
            if c.root == root {
                return Ok(c.clone());
            }
        }

        // 3) kivehetjük a root AST-t (clone, hogy ne legyen borrow fight)
        let root_ast = self
            .get(&root)
            .ok_or_else(|| ModuleLoadError::Parse("root missing after load".into()))?
            .ast
            .clone();

        let root_entry = self.get(&root).unwrap();
        let deps = root_entry.deps.clone();

        // 4) imported list (alias opcionális: ns = alias || dep_ast.name)
        let mut imported: Vec<(SymId, AstSym<'static>)> = Vec::new();
        let mut imports_map: Vec<(SymId, ModuleKey)> = Vec::new();

        for (imp, dep_key) in root_ast.imports.iter().zip(deps.iter()) {
            let dep_ast = self
                .get(dep_key)
                .ok_or_else(|| ModuleLoadError::Parse(format!("dep missing: {}", dep_key.0.display())))?
                .ast
                .clone();

            let ns = imp.alias.unwrap_or(dep_ast.name);
            imports_map.push((ns, dep_key.clone()));
            imported.push((ns, dep_ast));
        }

        // 5) global index: root + deps namespace alatt
        let mut idx = Index { by_path: std::collections::HashMap::new() };
        let mut next: u32 = 0;

        build_index_sym_with_prefix(&root_ast, &[], &self.it, &mut idx, &mut next)
            .map_err(|e| ModuleLoadError::Parse(format!("index root failed: {e:?}")))?;

        for (ns, dep_ast) in imported.iter() {
            build_index_sym_with_prefix(dep_ast, &[*ns], &self.it, &mut idx, &mut next)
                .map_err(|e| ModuleLoadError::Parse(format!("index dep failed: {e:?}")))?;
        }

        // 6) resolve/validate (ajánlott)
        validate_all_refs_sym_with_prefix(&root_ast, &[], &idx, &mut self.it)
            .map_err(|e| ModuleLoadError::Parse(format!("resolve/validate failed: {e:?}")))?;

        // 7) lower program IR (2A) + merkle
        let mut ir = lower_program_to_ir(&root_ast, &imported, &idx, &mut self.it)
            .map_err(|e| ModuleLoadError::Parse(format!("lower failed: {e:?}")))?;

        compute_merkle_hashes(&mut ir, &self.it);

        let compiled = Arc::new(CompiledProgram {
            root: root.clone(),
            idx,
            ir,
            imports: imports_map,
        });

        self.last_compiled = Some(compiled.clone());
        Ok(compiled)
    }
}

fn resolve_import_path(base_dir: &Path, import_path: &str) -> PathBuf {
    let p = Path::new(import_path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base_dir.join(p)
    }
}

fn io_diag(op: &'static str, path: &Path, e: String) -> ModuleLoadError {
    ModuleLoadError::Io(IoDiag {
        op,
        path: path.to_path_buf(),
        cwd: std::env::current_dir().ok(),
        err: std::io::Error::new(std::io::ErrorKind::Other, e),
    })
}
