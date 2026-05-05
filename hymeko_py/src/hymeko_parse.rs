//! Bridge `parser::parse_description` into Python: read a .hymeko
//! source, walk the AST, and emit a Python dict-tree the driver can
//! traverse.  Lossy on a few fields (lifetimes, ConstExpr) — the
//! driver only needs the hierarchy + values that drive instantiation.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

type PyObject = Py<PyAny>;

use parser::ast::{Anno, ConstExpr, Description, EdgeDecl, HyperItem,
                    NodeDecl, SignedRef, Value};
use parser::parse_description as rs_parse;

/// Parse a .hymeko source string and return a nested Python
/// dictionary tree mirroring the AST.  Schema (informally):
///
/// ```python
/// {
///   "name": str,                   # description name
///   "items": [item, ...],          # top-level HyperItems
/// }
/// item ∈
///   {"kind": "node", "name": str, "tags": [str], "value": value|None,
///    "bases": [signed_ref], "body": [item] | None}
///   {"kind": "edge", "name": str, "tags": [str], "value": value|None,
///    "bases": [signed_ref], "body": [item]}
///   {"kind": "arc",  "tags": [str], "value": value|None,
///    "refs": [signed_ref]}
/// signed_ref ∈
///   {"sign": "+"|"-"|"~", "path": [str], "tags": [str]}
/// value ∈
///   str | float | bool | list | {"ref": [str]} | {"expr": str}
/// ```
#[pyfunction]
pub fn parse_hymeko_rs<'py>(py: Python<'py>, source: &str) -> PyResult<Py<PyDict>> {
    let desc = rs_parse(source).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("parse error: {e:?}"))
    })?;
    let out = PyDict::new(py);
    out.set_item("name", desc.name)?;
    out.set_item("items", emit_items(py, &desc.items)?)?;
    out.set_item("imports", emit_imports(py, &desc)?)?;
    out.set_item("usings", emit_usings(py, &desc)?)?;
    out.set_item("consts", emit_consts(py, &desc)?)?;
    Ok(out.into())
}

fn emit_imports<'py>(py: Python<'py>, d: &Description<&str>) -> PyResult<Py<PyList>> {
    let lst = PyList::empty(py);
    for i in &d.imports {
        let dct = PyDict::new(py);
        dct.set_item("path", &*i.path)?;
        dct.set_item("alias", i.alias.unwrap_or(""))?;
        lst.append(dct)?;
    }
    Ok(lst.into())
}

fn emit_usings<'py>(py: Python<'py>, d: &Description<&str>) -> PyResult<Py<PyList>> {
    let lst = PyList::empty(py);
    for u in &d.usings {
        let dct = PyDict::new(py);
        dct.set_item("path", u.path.path.join("."))?;
        dct.set_item("alias", u.alias)?;
        lst.append(dct)?;
    }
    Ok(lst.into())
}

fn emit_consts<'py>(py: Python<'py>, d: &Description<&str>) -> PyResult<Py<PyList>> {
    let lst = PyList::empty(py);
    for c in &d.consts {
        let dct = PyDict::new(py);
        dct.set_item("name", c.name)?;
        dct.set_item("value", emit_const_expr(py, &c.value)?)?;
        lst.append(dct)?;
    }
    Ok(lst.into())
}

fn emit_items<'py, 'a>(
    py: Python<'py>,
    items: &[HyperItem<'a, &'a str>],
) -> PyResult<Py<PyList>> {
    let lst = PyList::empty(py);
    for it in items {
        match it {
            HyperItem::Node(n) => lst.append(emit_node(py, n)?)?,
            HyperItem::Edge(e) => lst.append(emit_edge(py, e)?)?,
            HyperItem::Arc(a) => {
                let dct = PyDict::new(py);
                dct.set_item("kind", "arc")?;
                dct.set_item("tags", emit_tags(py, &a.anno)?)?;
                dct.set_item("value", emit_anno_value(py, &a.anno)?)?;
                let refs = PyList::empty(py);
                for r in &a.inner.refs {
                    refs.append(emit_signed_ref(py, r)?)?;
                }
                dct.set_item("refs", refs)?;
                lst.append(dct)?;
            }
        }
    }
    Ok(lst.into())
}

fn emit_node<'py, 'a>(
    py: Python<'py>,
    n: &NodeDecl<'a, &'a str>,
) -> PyResult<Py<PyDict>> {
    let dct = PyDict::new(py);
    dct.set_item("kind", "node")?;
    dct.set_item("name", n.inner.name)?;
    dct.set_item("tags", emit_tags(py, &n.anno)?)?;
    dct.set_item("value", emit_anno_value(py, &n.anno)?)?;
    let bases = PyList::empty(py);
    for b in &n.inner.bases {
        bases.append(emit_signed_ref(py, b)?)?;
    }
    dct.set_item("bases", bases)?;
    if let Some(body) = &n.inner.body {
        dct.set_item("body", emit_items(py, body)?)?;
    } else {
        dct.set_item("body", py.None())?;
    }
    Ok(dct.into())
}

fn emit_edge<'py, 'a>(
    py: Python<'py>,
    e: &EdgeDecl<'a, &'a str>,
) -> PyResult<Py<PyDict>> {
    let dct = PyDict::new(py);
    dct.set_item("kind", "edge")?;
    dct.set_item("name", e.inner.name)?;
    dct.set_item("tags", emit_tags(py, &e.anno)?)?;
    dct.set_item("value", emit_anno_value(py, &e.anno)?)?;
    let bases = PyList::empty(py);
    for b in &e.inner.bases {
        bases.append(emit_signed_ref(py, b)?)?;
    }
    dct.set_item("bases", bases)?;
    dct.set_item("body", emit_items(py, &e.inner.body)?)?;
    Ok(dct.into())
}

fn emit_signed_ref<'py, 'a>(
    py: Python<'py>,
    r: &SignedRef<'a, &'a str>,
) -> PyResult<Py<PyDict>> {
    let dct = PyDict::new(py);
    let (sign, atom) = match r {
        SignedRef::Plus(a) => ("+", a),
        SignedRef::Minus(a) => ("-", a),
        SignedRef::Neutral(a) => ("~", a),
    };
    dct.set_item("sign", sign)?;
    dct.set_item("path", atom.target.path.clone())?;
    dct.set_item("tags", emit_tags(py, &atom.anno)?)?;
    Ok(dct.into())
}

fn emit_tags<'py, 'a>(py: Python<'py>, a: &Anno<'a, &'a str>) -> PyResult<Py<PyList>> {
    let lst = PyList::empty(py);
    for t in &a.tags {
        lst.append(*t)?;
    }
    Ok(lst.into())
}

fn emit_anno_value<'py, 'a>(
    py: Python<'py>,
    a: &Anno<'a, &'a str>,
) -> PyResult<PyObject> {
    if let Some(v) = &a.value {
        emit_value(py, v)
    } else {
        Ok(py.None())
    }
}

fn emit_value<'py, 'a>(
    py: Python<'py>,
    v: &Value<'a, &'a str>,
) -> PyResult<PyObject> {
    use pyo3::IntoPyObjectExt;
    match v {
        Value::Str(s) => Ok((s.as_ref()).into_py_any(py)?),
        Value::Num(n) => Ok((*n).into_py_any(py)?),
        Value::List(xs) => {
            let lst = PyList::empty(py);
            for x in xs {
                lst.append(emit_value(py, x)?)?;
            }
            Ok(lst.into())
        }
        Value::Ref(r) => {
            let d = PyDict::new(py);
            d.set_item("ref", r.path.clone())?;
            Ok(d.into())
        }
        Value::Expr(e) => emit_const_expr(py, e),
    }
}

fn emit_const_expr<'py>(
    py: Python<'py>,
    e: &ConstExpr<&str>,
) -> PyResult<PyObject> {
    use pyo3::IntoPyObjectExt;
    // The driver only needs literal numbers and identifier refs in
    // practice; complex expressions surface as a textual placeholder.
    match e {
        ConstExpr::Lit(n) => Ok((*n).into_py_any(py)?),
        ConstExpr::Ref(name) => {
            let d = PyDict::new(py);
            d.set_item("ref", *name)?;
            Ok(d.into())
        }
        ConstExpr::Pi => Ok(std::f64::consts::PI.into_py_any(py)?),
        _ => {
            let d = PyDict::new(py);
            d.set_item("expr", "<complex>")?;
            Ok(d.into())
        }
    }
}
