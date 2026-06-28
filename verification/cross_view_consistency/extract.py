"""Extraction functions X_f: read structural invariants back OUT of an *emitted* target view.

This is the formal device behind the cross-view-consistency claim of the HyMeKo T-SMC article. For each
registered format f the article emits epsilon_f(H); here we define X_f that parses that concrete syntax back into
a format-agnostic `KinematicInvariant`. The article's "cross-view consistency / no drift" claim is then the
*commuting square*

    X_f(epsilon_f(H)) == X_g(epsilon_g(H))   for every pair of views f, g     (mutual; non-circular)
    X_f(epsilon_f(H)) == Q(H)                 anchored to the IR query where available,

each X_f parsing a *different* concrete syntax (URDF/SDF XML, MJCF nested XML, DOT, Mermaid) independently, so
agreement is not an artifact of a shared parser.

Design (CLAUDE.md paradigm hierarchy): a `ViewExtractor` ABC with one Strategy impl per format, dispatched by the
CLI `--format` key. No global state; every extractor is a pure function of the emitted text.

Two emitter conventions are normalised here and named explicitly (they are conventions, not drift):
  (W) the synthetic URDF `world` link is a ground anchor, not a robot link -> the comparable link set is the
      mass-bearing links;
  (F) the fixed root weld (`j_fix`, world->base_link) is an explicit joint in URDF/SDF/DOT/Mermaid but *implicit*
      in MJCF (a root body with no joint is welded to the world) -> the comparable joint set is the *actuated*
      joints; fixed welds are reported separately.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

Axis = tuple[int, int, int]

# DOT / Mermaid encode the joint axis as a single letter; XML formats give the unit vector.
_LETTER_TO_AXIS: dict[str, Axis] = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}


def _round_mass(value: str | float) -> float:
    """Canonicalise a mass literal across formats ('8', '8.0', '8.00 kg') to a comparable float.

    Precondition: `value` parses as a float after stripping a trailing unit. Postcondition: returns the mass
    rounded to 3 decimals (sub-gram resolution), so textual formatting differences do not break equality.
    """
    if isinstance(value, str):
        value = value.strip().split()[0]  # drop a trailing ' kg'
    return round(float(value), 3)


def is_acyclic(edges: "frozenset[tuple[str, str]] | set[tuple[str, str]]") -> bool:
    """True iff the directed graph over (src, dst) edges has no cycle (DFS three-colouring).

    A *global* structural invariant, used for the kinematic forest (parent->child) and the requirement-derivation
    graph (derived->source). Precondition: edges is a finite set of 2-tuples. Fixtures are small, so the recursive
    DFS is adequate.
    """
    from collections import defaultdict
    adj: dict[str, list[str]] = defaultdict(list)
    nodes: set[str] = set()
    for src, dst in edges:
        adj[src].append(dst)
        nodes.update((src, dst))
    color: dict[str, int] = {}

    def visit(u: str) -> bool:
        color[u] = 1  # grey
        for v in adj[u]:
            c = color.get(v, 0)
            if c == 1 or (c == 0 and not visit(v)):
                return False
            _ = c
        color[u] = 2  # black
        return True

    return all(color.get(n, 0) != 0 or visit(n) for n in nodes)


def is_forest(chain: "frozenset[tuple[str, str, str]]") -> bool:
    """True iff the (joint, parent, child) chain is an acyclic single-parent structure (a rooted forest/tree) ---
    the global well-formedness invariant of a kinematic model."""
    edges = {(p, c) for _j, p, c in chain}
    parents: dict[str, set[str]] = {}
    for _j, p, c in chain:
        parents.setdefault(c, set()).add(p)
    single_parent = all(len(ps) == 1 for ps in parents.values())
    return single_parent and is_acyclic(edges)


def _axis_from_vector(text: str) -> Axis:
    """Parse a '0 0 1'-style axis triple into a sign-normalised integer unit vector."""
    parts = [float(x) for x in text.replace(",", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"axis vector must have 3 components, got {text!r}")
    return tuple(int(round(p)) for p in parts)  # type: ignore[return-value]


@dataclass(frozen=True)
class KinematicInvariant:
    """Format-agnostic structural invariant recovered from one emitted view.

    Invariants (the comparable content of the commuting square):
      links            - mass-bearing link names (convention W: excludes the synthetic `world` anchor);
      mass             - link -> mass (kg), the per-link mass map;
      actuated_joints  - non-fixed joint names (convention F: the format-independent joint set);
      chain            - (joint, parent_link, child_link) for actuated joints = parent/child polarity;
      axes             - joint -> axis unit vector for actuated joints.
    `fixed_joints` is recorded but kept out of the comparison core (it is convention-dependent).
    """

    fmt: str
    links: frozenset[str]
    mass: frozenset[tuple[str, float]]
    actuated_joints: frozenset[str]
    chain: frozenset[tuple[str, str, str]]
    axes: frozenset[tuple[str, Axis]]
    fixed_joints: frozenset[str] = field(default_factory=frozenset)

    def core(self) -> tuple:
        """The format-independent comparable tuple. Two views are cross-consistent iff their cores are equal."""
        return (self.links, self.mass, self.actuated_joints, self.chain, self.axes)


class ViewExtractor(ABC):
    """Strategy: X_f for one target format. `extract` is a pure function of the emitted text."""

    fmt: str

    @abstractmethod
    def extract(self, text: str) -> KinematicInvariant:
        """Parse emitted `text` into a `KinematicInvariant`.

        Precondition: `text` is the emitter's output for format `self.fmt`. Raises `ValueError` on text that is
        not parseable as this format (never silently returns an empty/partial invariant).
        """
        raise NotImplementedError


class _XmlExtractor(ViewExtractor):
    """Shared parse/guard for the XML-based formats (URDF, SDF, MJCF)."""

    def _root(self, text: str) -> ET.Element:
        if not text.strip():
            raise ValueError(f"{self.fmt}: empty emitter output")
        try:
            return ET.fromstring(text)
        except ET.ParseError as err:
            raise ValueError(f"{self.fmt}: malformed XML: {err}") from err


class UrdfExtractor(_XmlExtractor):
    fmt = "urdf"

    def extract(self, text: str) -> KinematicInvariant:
        root = self._root(text)
        links, mass = set(), set()
        for link in root.findall("link"):
            name = link.get("name", "")
            m = link.find("./inertial/mass")
            if m is not None and m.get("value") is not None:  # convention W: world link has no inertial
                links.add(name)
                mass.add((name, _round_mass(m.get("value", "0"))))
        actuated, chain, axes, fixed = set(), set(), set(), set()
        for joint in root.findall("joint"):
            jn = joint.get("name", "")
            # NB: an ElementTree Element with no children is falsy, so `find(...) or default`
            # silently drops a present-but-childless <parent>/<child>; test `is not None`.
            p, c = joint.find("parent"), joint.find("child")
            parent = p.get("link", "") if p is not None else ""
            child = c.get("link", "") if c is not None else ""
            ax = joint.find("axis")
            if joint.get("type") == "fixed" or ax is None:  # convention F
                fixed.add(jn)
                continue
            actuated.add(jn)
            chain.add((jn, parent, child))
            axes.add((jn, _axis_from_vector(ax.get("xyz", "0 0 0"))))
        return KinematicInvariant(self.fmt, frozenset(links), frozenset(mass),
                                  frozenset(actuated), frozenset(chain), frozenset(axes), frozenset(fixed))


class SdfExtractor(_XmlExtractor):
    fmt = "sdf"

    def extract(self, text: str) -> KinematicInvariant:
        root = self._root(text)
        model = root.find("model") if root.tag == "sdf" else root
        if model is None:
            raise ValueError("sdf: no <model> element")
        links, mass = set(), set()
        for link in model.findall("link"):
            name = link.get("name", "")
            links.add(name)
            m = link.find("./inertial/mass")
            if m is not None and m.text:
                mass.add((name, _round_mass(m.text)))
        actuated, chain, axes, fixed = set(), set(), set(), set()
        for joint in model.findall("joint"):
            jn = joint.get("name", "")
            parent = (joint.findtext("parent") or "").strip()
            child = (joint.findtext("child") or "").strip()
            xyz = joint.findtext("./axis/xyz")
            if joint.get("type") == "fixed" or xyz is None:
                fixed.add(jn)
                continue
            actuated.add(jn)
            chain.add((jn, parent, child))
            axes.add((jn, _axis_from_vector(xyz)))
        return KinematicInvariant(self.fmt, frozenset(links), frozenset(mass),
                                  frozenset(actuated), frozenset(chain), frozenset(axes), frozenset(fixed))


class MjcfExtractor(_XmlExtractor):
    """MJCF nests bodies; the joint inside a body actuates that body relative to its enclosing (parent) body.

    Convention F: a body that carries no joint is welded to its enclosing frame. For the root body under
    <worldbody> the enclosing frame is the implicit world, so a jointless root yields a synthetic fixed joint
    'j_fix:<root>' (matching the explicit `j_fix` URDF/SDF/DOT emit); a root WITH a joint (a slide/continuous
    base, e.g. `rail`/`slide_x`/`base`) is connected to 'world', matching the other formats' parent='world'.
    """

    fmt = "mjcf"
    WORLD = "world"

    def extract(self, text: str) -> KinematicInvariant:
        root = self._root(text)
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("mjcf: no <worldbody>")
        links, mass, actuated, chain, axes, fixed = set(), set(), set(), set(), set(), set()

        def walk(body: ET.Element, parent_link: str) -> None:
            name = body.get("name", "")
            links.add(name)
            inertial = body.find("inertial")
            if inertial is not None and inertial.get("mass") is not None:
                mass.add((name, _round_mass(inertial.get("mass", "0"))))
            joints = body.findall("joint")
            if not joints:
                fixed.add(f"j_fix:{name}")           # implicit weld to enclosing frame (convention F)
            for joint in joints:
                jn = joint.get("name", "")
                actuated.add(jn)
                chain.add((jn, parent_link, name))
                axes.add((jn, _axis_from_vector(joint.get("axis", "0 0 0"))))
            for child in body.findall("body"):
                walk(child, name)

        for top in worldbody.findall("body"):
            walk(top, self.WORLD)                    # the root's implicit parent is the world frame
        return KinematicInvariant(self.fmt, frozenset(links), frozenset(mass),
                                  frozenset(actuated), frozenset(chain), frozenset(axes), frozenset(fixed))


class _GraphExtractor(ViewExtractor):
    """Shared logic for the line-oriented graph formats (DOT, Mermaid): node labels carry mass, edge labels carry
    the joint name and axis letter. Subclasses supply the node/edge regexes and the fixed-edge predicate."""

    node_re: re.Pattern[str]
    edge_re: re.Pattern[str]

    def _is_fixed(self, edge_line: str, joint_name: str, axis_letter: str | None) -> bool:
        raise NotImplementedError

    def extract(self, text: str) -> KinematicInvariant:
        if not text.strip():
            raise ValueError(f"{self.fmt}: empty emitter output")
        links, mass = set(), set()
        for m in self.node_re.finditer(text):
            name, kg = m.group("name"), m.group("mass")
            links.add(name)
            mass.add((name, _round_mass(kg)))
        actuated, chain, axes, fixed = set(), set(), set(), set()
        for m in self.edge_re.finditer(text):
            parent, child, jn = m.group("parent"), m.group("child"), m.group("joint")
            letter = (m.groupdict().get("axis") or "").upper() or None
            if self._is_fixed(m.group(0), jn, letter):
                fixed.add(jn)
                continue
            actuated.add(jn)
            chain.add((jn, parent, child))
            if letter in _LETTER_TO_AXIS:
                axes.add((jn, _LETTER_TO_AXIS[letter]))
        return KinematicInvariant(self.fmt, frozenset(links), frozenset(mass),
                                  frozenset(actuated), frozenset(chain), frozenset(axes), frozenset(fixed))


class DotExtractor(_GraphExtractor):
    fmt = "dot"
    # "link_0" [label="link_0\n3.0 kg"];
    node_re = re.compile(r'"(?P<name>[^"]+)"\s*\[label="(?P=name)\\n(?P<mass>[\d.]+)\s*kg"')
    #   "base_link" -> "link_0" [label="j0\n(Z)", style=bold];
    edge_re = re.compile(
        r'"(?P<parent>[^"]+)"\s*->\s*"(?P<child>[^"]+)"\s*\[label="(?P<joint>[^"\\]+)'
        r'(?:\\n\((?P<axis>[XYZ])\))?"(?P<rest>[^\]]*)\]')

    def _is_fixed(self, edge_line: str, joint_name: str, axis_letter: str | None) -> bool:
        return "style=dashed" in edge_line or axis_letter is None


class MermaidExtractor(_GraphExtractor):
    fmt = "mermaid"
    # link_0["<b>link_0</b><br/>3.00 kg"]:::link
    node_re = re.compile(r'(?P<name>\w+)\["<b>(?P=name)</b><br/>(?P<mass>[\d.]+)\s*kg"\]')
    # base_link -->|"j0 (rev, Z)"| link_0     and    world -.->|"j_fix (fixed)"| base_link
    edge_re = re.compile(
        r'(?P<parent>\w+)\s*-(?P<style>\.?)->\|"(?P<joint>\w+)\s*\((?P<kind>[^),]+)'
        r'(?:,\s*(?P<axis>[XYZ]))?\)"\|\s*(?P<child>\w+)')

    def _is_fixed(self, edge_line: str, joint_name: str, axis_letter: str | None) -> bool:
        return "fixed" in edge_line or axis_letter is None


# Registry: CLI --format key -> extractor instance. The robotics cross-view set.
EXTRACTORS: dict[str, ViewExtractor] = {
    e.fmt: e for e in (UrdfExtractor(), SdfExtractor(), MjcfExtractor(), DotExtractor(), MermaidExtractor())
}
