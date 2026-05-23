# HyMeKo Query-Driven Rewrite Engine — Tutorial

## 0. Why This Exists

The old approach: every output format (URDF, SDF, MJCF, DOT) was a hardcoded Rust function inside `hymeko_query`. Each format knew how to traverse the IR, extract joint/link data, and emit XML. Adding Gazebo meant writing 200 lines of Rust. Adding MuJoCo meant 300 more. The query engine became a dumping ground for domain-specific code.

The new approach: a transform is just two files. The query engine stays generic. The domain knowledge lives outside the codebase.

```
transforms/urdf/
├── queries.hymeko           ← "what to find" (HyMeKo syntax)
└── template.urdf.xml        ← "what to write" (XML + template tags)
```

The rewrite engine connects them:

```
queries.hymeko ──parse──→ predicates ──query──→ matches ──render──→ output
                                                   ↑
template.xml ──parse──→ blocks ─────────────────────┘
```

---

## 1. Architecture Overview

### 1.1 The Three Layers

```
┌───────────────────────────────────────────────┐
│  Layer 1: Query Engine (unchanged)            │
│  QueryEngine, Predicate, QueryMatch           │
│  "Find all edges inheriting from rev_joint"   │
├───────────────────────────────────────────────┤
│  Layer 2: Rewrite Engine (new)                │
│  MatchContext, Template, TransformSpec         │
│  "Extract field:mass from match, render XML"  │
├───────────────────────────────────────────────┤
│  Layer 3: Transform Definitions (external)    │
│  queries.hymeko + template.* files            │
│  "URDF needs links, joints, axes..."          │
└───────────────────────────────────────────────┘
```

Layer 1 knows about hypergraphs. Layer 2 knows about matching and templating. Layer 3 knows about URDF/SDF/DOT. The layers never leak upward.

### 1.2 File Map

```
hymeko_query/src/
├── engine.rs              Layer 1 — QueryEngine, QueryMatch
├── predicate.rs           Layer 1 — Predicate tree
├── interpret.rs           Layer 1 — .hymeko AST → Predicate
├── rewrite/
│   ├── mod.rs             Layer 2 — execute_transform()
│   ├── match_context.rs   Layer 2 — field extraction from IR
│   └── template.rs        Layer 2 — template parser + renderer
├── formats/               (old Layer 3, kept for backward compat)
│   ├── urdf.rs
│   └── sdf.rs
└── codegen.rs             (old dispatcher, kept for backward compat)

transforms/                Layer 3 — external definitions
├── urdf/
│   ├── queries.hymeko
│   └── template.urdf.xml
├── sdf/
│   ├── queries.hymeko
│   └── template.sdf.xml
└── dot/
    ├── queries.hymeko
    └── template.dot
```

---

## 2. Creating a Transform: Step-by-Step

### 2.1 Understand What You Want to Extract

Before writing anything, answer: "What entities from the hypergraph does this format need?"

For URDF, that's:
- **Links** (nodes inheriting from `link`)
- **Joints** by type (edges inheriting from `fixed_joint`, `rev_joint`, etc.)
- For each link: `mass`, `geometry`, `origin`, `color`
- For each joint: parent link (+binding), child link (-binding), axis

### 2.2 Write the Query File

Create `transforms/<name>/queries.hymeko`.

The query file uses the exact same HyMeKo grammar as descriptions. Each top-level element inside the `context` block becomes a **named query**. The element's name becomes the **label** you reference in the template.

```
my_transform {}
context
{
    // Label: "links"
    // Predicate: node AND inherits("link")
    links: link {}

    // Label: "revolute_joints"
    // Predicate: edge AND inherits("rev_joint")
    revolute_joints: rev_joint {}
}
```

**Rules:**

| Syntax in query file | Resulting predicate |
|---------------------|-------------------|
| `links: link {}` | Node, inherits from "link" |
| `_ : joint {}` | Any node/edge inheriting from "joint" |
| `@revolute: rev_joint {}` | Edge named "revolute" inheriting from "rev_joint" |
| `heavy_parts: link { mass <gt> 5.0; }` | Node inheriting "link" with mass > 5.0 |

The label (the name before the colon, or the node/edge name if no colon) is what you'll use in `{{#each label}}` in the template.

### 2.3 Write the Template File

Create `transforms/<name>/template.<ext>` (any extension).

The template is your target format with `{{...}}` interpolation tags.

**Minimal example — DOT:**

```dot
digraph {{config:robot_name}} {
{{#each links}}
  "{{name}}";
{{/each}}
{{#each revolute_joints}}
  "{{bind:+:0}}" -> "{{bind:-:0}}" [label="{{name}}"];
{{/each}}
}
```

**Full reference of template tags:**

#### Interpolation

| Tag | Context | Output |
|-----|---------|--------|
| `{{name}}` | Inside `{{#each}}` | Matched declaration's resolved name |
| `{{kind}}` | Inside `{{#each}}` | `"node"`, `"edge"`, or `"arc"` |
| `{{depth}}` | Inside `{{#each}}` | Depth in declaration tree (0 = root) |
| `{{id}}` | Inside `{{#each}}` | Raw DeclId index (for debugging) |
| `{{config:KEY}}` | Anywhere | Value from config map (e.g., robot_name) |

#### Field Access

| Tag | What it does |
|-----|-------------|
| `{{field:mass}}` | Find child named "mass" under the matched decl, return its value |
| `{{field:geometry.shape}}` | Dotted path: find "geometry" child, then "shape" under it |
| `{{field:origin}}` | If origin is a list `[1.0, 2.0, 3.0]`, renders as `"1.0 2.0 3.0"` |

Field resolution walks the IR tree:

```
matched_decl
├── mass 5.0            ← {{field:mass}} → "5"
├── link_geometry: cylinder
│   └── dimension [0.1, 0.3]   ← {{field:link_geometry.dimension}} → "0.1 0.3"
└── origin [0.0, 0.0, 0.5]    ← {{field:origin}} → "0 0 0.5"
```

If a field is missing, the tag renders as empty string `""`.

#### Arc Binding Access

For matched edges, arc bindings capture the signed references from the HyMeKo description.

Given `.hymeko`:
```
@j0: rev_joint {
    (+ base_link [[0.0, 0.0, 0.05], [0.0, 0.0, 0.0]], - link_0,
     - AXIS_Z);
}
```

The match captures:
- `bind:+:0` → `"base_link"` (first positive binding = parent)
- `bind:-:0` → `"link_0"` (first negative binding = child)
- `bind:-:1` → `"AXIS_Z"` (second negative binding = axis)
- `bind:-:all` → `"link_0 AXIS_Z"` (all negative, space-separated)

| Tag | Output |
|-----|--------|
| `{{bind:+:0}}` | First positive arc binding target name |
| `{{bind:-:0}}` | First negative arc binding target name |
| `{{bind:+:1}}` | Second positive binding (or empty if none) |
| `{{bind:-:all}}` | All negative bindings, space-separated |
| `{{bind:~:0}}` | First neutral binding |

#### Control Flow

**Iteration:**
```
{{#each links}}
  <link name="{{name}}"/>
{{/each}}
```

The label (`links`) must match a query label from `queries.hymeko`. Inside the block, `{{name}}`, `{{field:...}}`, `{{bind:...}}` resolve against each match in turn.

**Conditional:**
```
{{#if field:mass}}
  <mass value="{{field:mass}}"/>
{{/if}}
```

The block is emitted only if the field exists (is not empty). Works with any expression that resolves to a non-empty string.

**Comment:**
```
{{#comment}}
  This is stripped from the output entirely.
  Useful for template documentation.
{{/comment}}
```

### 2.4 Test It

```bash
# One-shot
hymeko transform robot.hymeko -t my_transform -o output.xml --name my_robot

# Or in the REPL
hymeko [robot]> tf my_transform output.xml
```

---

## 3. Walkthrough: Building an MJCF Transform

MuJoCo MJCF was previously a 180-line hardcoded Rust function. Let's rebuild it as a transform.

### 3.1 What MJCF Needs

MuJoCo's MJCF format structures robots as a nested `<body>` tree:

```xml
<mujoco model="robot">
  <worldbody>
    <body name="base">
      <inertial mass="5"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <body name="child">
        ...
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="j0_motor" joint="j0"/>
  </actuator>
</mujoco>
```

**Key difference from URDF:** MJCF is nested (body-in-body), URDF is flat (separate `<link>` and `<joint>` elements). Our template engine doesn't support recursive nesting (that would need a tree walker). So we use a flat approach with joints as connectors, same as URDF/SDF.

### 3.2 Query File

```bash
mkdir -p transforms/mjcf
```

`transforms/mjcf/queries.hymeko`:
```
mjcf_transform {}
context
{
    links: link {}
    frames: frame {}
    fixed_joints: fixed_joint {}
    revolute_joints: rev_joint {}
    continuous_joints: conti_joint {}
    prismatic_joints: prismatic_joint {}
}
```

### 3.3 Template File

`transforms/mjcf/template.mjcf.xml`:
```xml
<mujoco model="{{config:robot_name}}">
  <compiler angle="radian" meshdir="meshes/"/>
  <worldbody>
{{#each links}}
    <body name="{{name}}">
{{#if field:mass}}
      <inertial mass="{{field:mass}}" pos="0 0 0"/>
{{/if}}
{{#if field:link_geometry}}
      <geom type="cylinder" size="0.05 0.1"/>
{{/if}}
    </body>
{{/each}}
  </worldbody>
  <actuator>
{{#each revolute_joints}}
    <motor name="{{name}}_motor" joint="{{name}}" gear="1"/>
{{/each}}
{{#each continuous_joints}}
    <motor name="{{name}}_motor" joint="{{name}}" gear="1"/>
{{/each}}
  </actuator>
</mujoco>
```

### 3.4 Test

```
hymeko [anthropomorphic_arm]> tf mjcf
<mujoco model="robot">
  <compiler angle="radian" meshdir="meshes/"/>
  <worldbody>
    <body name="base_link">
      <inertial mass="25" pos="0 0 0"/>
      <geom type="cylinder" size="0.05 0.1"/>
    </body>
    ...
```

### 3.5 Total Effort

Two files, 30 lines. No Rust. No recompilation.

---

## 4. Walkthrough: Custom Transform for ROS 2 Launch File

This shows the engine isn't limited to XML robot descriptions.

### 4.1 Goal

Generate a ROS 2 Python launch file from a `.hymeko` robot description.

### 4.2 Query File

`transforms/ros2_launch/queries.hymeko`:
```
ros2_launch_transform {}
context
{
    links: link {}
    revolute_joints: rev_joint {}
    continuous_joints: conti_joint {}
    controllers: control {}
}
```

### 4.3 Template File

`transforms/ros2_launch/template.launch.py`:
```python
{{#comment}}
  Auto-generated ROS 2 launch file from HyMeKo description.
  Robot: {{config:robot_name}}
{{/comment}}
from launch import LaunchDescription
from launch_ros.actions import Node as RosNode

def generate_launch_description():
    ld = LaunchDescription()

    # Robot state publisher
    ld.add_action(RosNode(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='{{config:robot_name}}_state_publisher',
        parameters=[{'robot_description': '...'}],
    ))

    # Joint state broadcaster
    ld.add_action(RosNode(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
    ))

{{#each revolute_joints}}
    # Controller for {{name}}
    ld.add_action(RosNode(
        package='controller_manager',
        executable='spawner',
        arguments=['{{name}}_controller'],
    ))
{{/each}}

{{#each continuous_joints}}
    # Controller for {{name}}
    ld.add_action(RosNode(
        package='controller_manager',
        executable='spawner',
        arguments=['{{name}}_controller'],
    ))
{{/each}}

    return ld
```

**Result:** `hymeko transform robot.hymeko -t ros2_launch -o robot_launch.py`

---

## 5. Advanced Patterns

### 5.1 Multiple Geometry Types

When you need to switch on a field value (e.g., box vs cylinder vs sphere), the template engine doesn't have a `switch` block. Use conditional blocks on specific fields:

```xml
{{#if field:link_geometry.dimension}}
  {{#comment}} Has geometry with dimensions {{/comment}}
  <geometry>
    <cylinder radius="0.05" length="{{field:link_geometry.dimension}}"/>
  </geometry>
{{/if}}
```

For full geometry type switching, you have two options:

**Option A:** Separate query labels per geometry type:

```
box_links: link { link_geometry: box {} }
cylinder_links: link { link_geometry: cylinder {} }
sphere_links: link { link_geometry: sphere {} }
```

Then in the template:
```xml
{{#each box_links}}
  <geom type="box" size="{{field:link_geometry.dimension}}"/>
{{/each}}
{{#each cylinder_links}}
  <geom type="cylinder" size="{{field:link_geometry.dimension}}"/>
{{/each}}
```

**Option B:** Accept the approximation in the template (always emit cylinder), refine later.

### 5.2 Nested Field Paths

The `field:` expression supports dotted traversal:

```
robot
├── base_link: link
│   ├── mass 25.0
│   ├── link_geometry: cylinder
│   │   └── dimension [0.13, 0.05]
│   └── color -> link_color
└── link_color [1.0, 0.776, 0.0, 1.0]
```

- `{{field:mass}}` → `"25"`
- `{{field:link_geometry}}` → `"true"` (exists, but no scalar value)
- `{{field:link_geometry.dimension}}` → `"0.13 0.05"`

References are followed: if `color -> link_color`, then `{{field:color}}` follows the reference and returns the target's value.

### 5.3 Using Transforms Programmatically (Rust)

```rust
use hymeko_query::rewrite::{execute_transform, TransformSpec};
use std::collections::HashMap;

let spec = TransformSpec {
    name: "urdf".into(),
    query_source: std::fs::read_to_string("transforms/urdf/queries.hymeko")?,
    template_source: std::fs::read_to_string("transforms/urdf/template.urdf.xml")?,
};

let mut config = HashMap::new();
config.insert("robot_name".into(), "my_robot".into());

let output = execute_transform(&compiled.ir, &interner, &spec, &config)?;
```

### 5.4 Config Variables

The `config` map passes external parameters to the template. Currently the CLI sets `robot_name`, but you can extend it:

```rust
config.insert("robot_name".into(), "my_robot".into());
config.insert("author".into(), "Dr. Hajdu".into());
config.insert("version".into(), "1.0".into());
config.insert("date".into(), "2026-04-13".into());
```

Use in template: `{{config:author}}`, `{{config:version}}`.

---

## 6. Pipeline Diagram

```
  ┌──────────────────┐
  │ .hymeko source   │ (robot description)
  └────────┬─────────┘
           │ ModuleStore::compile()
           ▼
  ┌──────────────────┐
  │ Compiled IR      │ (DeclNodes, Edges, Arcs, Interner)
  └────────┬─────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────┐
│ queries │  │ template │  (from transforms/<name>/)
│ .hymeko │  │ .xml     │
└────┬────┘  └────┬─────┘
     │            │
     ▼            │
 parse_description()
     │            │
     ▼            │
 interpret_as_queries()
     │            │
     ▼            │
 Vec<NamedQuery>  │
     │            │
     ▼            │
 QueryEngine      │
 .query_batch()   │
     │            │
     ▼            │
 HashMap<label,   │
  Vec<QueryMatch>>│
     │            │
     ▼            ▼
  ┌──────────────────────────┐
  │ render(template, results)│
  │                          │
  │  For each {{#each label}}│
  │    MatchContext extracts  │
  │    fields from IR        │
  │    bind:+:0 reads arcs   │
  │    field:mass reads      │
  │    child values          │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────┐
  │ output string    │ (URDF XML, SDF XML, DOT, Python, ...)
  └──────────────────┘
```

---

## 7. Adding a New Transform: Checklist

```
[ ] 1. mkdir transforms/<name>/
[ ] 2. Write transforms/<name>/queries.hymeko
       - One top-level element per query category
       - Labels match what you need in the template
[ ] 3. Write transforms/<name>/template.<ext>
       - {{#each label}} for each query category
       - {{name}}, {{field:X}}, {{bind:+:0}} for data
       - {{#if field:X}} for optional sections
       - {{config:robot_name}} for parameters
[ ] 4. Test:
       hymeko [robot]> tf <name>
       hymeko [robot]> tf <name> output_file
[ ] 5. Verify output in target tool (Gazebo, MuJoCo, etc.)
```

No Rust code. No recompilation. No PR to `hymeko_query`.

---

## 8. Current Limitations and Future Work

### What Works Now

- Flat iteration over query results (`{{#each}}`)
- Field extraction with dotted paths (`{{field:a.b.c}}`)
- Signed arc bindings (`{{bind:+:0}}`, `{{bind:-:all}}`)
- Conditional blocks (`{{#if field:X}}`)
- Config variable injection (`{{config:KEY}}`)
- Reference following in field paths

### What Doesn't (Yet)

| Limitation | Workaround | Future |
|-----------|-----------|--------|
| No recursive nesting (MJCF's body-in-body) | Flatten with joints as connectors | Add `{{#tree root_label child_label}}` block |
| No `{{#switch}}` or `{{#else}}` | Use separate query labels per case | Add `{{#else}}` to `{{#if}}` |
| No arithmetic in templates | Pre-compute in config or field values | Add `{{expr:field:x * 0.5}}` |
| No cross-query joins | Use field paths that follow references | Add `{{#with label match_name}}` |
| Template doesn't know geometry type | Separate queries per shape | Add `{{field:link_geometry.@kind}}` for type tag |
| No indentation control | Manual spacing in template | Add `{{#indent N}}` |

### Roadmap

| Priority | Item | Target |
|----------|------|--------|
| 1 | `{{#else}}` for conditionals | Next session |
| 2 | `{{#tree}}` for recursive MJCF nesting | Post-COINS |
| 3 | Expression evaluation (`{{expr:...}}`) | Post-SMC |
| 4 | Transform validation CLI (`hymeko check-transform`) | May |
| 5 | Transform composition (pipe one transform into another) | June |
| 6 | Hot-reload transforms in daemon mode | July (Nagoya) |

---

## 9. Relationship to the Paper

For **IEEE COINS** (April 15): The "query-as-description" story is now complete. The same HyMeKo grammar describes hypergraphs (robot structure), queries (patterns over the structure), AND transforms (what to do with matches). One grammar, three roles. This is a clean architectural contribution.

For **IEEE SMC Regular** (April 19): The compilation pipeline now has a clear separation: `hymeko_core` (IR + tensor + computation), `hymeko_query` (generic query engine + rewrite), external transforms (domain-specific). Benchmarks run on the core, not on format-specific code.

The old `codegen.rs` → `formats/urdf.rs` path is still there and works. Keep it as the "reference implementation" for the paper. The rewrite engine is the generalization that goes into Section 5 ("Extensibility").
