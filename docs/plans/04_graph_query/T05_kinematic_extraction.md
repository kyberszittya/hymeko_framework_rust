# T05 — Kinematic Model Extraction

**Status:** ✅ DONE  
**File:** `hymeko_core/src/query/kinematic.rs` (390 lines)

---

## Purpose

Shared intermediate between all output formats (URDF, SDF, Gazebo, Isaac Sim). Instead of each format independently navigating the IR, they all consume a `KinematicModel`. This is the DRY principle applied to domain transforms.

## Data Structures

```rust
KinematicModel
├── name: String
├── links: Vec<LinkInfo>
│   ├── did: DeclId
│   ├── name: String
│   ├── mass: Option<f64>
│   ├── geometry: Option<GeometryInfo>     // box/cylinder/sphere + dimensions
│   ├── origin: Option<Vec<f64>>           // [x, y, z] or [x, y, z, r, p, y]
│   └── color: Option<Vec<f64>>            // [r, g, b, a] via ref-following
│
└── joints: Vec<JointInfo>
    ├── did: DeclId
    ├── name: String
    ├── joint_type: JointType              // Fixed/Continuous/Revolute/Prismatic
    ├── parent_link: String                // from + ref to link
    ├── child_link: String                 // from - ref to link
    ├── axis: Option<[f64; 3]>             // from - ref to axis_definition
    ├── origin_xyz: Option<[f64; 3]>       // from + ref weight annotation
    ├── origin_rpy_deg: Option<[f64; 3]>   // degrees, from + ref weight annotation
    └── limits: Option<JointLimits>        // TODO: extract from inherited limits
```

## How Joint Topology Is Extracted

Given a joint edge like:
```
@joint_fr: kinematics.conti_joint {
    + base_link, [[0.25, 0.25, 0.05], [-90.0, 0.0, 0.0]] - wheel_fr,
    - AXIS_Z
}
```

The extraction logic in `extract_joint_topology()`:

1. Iterates all arcs in the edge's `EdgeRec.arcs`
2. For each signed ref in each arc:
   - **Sign +1, target inherits `link`** → parent link. Reads `atom.weights` for `[[x,y,z],[r,p,y]]` origin.
   - **Sign -1, target inherits `link`** → child link.
   - **Sign -1, target inherits `axis_definition`** → axis. Reads child value `ax` on the axis node.

The link vs axis disambiguation uses `check_inherits_simple()`, a standalone function that walks the base chain without needing a full `QueryEngine`.

## Weight Annotation Extraction

The `[[0.25, 0.25, 0.05], [-90.0, 0.0, 0.0]]` on the plus-ref becomes:

```
atom.weights = Some(vec![
    ValueR::List(vec![Num(0.25), Num(0.25), Num(0.05)]),
    ValueR::List(vec![Num(-90.0), Num(0.0), Num(0.0)])
])
```

`extract_origin_from_weights()` reads `weights[0]` as XYZ and `weights[1]` as RPY (degrees).

## Geometry Extraction

Looks for a child named `link_geometry` that inherits from `box`/`cylinder`/`sphere`:

```
base_link: kinematics.elements.link {
    link_geometry: box {
        dimension [0.7, 0.5, 0.2];
    }
}
```

→ `GeometryInfo { shape: Box, dimensions: [0.7, 0.5, 0.2] }`

## Color via Ref-Following

```
color -> diff_robot.body_color;
```

The `color` child has `ValueR::Ref(did)` pointing to `body_color`. `find_child_ref_value_list()` follows this ref to extract the actual `[r, g, b, a]` values.

## Degree-to-Radian Conversion

`JointInfo::origin_rpy_rad()` converts RPY from degrees (as stored in the source) to radians (as URDF requires). This is done at the `JointInfo` level, not in the extraction, so SDF/Gazebo can also use it.
