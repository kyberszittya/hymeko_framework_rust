# T06–T07 — Domain Transforms (URDF + SDF)

**Status:** ✅ DONE  
**Files:** `hymeko_core/src/query/urdf.rs` (189 lines), `hymeko_core/src/query/sdf.rs` (114 lines)

---

## T06 — URDF Generation

### Public API

```rust
/// Predefined queries for URDF generation (6 patterns).
pub fn urdf_queries() -> Vec<NamedQuery>;

/// Generate URDF XML from compiled IR.
pub fn generate_urdf<R: NameResolver>(ir: &Ir, resolver: &R, robot_name: &str) -> String;

/// Validate that all joint parent/child refs point to known links.
pub fn validate_robot_schema<R: NameResolver>(ir: &Ir, resolver: &R) -> Vec<String>;
```

### Query Patterns

| Label | Predicate | Expected (robot_4wh) |
|-------|-----------|---------------------|
| `links` | `node() ∧ inherits("link")` | 6 |
| `fixed_joints` | `edge() ∧ inherits("fixed_joint")` | 1 |
| `continuous_joints` | `edge() ∧ inherits("conti_joint")` | 4 |
| `revolute_joints` | `edge() ∧ inherits("rev_joint")` | 0 |
| `prismatic_joints` | `edge() ∧ inherits("prismatic_joint")` | 0 |
| `axes` | `node() ∧ inherits("axis_definition")` | 4 |

### Output Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<robot name="diff_robot_4wh">
  <link name="base_link">
    <inertial><mass value="25"/></inertial>
    <visual>
      <origin xyz="0 0 0.05"/>
      <geometry><box size="0.7 0.5 0.2"/></geometry>
      <material name="color"><color rgba="0 0 1 1"/></material>
    </visual>
    <collision>...</collision>
  </link>

  <joint name="joint_fr" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_fr"/>
    <origin xyz="0.25 0.25 0.05" rpy="-1.5708 0.0000 0.0000"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
```

### Conversions

- RPY: degrees → radians (`× π/180`)
- Axis: float → integer cast for clean output (`1.0` → `1`)
- XML escaping on all attribute values

### Schema Validation

`validate_robot_schema()` checks that every joint's `parent_link` and `child_link` names exist in the link set. Returns a `Vec<String>` of error messages (empty = valid).

---

## T07 — SDF Generation

### Differences from URDF

| Feature | URDF | SDF 1.7 |
|---------|------|---------|
| Origin | `<origin xyz="..." rpy="..."/>` | `<pose relative_to="parent">x y z r p y</pose>` |
| Inertia | `<mass value="25"/>` | `<mass>25</mass>` + `<inertia>` matrix |
| Continuous joint | `type="continuous"` | `type="revolute"` with `<limit>` ±1e16 |
| Geometry | Self-closing tags | Nested `<size>`, `<radius>`, `<length>` elements |
| Visual/Collision names | Anonymous | Named (`base_link_visual`) |

### Public API

```rust
pub fn generate_sdf<R: NameResolver>(ir: &Ir, resolver: &R, model_name: &str) -> String;
```

### Inertia Approximation

SDF requires a full `<inertia>` matrix. Currently uses diagonal approximation:

```
ixx = iyy = izz = mass × 0.01
ixy = ixz = iyz = 0
```

This is a placeholder. Proper inertia computation from geometry dimensions is a post-deadline improvement.

---

## Remaining Domain Transforms

### T11 — Gazebo World Configuration (❌ NOT DONE)

Needs to extract `sim_plugin` and `control_plugin` edges from the IR:

```
@sim_control_plugin: kinematics.sim_plugin {
    plugin "gz_ros2_control::GazeboSimROS2ControlPlugin",
    filename "gz_ros2_control-system",
    parameters "diff_control.yaml"
}
```

Output: `<world>` with `<include>` for the SDF model + `<plugin>` elements for ros2_control.

### T12 — Isaac Sim USD Export (❌ NOT DONE)

Needs USD Prim hierarchy with `UsdPhysics.ArticulationRootAPI` and joint drive parameters. Significantly more complex than XML-based formats.
