# Quickstart: Emit URDF for ROS

Goal: take a `.hymeko` description of a robot and produce a URDF file ROS can consume.

## Pick an existing example

The repo ships a real Atlas-class humanoid (drchubo) in `data/robotics_imported/drchubo/`. For this tutorial use the smaller WAM 7-DOF arm:

```bash
ls data/robotics_imported/wam/
# meta_kinematics.hymeko  wam.hymeko
```

## Emit

```bash
target/release/hymeko emit \
    data/robotics_imported/wam/wam.hymeko \
    --format urdf \
    --name wam7 \
    -o /tmp/wam7.urdf
```

Output:

```
Wrote 4218 bytes to /tmp/wam7.urdf
```

Inspect:

```bash
head -20 /tmp/wam7.urdf
# <?xml version="1.0" encoding="UTF-8"?>
# <robot name="wam7">
#   <link name="base_link">
#     <inertial>
#       <mass value="1.0"/>
#       ...
#     </inertial>
#   </link>
#   ...
```

## What happened under the hood

1. **Parse + resolve** the `.hymeko` (same as [Quickstart 1](./01-parse.md)).
2. **Extract a `KinematicModel`** by querying the IR for nodes inheriting from `link` and edges inheriting from `fixed_joint` / `rev_joint` / `prismatic_joint` / `conti_joint`.
3. **Emit URDF XML** by walking the model and writing `<robot>` / `<link>` / `<joint>` blocks.

The same code path runs whether you use the CLI, the Python wheel (`PyHypergraphIR.to_urdf("wam7")`), or the WASM demo. After the recent **EmissionPipeline cleanup** (May 2026), the IR-taking and model-taking entry points share a single emission step:

```rust
// hymeko_formats/src/urdf.rs
pub fn generate_urdf<R: NameResolver>(ir: &Ir, resolver: &R, robot_name: &str) -> String {
    let engine = QueryEngine::new(ir, resolver);
    let model = extract_kinematic_model(&engine, robot_name);
    generate_urdf_from_model(&model)   // single source of truth
}
```

## Visual check

```bash
sudo apt install ros-humble-urdfdom  # one-time
check_urdf /tmp/wam7.urdf

# robot name is: wam7
# ---------- Successfully Parsed XML ---------------
# root Link: base_link has 1 child(ren)
#     child(1):  shoulder_link
#         child(1):  upper_arm_link
#         ...
```

Or render to PNG via the meshcat / RViz pipeline.

## Same thing in Python

```python
import hymeko

src = open("data/robotics_imported/wam/wam.hymeko").read()
ir = hymeko.parse_hymeko_rs(src)
# Compile + emit:
doc = hymeko.compile_description(src)
urdf_xml = doc.to_urdf("wam7")
print(urdf_xml[:200])
```

## Adding inertial / collision data

The URDF emitter respects whatever the IR carries. To add mass to a link, set the `mass` field in the `.hymeko`:

```hymeko
shoulder_link: kin.link {
    mass 2.5;
    geometry kin.cylinder { radius 0.05; length 0.3; }
}
```

Re-emit and the `<inertial>` block appears in the URDF.

## Next

- [Emit SDF for Gazebo](./03-emit-sdf.md) — same `.hymeko`, different sim target
- [Emit MJCF for MuJoCo](./04-emit-mjcf.md) — same `.hymeko`, MuJoCo target
- [Add a new format](../recipes/add-a-format.md) — extend beyond URDF/SDF/MJCF
