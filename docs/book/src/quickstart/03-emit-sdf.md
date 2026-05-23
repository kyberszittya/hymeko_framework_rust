# Quickstart: Emit SDF for Gazebo

Goal: produce a Gazebo-compatible SDF 1.7 file from the same `.hymeko` source you used for URDF.

```bash
target/release/hymeko emit \
    data/robotics_imported/wam/wam.hymeko \
    --format sdf \
    --name wam7 \
    -o /tmp/wam7.sdf
```

Output:

```
Wrote 3812 bytes to /tmp/wam7.sdf
```

The structure mirrors the URDF emit but produces SDF schema (`<sdf version="1.7">` / `<model>` / nested geometry under `<visual>` and `<collision>`). Internally both formats share the **same KinematicModel extraction** — only the emission template differs.

## Drop into a Gazebo world

```bash
gz sim --headless-rendering /tmp/wam7.sdf
```

Or wrap it in a world file:

```xml
<sdf version="1.7">
  <world name="default">
    <include><uri>/tmp/wam7.sdf</uri></include>
  </world>
</sdf>
```

For full Gazebo worlds, see also [`hymeko emit --format gazebo_world`](../recipes/add-a-format.md) — a separate transform that produces world XML directly from a `.hymeko` scene description.

## URDF vs SDF differences

| concern | URDF | SDF |
|---|---|---|
| origin convention | `<origin xyz rpy>` | `<pose>x y z rx ry rz</pose>` |
| visual/collision | inline under `<link>` | named child elements `<visual name>` |
| inertia | full 3×3 spec | also full but different element names |
| world / simulation | not part of URDF | first-class in SDF |

The HyMeKo IR holds the canonical kinematic structure; both emitters express it in their target's conventions.

## Same thing in Python

```python
import hymeko

src = open("data/robotics_imported/wam/wam.hymeko").read()
doc = hymeko.compile_description(src)
sdf_xml = doc.to_sdf("wam7")
open("/tmp/wam7.sdf", "w").write(sdf_xml)
```

## Next

- [Emit MJCF for MuJoCo](./04-emit-mjcf.md)
- [Generate a PyTorch nn.Module](./06-emit-torch.md)
