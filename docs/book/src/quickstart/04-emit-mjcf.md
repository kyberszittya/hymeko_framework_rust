# Quickstart: Emit MJCF for MuJoCo

```bash
target/release/hymeko emit \
    data/robotics_imported/wam/wam.hymeko \
    --format mjcf \
    --name wam7 \
    -o /tmp/wam7.xml
```

MJCF is MuJoCo's native format; structure differs from URDF/SDF (it uses an explicit `<worldbody>` containing nested `<body>` elements with parent-child relationships, no separate joint declarations — joints live inside their child body).

## Run in MuJoCo

```python
import mujoco
model = mujoco.MjModel.from_xml_path("/tmp/wam7.xml")
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
print(data.qpos)  # initial joint positions
```

## Caveats

The MJCF emitter currently produces a kinematic structure suitable for forward simulation but does not auto-generate:
- Friction / contact pair definitions
- Actuator declarations (you need to add `<actuator>` blocks manually)
- Sensor definitions

If you need those, augment the emitted file post-hoc, or model them in HyMeKo and extend the MJCF template (`transforms/mjcf/template.xml`). See [Add a new format](../recipes/add-a-format.md) for how the template engine consumes the IR.

## Next

- [Generate a PyTorch nn.Module](./06-emit-torch.md) — a non-robotics emit target
- [Add a new format](../recipes/add-a-format.md)
