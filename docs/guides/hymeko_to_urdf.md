# Tutorial — Rewriting a HyMeKo Description to URDF

This tutorial walks you end-to-end from a HyMeKo robot description to a
working URDF file you can drop into ROS, RViz, or Gazebo. It assumes
nothing besides a built `hymeko` binary and the `transforms/urdf/`
directory shipped with the framework.

If you want the deeper reference on *how* transforms work, read
[`transforms.md`](transforms.md). This file is the **5-minute hands-on
path**.

---

## What you will build

A two-link arm (a base and a single rotating wheel-like link) joined by
a continuous joint. By the end you will:

1. Understand the minimum set of HyMeKo declarations URDF needs.
2. Run the URDF transform on it and read the output.
3. Repeat the same flow on a richer example (`robot_4wh.hymeko`).
4. Know which knobs to turn when the URDF is missing something.

---

## Step 0 — Verify the toolchain

```bash
cargo build --release -p hymeko_cli
ls transforms/urdf/                # queries.hymeko + template.urdf.xml
./target/release/hymeko --help     # confirm the binary works
```

You should see `transform` listed as a subcommand.

---

## Step 1 — Author a minimal HyMeKo description

Create `data/robotics/mini_arm.hymeko`:

```hymeko
mini_arm_description {
    @"meta_kinematics.hymeko";
}

mini_arm: meta_kinematics.kinematics.elements,
          meta_kinematics.kinematics.geometry,
          meta_kinematics.kinematics.axes
{
    base_link: meta_kinematics.kinematics.elements.link {
        mass 5.0;
        link_geometry: meta_kinematics.kinematics.geometry.box {
            dimension [0.3, 0.3, 0.1];
        }
        visual    -> link_geometry;
        collision -> link_geometry;
        origin [0.0, 0.0, 0.05];
    }

    spinner: meta_kinematics.kinematics.elements.link {
        mass 1.0;
        link_geometry: meta_kinematics.kinematics.geometry.cylinder {
            dimension [0.1, 0.2];
        }
        visual    -> link_geometry;
        collision -> link_geometry;
        origin [0.0, 0.0, 0.1];
    }

    @spin_joint: meta_kinematics.kinematics.conti_joint {
        (+ base_link [[0.0, 0.0, 0.1], [0.0, 0.0, 0.0]],
         - spinner,
         - meta_kinematics.kinematics.axes.AXIS_Z);
    }
}
```

The pieces a URDF transform actually consumes:

| Construct | URDF role |
|-----------|-----------|
| `link` decl | becomes `<link>` |
| `mass` field | becomes `<inertial><mass/>` |
| `link_geometry` field | gates the `<visual>` / `<collision>` block |
| `@<…>_joint` edge | becomes `<joint>` |
| `+ parent`, `- child` bindings | become `<parent link/>`, `<child link/>` |

Anything else (controllers, sensors, topics, gazebo plugins) is silently
ignored by the URDF template. Strip-it-down rule: **if URDF doesn't have
a tag for it, the URDF transform won't either.**

---

## Step 2 — Run the transform

One-shot, write to a file:

```bash
./target/release/hymeko transform \
    data/robotics/mini_arm.hymeko \
    -t urdf \
    --name mini_arm \
    -o mini_arm.urdf
```

Or interactively in the REPL:

```text
./target/release/hymeko
hymeko> load data/robotics/mini_arm.hymeko
hymeko [mini_arm_description]> name mini_arm
hymeko [mini_arm_description]> tf urdf mini_arm.urdf
```

The flag `--name` (or REPL `name <…>`) sets `{{config:robot_name}}`,
which lands as `<robot name="…">`.

---

## Step 3 — Read the output

`mini_arm.urdf` will look like:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<robot name="mini_arm">
  <link name="base_link">
    <inertial>
      <mass value="5"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <link name="spinner">…</link>

  <joint name="spin_joint" type="continuous">
    <parent link="base_link"/>
    <child link="spinner"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
```

Two things to notice — both are **template-level shortcuts**, not bugs in
your description:

1. The geometry is hard-coded to `<cylinder radius="0.05" length="0.1"/>`
   regardless of what you wrote. The shipped template is a placeholder;
   Step 5 shows how to make it honest.
2. `<axis xyz="0 0 1"/>` is hard-coded too. Same reason.

This is by design. The transform engine is mechanical; the *URDF
conventions* live entirely inside `transforms/urdf/template.urdf.xml`,
and you are expected to edit that template to fit your project.

---

## Step 4 — Try the bigger example

```bash
./target/release/hymeko transform \
    data/robotics/robot_4wh.hymeko \
    -t urdf \
    --name diff_robot \
    -o robot_4wh.urdf
```

You'll get a 6-link, 5-joint URDF (1 fixed + 4 continuous). Notice that
all the controller / sensor / topic declarations from `robot_4wh.hymeko`
are dropped — the URDF queries (`transforms/urdf/queries.hymeko`) only
ask for links and joint edges:

```hymeko
urdf_transform {}
context
{
    links: link {}
    @fixed_joints: fixed_joint {}
    @revolute_joints: rev_joint {}
    @continuous_joints: conti_joint {}
    @prismatic_joints: prismatic_joint {}
    frames: frame {}
}
```

If something you want is missing from the URDF, the cause is almost
always one of:

- the entity isn't a `link` or a joint-typed edge, or
- the template doesn't reference the field.

---

## Step 5 — Make geometry actually reflect the description

Open `transforms/urdf/template.urdf.xml` and replace the hard-coded
geometry block. Split links by geometry kind in `queries.hymeko`:

```hymeko
urdf_transform {}
context
{
    box_links:      link { link_geometry: box {} }
    cylinder_links: link { link_geometry: cylinder {} }
    sphere_links:   link { link_geometry: sphere {} }

    @fixed_joints:      fixed_joint {}
    @revolute_joints:   rev_joint {}
    @continuous_joints: conti_joint {}
    @prismatic_joints:  prismatic_joint {}
}
```

Then in the template emit one `{{#each}}` block per geometry kind:

```xml
{{#each box_links}}
  <link name="{{name}}">
    <visual>
      <origin xyz="{{field:origin}}" rpy="0 0 0"/>
      <geometry><box size="{{field:link_geometry.dimension}}"/></geometry>
    </visual>
  </link>
{{/each}}

{{#each cylinder_links}}
  <link name="{{name}}">
    <visual>
      <origin xyz="{{field:origin}}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{{field:link_geometry.dimension}}"
                  length="{{field:link_geometry.dimension}}"/>
      </geometry>
    </visual>
  </link>
{{/each}}
```

(Splitting `[radius, length]` into two scalars is a known sharp edge —
list values render space-separated. For now, dump both into the
attributes and let URDF parsers complain, or pre-split the dimensions
into named children in your description.)

Re-run the transform — geometry now matches the description.

---

## Step 6 — Validate the URDF

```bash
check_urdf mini_arm.urdf                        # ROS' urdf_parser
ros2 launch urdf_tutorial display.launch.py \
     model:=$(pwd)/mini_arm.urdf                # visualise in RViz
```

Both are external tools — HyMeKo is intentionally agnostic about what
consumes its output.

---

## When it doesn't work

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `<robot name="robot">` instead of your name | Forgot `--name` / `name` in REPL | Pass `--name my_robot` |
| Joint missing from output | Joint declared as a node, not an edge | Prefix with `@`: `@joint_x: …` |
| Link missing from output | Link doesn't inherit from `meta_kinematics.kinematics.elements.link` | Inherit it, or extend `queries.hymeko` to match your base type |
| `<parent link=""/>` empty | Joint binding lacks a `+` (parent) target | `(+ parent_link, - child_link, - AXIS_Z)` |
| Geometry always cylinder 0.05/0.1 | Shipped template is a placeholder | Edit `transforms/urdf/template.urdf.xml` per Step 5 |
| Field comes back blank | Field name typo, or the field was set on a *referenced* node | Drop into the REPL: `query` to inspect matches, `qfile` for ad-hoc queries |

---

## Where to go next

- **Reference**: [`transforms.md`](transforms.md) — full tag table,
  control-flow syntax, programmatic API.
- **Other targets**: `transforms/{sdf, mjcf, dot, ros2_launch}/` follow
  the same recipe; copy one as a starting point.
- **Engine internals**: `docs/plans/04_graph_query/T11_rewrite_engine.md`
  if you need to understand or extend the matcher / template engine
  itself.

URDF is just one rendering of the underlying graph — once the
description is good, every other format is one transform away.