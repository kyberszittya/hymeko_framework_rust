# Quickstart: Emit SysML 2 textual

The `sysml` format produces SysML 2 textual concrete syntax — the modern (2024-stable) MBSE notation consumed by [Eclipse Papyrus](https://www.eclipse.org/papyrus/), [Modelix](https://modelix.org/), and the [OMG SysML v2 Playground](https://sysml-v2.github.io/playground.html).

```bash
target/release/hymeko emit \
    data/robotics/mini_arm.hymeko \
    --format sysml \
    --name mini_arm \
    -o /tmp/mini_arm.sysml
```

## What it emits

```sysml
package mini_arm {

    part def Link {
        attribute mass : Real;
    }

    part def ContinuousJoint {
        end parent : Link;
        end child  : Link;
        attribute axis : Vector3;
    }

    part base_link : Link {
        :>> mass = 5;
    }

    part spinner : Link {
        :>> mass = 1;
    }

    connection spin_joint : ContinuousJoint {
        end ::> base_link;
        end ::> spinner;
    }
}
```

## The mapping

| HyMeKo | SysML 2 |
|---|---|
| `link` decl (Node) | `part def Link` instance via `part name : Link` |
| `mass` field | `:>> mass = <value>;` redefinition |
| `fixed_joint` edge | `connection name : FixedJoint { end ::> parent; end ::> child; }` |
| `rev_joint` / `conti_joint` / `prismatic_joint` | matching connection def |
| `+arc-ref` | first connection endpoint (parent link) |
| `-arc-ref` | second connection endpoint (child link) |

Larger robots (e.g. WAM 7-DOF arm at `data/robotics_imported/wam/wam.hymeko`) emit a complete package with 8 links + 7 RevoluteJoint connections.

## Why this matters

- **MBSE interop**: SysML 2 is the standard model-based systems engineering notation. Adding it as an emit target gives the framework a path into model-based systems audiences without leaving the HyMeKo toolchain.
- **Hypergraph affinity**: SysML 2 itself is a hypergraph notation (parts, ports, connections, requirements). The HyMeKo IR's signed hyperedges map naturally to SysML connections — the `+arc-ref` / `-arc-ref` semantics translate directly to typed connection endpoints.
- **Two-way bridge potential**: SysML 2 textual is a parseable language. A future `hymeko_sysml::parse` (mirroring `hymeko_urdf::parse`) would make HyMeKo a translation hub: URDF ↔ HyMeKo ↔ SysML.

## Inspect in a SysML 2 tool

The output is plain text — drop `/tmp/mini_arm.sysml` into:
- The OMG playground (paste into the editor pane)
- Eclipse Papyrus (open as a SysML 2 textual file)
- Modelix (import via the SysML 2 importer)

## Customising the template

The emitter is template-driven — `transforms/sysml/template.sysml` is the single source of truth. To change the mapping (e.g. add `axis`-attribute redefinitions, emit ports for joints, group parts by subsystem), edit that file. No Rust changes needed.

## Caveats

- The current template hardcodes joint type mappings (Fixed/Revolute/Continuous/Prismatic). Custom joint kinds in your `meta_kinematics.hymeko` won't be auto-recognised; see [Add a new layer kind](../recipes/add-a-layer-kind.md) for the schema-extension pattern (the same approach applies to joint kinds).
- SysML 2 textual is the target; the `.sysml` extension (and JSON / XMI variants) are not currently emitted. The textual form is the most stable and human-readable surface.

## Next

- [Emit URDF for ROS](./02-emit-urdf.md) — same `.hymeko`, different MBSE target
- [Emit DOT for visualization](./05-emit-dot.md)
- [Add a new format](../recipes/add-a-format.md) — extend beyond what ships
