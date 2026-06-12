// Kinematic stereotype view — shows BOTH (per the user's choice):
//   • a true robot render (three.js) from the emitted URDF (geometric truth), and
//   • the αₖ regime compass + signed-topology ring (reused regime canvases).
// No WASM change: URDF comes from ir.to_urdf("robot"); the regime panels come
// from the snapshot via the pure adapters.
//
// View contract: { name, mount(container), render(snapshot, ir), unmount() }.
import { snapshotToKinematicGraph, cycleArity, parseUrdf } from "./adapters.js?v=13";
import { Compass, Topo } from "./regime_classes.js?v=13";

const LINK_COLOR = 0x7dd3fc;

export function createKinematicView() {
  let host, robotCanvas, note, compassCv, topoCv;
  let THREE, scene, camera, renderer, raf = 0;
  let robotGroup = null, compass = null, topo = null;
  let theta = 0.8, phi = 1.1, radius = 3, target = null, autoRot = true;
  let drag = false, lx = 0, ly = 0, moved = false;
  let t0 = 0, light = true;
  let onUp = null, onMove = null; // window handlers, removed on unmount
  let raycaster = null, pointer = null, tooltip = null, selBox = null;
  let pickMeshes = [], selectedMesh = null;
  const BG_DARK = 0x0e1320, BG_LIGHT = 0xf7f7f8;

  function mount(container) {
    host = container; host.innerHTML = "";
    const root = el("div", "kin-root");
    const main = el("div", "kin-main");
    const tb = el("div", "view3d-toolbar");
    const bgBtn = btn("BG", () => { light = !light; if (scene) scene.background = new THREE.Color(light ? BG_LIGHT : BG_DARK); });
    tb.append(bgBtn);
    robotCanvas = el("canvas", "kin-canvas");
    note = el("div", "kin-note");
    tooltip = el("div", "view3d-tip");
    selBox = el("div", "view3d-sel");
    main.append(tb, robotCanvas, note, tooltip, selBox);
    const side = el("div", "kin-side");
    side.append(panel("αₖ regime", (c) => (compassCv = c)), panel("signed topology", (c) => (topoCv = c)));
    root.append(main, side);
    host.append(root);

    THREE = globalThis.THREE;
    if (!THREE) { note.textContent = "three.js failed to load (offline?)"; return; }
    scene = new THREE.Scene(); scene.background = new THREE.Color(light ? BG_LIGHT : BG_DARK);
    camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
    renderer = new THREE.WebGLRenderer({ canvas: robotCanvas, antialias: true });
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const d = new THREE.DirectionalLight(0xffffff, 0.7); d.position.set(1, 2, 1); scene.add(d);
    scene.add(new THREE.GridHelper(4, 16, 0x9ab, 0xccd));
    target = new THREE.Vector3(0, 0, 0);
    raycaster = new THREE.Raycaster(); pointer = new THREE.Vector2();
    compass = new Compass(compassCv); topo = new Topo(topoCv);
    bindPointer();
    bindPicking();
    t0 = performance.now();
    loop();
  }

  function render(snapshot, ir) {
    // Regime panels from the snapshot.
    const g = snapshotToKinematicGraph(snapshot || {});
    const cpa = cycleArity(g);
    if (compass) compass.set(cpa);
    if (topo) topo.set({ n_links: g.nLinks, n_joints: g.joints.length, edges: g.edges, joints: g.joints });
    if (!THREE) return;
    // Robot render from the emitted URDF.
    if (robotGroup) { scene.remove(robotGroup); disposeGroup(robotGroup); robotGroup = null; }
    let urdf = "";
    try { urdf = ir && ir.to_urdf ? ir.to_urdf("robot") : ""; } catch (_e) { urdf = ""; }
    const { links, joints } = parseUrdf(urdf);
    robotGroup = buildRobot(links, joints);
    scene.add(robotGroup);
    fitCamera();
    const withGeo = links.filter((l) => l.geometry).length;
    note.textContent = withGeo
      ? `${links.length} links (${withGeo} with geometry) · ${joints.length} joints`
      : `${links.length} links · no <visual> geometry in source — showing joint frames`;
  }

  function buildRobot(links, joints) {
    const grp = new THREE.Group();
    pickMeshes = []; selectedMesh = null; if (selBox) selBox.style.display = "none";
    const world = placeLinks(links, joints);
    for (const link of links) {
      const node = new THREE.Group();
      node.applyMatrix4(world.get(link.name));
      const mesh = link.geometry ? meshFor(link.geometry) : axes(0.06);
      if (link.geometry) {
        mesh.position.set(link.origin[0], link.origin[1], link.origin[2]);
        mesh.userData = { kind: "link", name: link.name, geometry: link.geometry };
        pickMeshes.push(mesh);
      }
      node.add(mesh);
      grp.add(node);
    }
    // Joint frames as small markers (always, for orientation).
    for (const j of joints) {
      const w = world.get(j.child); if (!w) continue;
      const a = axes(0.04); a.applyMatrix4(w); grp.add(a);
    }
    return grp;
  }

  function placeLinks(links, joints) {
    const childOf = new Map(); const childrenOf = new Map();
    for (const j of joints) {
      if (j.child) childOf.set(j.child, j);
      if (j.parent) { if (!childrenOf.has(j.parent)) childrenOf.set(j.parent, []); childrenOf.get(j.parent).push(j); }
    }
    const world = new Map(); const queue = [];
    for (const l of links) if (!childOf.has(l.name)) { world.set(l.name, new THREE.Matrix4()); queue.push(l.name); }
    while (queue.length) {
      const name = queue.shift();
      for (const j of childrenOf.get(name) || []) {
        if (!j.child || world.has(j.child)) continue;
        const T = originMatrix(j.origin_xyz, j.origin_rpy);
        world.set(j.child, world.get(name).clone().multiply(T)); queue.push(j.child);
      }
    }
    for (const l of links) if (!world.has(l.name)) world.set(l.name, new THREE.Matrix4());
    return world;
  }

  function originMatrix(xyz, rpy) {
    const m = new THREE.Matrix4();
    m.makeRotationFromEuler(new THREE.Euler(rpy[0] || 0, rpy[1] || 0, rpy[2] || 0, "XYZ"));
    m.setPosition(xyz[0] || 0, xyz[1] || 0, xyz[2] || 0);
    return m;
  }

  function meshFor(g) {
    const mat = new THREE.MeshLambertMaterial({ color: LINK_COLOR, transparent: true, opacity: 0.92 });
    let geo;
    if (g.shape === "box") geo = new THREE.BoxGeometry(g.size[0], g.size[1], g.size[2]);
    else if (g.shape === "cylinder") geo = new THREE.CylinderGeometry(g.radius, g.radius, g.length, 28);
    else if (g.shape === "sphere") geo = new THREE.SphereGeometry(g.radius, 24, 18);
    else return axes(0.06);
    const mesh = new THREE.Mesh(geo, mat);
    if (g.shape === "cylinder") mesh.rotation.x = Math.PI / 2; // URDF Z-axis → three Y-axis
    return mesh;
  }

  function axes(s) {
    const g = new THREE.Group();
    const seg = (col, x, y, z) => {
      const m = new THREE.LineBasicMaterial({ color: col });
      const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(x, y, z)]);
      g.add(new THREE.Line(geo, m));
    };
    seg(0xff6b6b, s, 0, 0); seg(0x6bff6b, 0, s, 0); seg(0x6b9bff, 0, 0, s);
    return g;
  }

  function fitCamera() {
    const box = new THREE.Box3().setFromObject(robotGroup);
    if (box.isEmpty()) { radius = 3; target.set(0, 0, 0); return; }
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    target.copy(box.getCenter(new THREE.Vector3()));
    radius = Math.max(0.3, size * 1.6);
  }

  function loop() {
    raf = requestAnimationFrame(loop);
    resize();
    const t = (performance.now() - t0) / 1000;
    if (autoRot) theta += 0.004;
    camera.position.set(
      target.x + radius * Math.sin(phi) * Math.cos(theta),
      target.y + radius * Math.cos(phi),
      target.z + radius * Math.sin(phi) * Math.sin(theta),
    );
    camera.lookAt(target);
    renderer.render(scene, camera);
    if (compass) { compass.resize(); compass.draw(t); }
    if (topo) { topo.resize(); topo.draw(t); }
  }

  function resize() {
    const r = robotCanvas.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
    const w = r.width || 1, h = r.height || 1;
    if (robotCanvas.width !== Math.floor(w * dpr) || robotCanvas.height !== Math.floor(h * dpr)) {
      renderer.setPixelRatio(dpr); renderer.setSize(w, h, false);
      camera.aspect = w / h; camera.updateProjectionMatrix();
    }
  }

  function disposeGroup(grp) {
    grp.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) { Array.isArray(o.material) ? o.material.forEach((m) => m.dispose()) : o.material.dispose(); }
    });
  }

  function bindPointer() {
    robotCanvas.addEventListener("pointerdown", (e) => { drag = true; moved = false; lx = e.clientX; ly = e.clientY; autoRot = false; });
    robotCanvas.addEventListener("wheel", (e) => { e.preventDefault(); radius = Math.max(0.2, radius + e.deltaY * 0.002 * radius); }, { passive: false });
    onUp = () => { drag = false; };
    onMove = (e) => {
      if (!drag) return;
      moved = true;
      theta += (e.clientX - lx) * 0.01;
      phi = Math.max(0.15, Math.min(3.0, phi - (e.clientY - ly) * 0.008));
      lx = e.clientX; ly = e.clientY;
    };
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointermove", onMove);
  }

  // ── Hover tooltip + click selection on robot links ────────────────
  function bindPicking() {
    robotCanvas.addEventListener("pointermove", (e) => {
      if (drag) { tooltip.style.display = "none"; return; }
      const hit = pick(e.clientX, e.clientY);
      if (!hit) { tooltip.style.display = "none"; return; }
      const info = infoFor(hit.object);
      tooltip.innerHTML = `<b>${esc(info.title)}</b>` + info.lines.map((l) => `<div>${esc(l)}</div>`).join("");
      const r = robotCanvas.getBoundingClientRect();
      tooltip.style.left = (e.clientX - r.left + 14) + "px";
      tooltip.style.top = (e.clientY - r.top + 12) + "px";
      tooltip.style.display = "block";
    });
    robotCanvas.addEventListener("pointerleave", () => { tooltip.style.display = "none"; });
    robotCanvas.addEventListener("click", (e) => {
      if (moved) return;
      selectMesh(pick(e.clientX, e.clientY)?.object || null);
    });
  }

  function pick(clientX, clientY) {
    if (!raycaster || !pickMeshes.length) return null;
    const r = robotCanvas.getBoundingClientRect();
    pointer.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    return raycaster.intersectObjects(pickMeshes, false)[0] || null;
  }

  function infoFor(obj) {
    const u = obj.userData, g = u.geometry;
    const geoLine = g
      ? (g.shape === "box" ? `box ${g.size.join(" × ")}`
        : g.shape === "cylinder" ? `cylinder r=${g.radius} l=${g.length}`
        : g.shape === "sphere" ? `sphere r=${g.radius}` : g.shape)
      : "no visual geometry";
    return { title: u.name, lines: ["link", geoLine] };
  }

  function selectMesh(obj) {
    if (selectedMesh && selectedMesh.material) selectedMesh.material.emissive?.setHex(0x000000);
    selectedMesh = obj;
    if (!obj) { selBox.style.display = "none"; return; }
    obj.material.emissive?.setHex(0x553311);
    const info = infoFor(obj);
    selBox.innerHTML = `<b>${esc(info.title)}</b>` + info.lines.map((l) => `<div>${esc(l)}</div>`).join("");
    selBox.style.display = "block";
  }

  function unmount() {
    if (raf) cancelAnimationFrame(raf); raf = 0;
    if (onUp) window.removeEventListener("pointerup", onUp);
    if (onMove) window.removeEventListener("pointermove", onMove);
    onUp = onMove = null;
    if (robotGroup) { if (scene) scene.remove(robotGroup); disposeGroup(robotGroup); robotGroup = null; }
    if (compass) compass.dispose();
    if (topo) topo.dispose();
    if (renderer) { renderer.forceContextLoss?.(); renderer.dispose(); }
    if (host) host.innerHTML = "";
    scene = null; renderer = null; compass = null; topo = null;
  }

  return { name: "Kinematic", mount, render, unmount };
}

function panel(title, sink) {
  const p = el("div", "kin-panel");
  const h = el("h4", "kin-h"); h.textContent = title;
  const c = el("canvas", "kin-mini"); sink(c);
  p.append(h, c); return p;
}
function el(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }
function btn(label, onclick) { const b = el("button", "view3d-btn"); b.textContent = label; b.onclick = onclick; return b; }
function esc(s) { return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
