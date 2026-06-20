import init, { adversarial_world_json, adversarial_world_seed_json, generated_random_world_json, generated_world_json, randomized_world_json, solved_grid_game_json, solved_world_json, virus_world_json, virus_world_seed_json } from "../editor/pkg/hymeko_wasm.js";

const board = document.querySelector("#board");
const statusEl = document.querySelector("#status");
const gateEl = document.querySelector("#gate");
const statesEl = document.querySelector("#states");
const edgesEl = document.querySelector("#edges");
const simStateEl = document.querySelector("#simState");
const simProgressEl = document.querySelector("#simProgress");
const evolutionPlot = document.querySelector("#evolutionPlot");
const worldSelect = document.querySelector("#worldSelect");
const customTopology = document.querySelector("#customTopology");
const worldWidth = document.querySelector("#worldWidth");
const worldHeight = document.querySelector("#worldHeight");
const btnGenerateWorld = document.querySelector("#btnGenerateWorld");
const btnRandomizeWorld = document.querySelector("#btnRandomizeWorld");
const btnReset = document.querySelector("#btnReset");
const btnStep = document.querySelector("#btnStep");
const btnPlay = document.querySelector("#btnPlay");
const policyEl = document.querySelector("#policy");
const evolutionEl = document.querySelector("#evolution");
const evolutionSlider = document.querySelector("#evolutionSlider");
const evolutionLabel = document.querySelector("#evolutionLabel");
const evolutionPolicyEl = document.querySelector("#evolutionPolicy");
const evolutionSourceEl = document.querySelector("#evolutionSource");
const mdpEl = document.querySelector("#mdp");
const sourceEl = document.querySelector("#source");
const graphEl = document.querySelector("#graph");

let currentGame = null;
let traceIndex = 0;
let playTimer = null;
let evolutionStageIndex = 0;
let lastSeed = 1;

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".pane").forEach((x) => x.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`#${button.dataset.tab}`).classList.add("active");
  });
});

try {
  await init();
  await loadWorld(worldSelect?.value ?? "grid4");
} catch (err) {
  statusEl.textContent = String(err);
}

worldSelect?.addEventListener("change", async () => {
  stopPlay();
  statusEl.textContent = "Solving selected world...";
  if (worldSelect.value === "custom") {
    await loadGeneratedWorld();
  } else {
    await loadWorld(worldSelect.value);
  }
});

async function loadWorld(worldId, seed = null) {
  try {
    const json = worldId === "adversarial" && seed != null && typeof adversarial_world_seed_json === "function"
      ? adversarial_world_seed_json(seed)
      : worldId === "adversarial" && typeof adversarial_world_json === "function"
      ? adversarial_world_json()
      : worldId === "virus" && seed != null && typeof virus_world_seed_json === "function"
      ? virus_world_seed_json(seed)
      : worldId === "virus" && typeof virus_world_json === "function"
      ? virus_world_json()
      : seed != null && typeof randomized_world_json === "function"
      ? randomized_world_json(worldId, seed)
      : typeof solved_world_json === "function"
      ? solved_world_json(worldId)
      : solved_grid_game_json();
    const game = normalizeGame(JSON.parse(json));
    render(game);
  } catch (err) {
    statusEl.textContent = String(err);
  }
}

btnGenerateWorld?.addEventListener("click", async () => {
  stopPlay();
  if (worldSelect) worldSelect.value = "custom";
  statusEl.textContent = "Generating custom world...";
  await loadGeneratedWorld();
});

btnRandomizeWorld?.addEventListener("click", async () => {
  stopPlay();
  lastSeed = Math.floor(Math.random() * 1_000_000_000);
  statusEl.textContent = `Randomizing starts with seed ${lastSeed}...`;
  if (worldSelect?.value === "custom") {
    await loadGeneratedWorld(lastSeed);
  } else {
    await loadWorld(worldSelect?.value ?? "grid4", lastSeed);
  }
});

customTopology?.addEventListener("change", () => {
  const isHex = customTopology.value === "hex";
  worldWidth.min = isHex ? "1" : "2";
  worldWidth.max = isHex ? "6" : "16";
  worldHeight.min = isHex ? "1" : "2";
  worldHeight.max = isHex ? "6" : "16";
  if (isHex) {
    worldWidth.value = String(clampNumber(worldWidth.value, 1, 6));
    worldHeight.value = String(clampNumber(worldHeight.value, 1, 6));
  }
});

async function loadGeneratedWorld(seed = null) {
  try {
    const topology = customTopology?.value ?? "square";
    const max = topology === "hex" ? 6 : 16;
    const min = topology === "hex" ? 1 : 2;
    const width = clampNumber(worldWidth?.value, min, max);
    const height = clampNumber(worldHeight?.value, min, max);
    if (worldWidth) worldWidth.value = String(width);
    if (worldHeight) worldHeight.value = String(height);
    const json = seed != null && typeof generated_random_world_json === "function"
      ? generated_random_world_json(topology, width, height, seed)
      : typeof generated_world_json === "function"
      ? generated_world_json(topology, width, height)
      : solved_grid_game_json();
    const game = normalizeGame(JSON.parse(json));
    render(game);
  } catch (err) {
    statusEl.textContent = String(err);
  }
}

function clampNumber(value, min, max) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return min;
  return Math.max(min, Math.min(max, parsed));
}

function normalizeGame(game) {
  game.world_id ??= "grid4";
  game.world_label ??= "4x4 square grid";
  game.topology ??= "square";
  game.start_state ??= "c0_0";
  game.concepts ??= defaultConcepts();
  game.counts.concept_edges ??= 0;
  game.counts.action_edges ??= game.counts.transition_edges;
  game.trace ??= deriveTrace(game);
  game.evolution ??= deriveEvolution(game);
  game.counts.simulation_steps ??= game.trace.length;
  return game;
}

function defaultConcepts() {
  return [
    { id: "emergent_agent", label: "agent", role: "entity whose behavior emerges from state, observation, goals, and actions" },
    { id: "agent_state", label: "state hypernode", role: "mutable internal state bound to world cells and observations" },
    { id: "observation_space", label: "observation space", role: "observations are linked to agent state, not stored as isolated facts" },
    { id: "agent_goal", label: "goal subnode", role: "sub-goals that bias the policy: reach, avoid, and minimize cost" },
    { id: "action_hyperedge", label: "action hyperedge", role: "available control is an edge over agent, state, observation, goal, and next state" },
    { id: "reward_signal", label: "reward", role: "scalar feedback attached to action/transition hyperedges" },
    { id: "value_field", label: "value", role: "solved discounted return over state observations" },
    { id: "policy_selector", label: "policy", role: "agent-owned selector that contains and chooses action hyperedges" },
    { id: "agent_trace", label: "trace", role: "simulated rollout of the agent following selected action hyperedges" },
    { id: "mdp_observation", label: "observation", role: "local percept node produced from the agent-state/world-state binding" },
  ];
}

function deriveTrace(game) {
  const byState = new Map(game.policy_edges.map((edge) => [edge.from, edge]));
  const byCell = new Map(game.cells.map((cell) => [cell.id, cell]));
  const trace = [];
  let state = game.start_state ?? "c0_0";
  let cumulativeReward = 0;
  for (let step = 0; step < 32; step += 1) {
    const edge = byState.get(state);
    if (!edge) break;
    const reward = Number(edge.reward);
    cumulativeReward += reward;
    const cell = byCell.get(state);
    const next = byCell.get(edge.to);
    trace.push({
      step,
      state,
      observation: `o_${state}`,
      action: edge.action,
      next_state: edge.to,
      reward,
      cumulative_reward: cumulativeReward,
      value: cell?.value ?? 0,
      terminal: next?.tile === "goal" || next?.tile === "pit",
    });
    state = edge.to;
    if (trace.at(-1).terminal) break;
  }
  return trace;
}

function deriveEvolution(game) {
  return [
    {
      iteration: 0,
      label: "initial agent scaffold",
      changed_states: 0,
      cells: game.cells.map((cell) => ({ ...cell, value: 0, policy: null })),
      policy_edges: [],
      hymeko_source: `// Initial HyMeKo RL scaffold\nEmergentGridAgent {\n  agent_hikari {}\n  hikari_state { gamma 0.95; }\n  hikari_policy { /* actions appear here during learning */ }\n}`,
    },
    {
      iteration: game.iterations,
      label: "converged policy",
      changed_states: game.counts.playable_states,
      cells: game.cells,
      policy_edges: game.policy_edges,
      hymeko_source: game.hymeko_source,
    },
  ];
}

btnReset.addEventListener("click", () => {
  stopPlay();
  traceIndex = 0;
  applyTrace();
});

evolutionSlider.addEventListener("input", () => {
  stopPlay();
  updateEvolution(Number(evolutionSlider.value));
});

btnStep.addEventListener("click", () => {
  stopPlay();
  stepTrace();
});

btnPlay.addEventListener("click", () => {
  if (playTimer) {
    stopPlay();
    return;
  }
  btnPlay.textContent = "Pause";
  playTimer = window.setInterval(() => {
    if (!stepTrace()) stopPlay();
  }, 700);
});

function render(game) {
  currentGame = game;
  traceIndex = 0;
  statusEl.textContent = `${game.world_label}: ${game.iterations} sweeps, gamma=${game.gamma}`;
  gateEl.textContent = game.gate.accepted ? "ok" : "no";
  statesEl.textContent = game.counts.playable_states;
  edgesEl.textContent = game.counts.policy_edges;

  board.innerHTML = "";
  board.classList.toggle("hex-board", game.topology === "hex");
  board.classList.toggle("square-board", game.topology !== "hex");
  board.style.removeProperty("--cols");
  board.style.removeProperty("--rows");
  board.style.removeProperty("--hex-w");
  board.style.removeProperty("--hex-h");
  board.setAttribute("aria-label", `${game.world_label} solved reinforcement learning board`);

  const hexLayout = game.topology === "hex" ? computeHexLayout(game.cells) : null;
  if (game.topology !== "hex") {
    board.style.setProperty("--cols", String(game.width ?? 4));
    board.style.setProperty("--rows", String(game.height ?? 4));
  } else if (hexLayout) {
    board.style.setProperty("--hex-w", `${hexLayout.cellW}%`);
    board.style.setProperty("--hex-h", `${hexLayout.cellH}%`);
  }

  for (const cell of game.cells) {
    const tile = document.createElement("article");
    tile.className = `cell ${cell.tile}`;
    tile.dataset.cellId = cell.id;
    if (game.topology === "hex" && hexLayout) {
      const pos = hexLayout.positions.get(cell.id);
      tile.style.left = `${pos.left}%`;
      tile.style.top = `${pos.top}%`;
    } else {
      tile.style.gridColumn = `${cell.x + 1}`;
      tile.style.gridRow = `${cell.y + 1}`;
    }

    const coord = document.createElement("div");
    coord.className = "coord";
    coord.textContent = cell.id;

    const mark = document.createElement("div");
    mark.className = "mark";
    mark.textContent = glyphFor(cell);

    const value = document.createElement("div");
    value.className = "value";
    value.textContent = cell.tile === "wall" ? "" : cell.value.toFixed(2);

    tile.append(coord, mark, value);
    board.append(tile);
  }

  policyEl.textContent = renderPolicy(game);
  renderPlot(game);
  renderEvolution(game);
  mdpEl.innerHTML = renderMdp(game);
  sourceEl.textContent = game.hymeko_source;
  graphEl.textContent = game.mermaid;
  simProgressEl.max = Math.max(1, game.trace.length);
  applyTrace();
}

function computeHexLayout(cells) {
  const raw = cells.map((cell) => {
    const q = Number(cell.q ?? cell.x ?? 0);
    const r = Number(cell.r ?? cell.y ?? 0);
    return {
      id: cell.id,
      px: Math.sqrt(3) * (q + r / 2),
      py: 1.5 * r,
    };
  });
  const minX = Math.min(...raw.map((p) => p.px));
  const maxX = Math.max(...raw.map((p) => p.px));
  const minY = Math.min(...raw.map((p) => p.py));
  const maxY = Math.max(...raw.map((p) => p.py));
  const spanX = Math.max(1, maxX - minX);
  const spanY = Math.max(1, maxY - minY);
  const cellW = Math.min(15, 72 / Math.max(1, spanX + 1.8));
  const cellH = cellW * 0.92;
  const positions = new Map();
  for (const point of raw) {
    positions.set(point.id, {
      left: 6 + ((point.px - minX) / spanX) * 88,
      top: 6 + ((point.py - minY) / spanY) * 88,
    });
  }
  return { positions, cellW, cellH };
}

function glyphFor(cell) {
  if (cell.tile === "wall") return "#";
  if (cell.tile === "goal") return "G";
  if (cell.tile === "pit") return "X";
  if (cell.tile === "rival") return "K";
  if (cell.tile === "automaton") return "CA";
  if (cell.tile === "virus") return "V";
  return cell.policy?.glyph ?? ".";
}

function renderPolicy(game) {
  const lines = [
    `gate: nodes=${game.gate.node_count}, edges=${game.gate.edge_count}, arcs=${game.gate.arc_count}`,
    `available action hyperedges: ${game.counts.action_edges}`,
    `simulation trace steps: ${game.counts.simulation_steps}`,
    "",
  ];
  for (const edge of game.policy_edges) {
    const reward = edge.reward >= 0 ? `+${edge.reward.toFixed(2)}` : edge.reward.toFixed(2);
    lines.push(`${edge.from.padEnd(4)} --${edge.action.padEnd(5)} r=${reward}--> ${edge.to}`);
  }
  return lines.join("\n");
}

function renderEvolution(game) {
  evolutionSlider.max = Math.max(0, game.evolution.length - 1);
  evolutionSlider.value = String(game.evolution.length - 1);
  updateEvolution(game.evolution.length - 1);
}

function updateEvolution(index) {
  if (!currentGame?.evolution?.length) return;
  evolutionStageIndex = Math.max(0, Math.min(index, currentGame.evolution.length - 1));
  const stage = currentGame.evolution[evolutionStageIndex];
  evolutionLabel.textContent = `${stage.label} | sweep ${stage.iteration} | changed states ${stage.changed_states}`;
  evolutionPolicyEl.textContent = renderEvolutionPolicy(stage);
  evolutionSourceEl.textContent = stage.hymeko_source;

  const byCell = new Map(stage.cells.map((cell) => [cell.id, cell]));
  document.querySelectorAll(".cell").forEach((tile) => {
    const cell = byCell.get(tile.dataset.cellId);
    if (!cell) return;
    tile.classList.remove("active", "next", "visited", "ego-live", "rival-live", "automaton-live", "virus-live", "cleanse-live");
    tile.querySelector(".mark").textContent = glyphFor(cell);
    tile.querySelector(".value").textContent = cell.tile === "wall" ? "" : cell.value.toFixed(2);
    tile.classList.toggle("unlearned", !cell.policy && !["wall", "goal", "pit", "rival", "automaton", "virus"].includes(cell.tile));
  });
  applyExplorationPaths(evolutionStageIndex);
  updatePlotMarker();
}

function applyExplorationPaths(stageIndex) {
  document.querySelectorAll(".cell").forEach((tile) => {
    tile.classList.remove("path-learned", "path-current", "path-start", "path-terminal");
    tile.style.removeProperty("--path-color");
    tile.style.removeProperty("--path-glow");
  });
  if (!currentGame || currentGame.trace?.some((step) => step.rival_state)) return;

  for (let i = 1; i <= stageIndex; i += 1) {
    const stage = currentGame.evolution[i];
    const trace = deriveStagePath(currentGame, stage);
    const isCurrent = i === stageIndex;
    trace.forEach((stateId, order) => {
      const tile = document.querySelector(`[data-cell-id="${stateId}"]`);
      if (!tile) return;
      const hue = (205 + i * 43 + order * 18) % 360;
      tile.classList.add("path-learned");
      tile.classList.toggle("path-current", isCurrent);
      tile.classList.toggle("path-start", order === 0 && isCurrent);
      tile.classList.toggle("path-terminal", order === trace.length - 1 && isCurrent);
      tile.style.setProperty("--path-color", `hsl(${hue} 78% 58%)`);
      tile.style.setProperty("--path-glow", `hsla(${hue}, 78%, 58%, ${isCurrent ? 0.38 : 0.20})`);
    });
  }
}

function deriveStagePath(game, stage) {
  if (!stage?.policy_edges?.length) return [];
  const byState = new Map(stage.policy_edges.map((edge) => [edge.from, edge]));
  const byCell = new Map(stage.cells.map((cell) => [cell.id, cell]));
  const out = [];
  const seen = new Set();
  let state = game.start_state ?? "c0_0";
  const maxSteps = Math.max(8, game.cells.length * 2);

  for (let i = 0; i < maxSteps; i += 1) {
    out.push(state);
    const cell = byCell.get(state);
    if (cell?.tile === "goal" || cell?.tile === "pit") break;
    const edge = byState.get(state);
    if (!edge) break;
    const guard = `${edge.from}:${edge.action}:${edge.to}`;
    if (seen.has(guard)) break;
    seen.add(guard);
    state = edge.to;
  }
  return out;
}

function renderEvolutionPolicy(stage) {
  const lines = [
    `stage: ${stage.label}`,
    `sweep: ${stage.iteration}`,
    `changed states since previous stage: ${stage.changed_states}`,
    "",
  ];
  if (!stage.policy_edges.length) {
    lines.push("no policy-owned action hyperedges yet");
    return lines.join("\n");
  }
  for (const edge of stage.policy_edges) {
    const reward = edge.reward >= 0 ? `+${edge.reward.toFixed(2)}` : edge.reward.toFixed(2);
    lines.push(`${edge.from.padEnd(4)} selects ${edge.action.padEnd(5)} -> ${edge.to}  r=${reward}`);
  }
  return lines.join("\n");
}

function renderPlot(game) {
  const stages = game.evolution ?? [];
  if (!stages.length) {
    evolutionPlot.innerHTML = "";
    return;
  }

  const w = 220;
  const h = 360;
  const pad = { left: 38, right: 14, top: 28, bottom: 34 };
  const innerW = w - pad.left - pad.right;
  const innerH = h - pad.top - pad.bottom;
  const maxChanged = Math.max(1, ...stages.map((stage) => stage.changed_states ?? 0));
  const avgValues = stages.map(avgStageValue);
  const minValue = Math.min(...avgValues, 0);
  const maxValue = Math.max(...avgValues, 1);
  const span = Math.max(1e-6, maxValue - minValue);
  const xOf = (i) => pad.left + (stages.length === 1 ? innerW / 2 : (i / (stages.length - 1)) * innerW);
  const yValue = (value) => pad.top + innerH - ((value - minValue) / span) * innerH;
  const yChanged = (value) => pad.top + innerH - (value / maxChanged) * innerH;

  const line = avgValues.map((value, i) => `${xOf(i).toFixed(1)},${yValue(value).toFixed(1)}`).join(" ");
  const bars = stages.map((stage, i) => {
    const x = xOf(i) - 7;
    const y = yChanged(stage.changed_states ?? 0);
    const bh = pad.top + innerH - y;
    return `<rect class="plot-bar" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="14" height="${bh.toFixed(1)}" rx="3">
      <title>${escapeHtml(stage.label)}: ${stage.changed_states} changed states</title>
    </rect>`;
  }).join("");
  const dots = stages.map((stage, i) => {
    const value = avgValues[i];
    return `<circle class="plot-dot" cx="${xOf(i).toFixed(1)}" cy="${yValue(value).toFixed(1)}" r="4">
      <title>${escapeHtml(stage.label)}: avg value ${value.toFixed(2)}</title>
    </circle>`;
  }).join("");
  const ticks = stages.map((stage, i) => `<text class="plot-tick" x="${xOf(i).toFixed(1)}" y="${h - 11}" text-anchor="middle">${stage.iteration}</text>`).join("");

  evolutionPlot.innerHTML = `
    <line class="plot-axis" x1="${pad.left}" y1="${pad.top + innerH}" x2="${w - pad.right}" y2="${pad.top + innerH}"></line>
    <line class="plot-axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + innerH}"></line>
    <text class="plot-label" x="${pad.left}" y="13">avg value</text>
    <text class="plot-label" x="${w - pad.right}" y="13" text-anchor="end">changed states</text>
    ${bars}
    <polyline class="plot-line" points="${line}"></polyline>
    ${dots}
    ${ticks}
    <text class="plot-label" x="${pad.left + innerW / 2}" y="${h - 2}" text-anchor="middle">learning sweep</text>
    <line id="plotMarker" class="plot-marker" x1="${xOf(evolutionStageIndex)}" y1="${pad.top}" x2="${xOf(evolutionStageIndex)}" y2="${pad.top + innerH}"></line>
  `;
  updatePlotMarker();
}

function updatePlotMarker() {
  const marker = document.querySelector("#plotMarker");
  if (!marker || !currentGame?.evolution?.length) return;
  const stages = currentGame.evolution;
  const padLeft = 38;
  const innerW = 220 - 38 - 14;
  const x = padLeft + (stages.length === 1 ? innerW / 2 : (evolutionStageIndex / (stages.length - 1)) * innerW);
  marker.setAttribute("x1", x.toFixed(1));
  marker.setAttribute("x2", x.toFixed(1));
}

function avgStageValue(stage) {
  const cells = stage.cells.filter((cell) => cell.tile !== "wall");
  if (!cells.length) return 0;
  return cells.reduce((sum, cell) => sum + Number(cell.value ?? 0), 0) / cells.length;
}

function stepTrace() {
  if (!currentGame) return false;
  if (traceIndex >= currentGame.trace.length - 1) {
    applyTrace();
    return false;
  }
  traceIndex += 1;
  applyTrace();
  return true;
}

function stopPlay() {
  if (playTimer) {
    window.clearInterval(playTimer);
    playTimer = null;
  }
  btnPlay.textContent = "Play";
}

function applyTrace() {
  if (!currentGame) return;
  const trace = currentGame.trace;
  if (!trace.length) {
    simStateEl.textContent = "no trace";
    simProgressEl.value = 0;
    return;
  }
  document.querySelectorAll(".cell").forEach((cell) => {
    cell.classList.remove("active", "next", "visited", "ego-live", "rival-live", "automaton-live", "virus-live", "cleanse-live");
  });
  for (let i = 0; i < Math.min(traceIndex, trace.length); i += 1) {
    const visited = document.querySelector(`[data-cell-id="${trace[i].state}"]`);
    if (visited) visited.classList.add("visited");
  }

  const step = trace[Math.min(traceIndex, trace.length - 1)];
  const active = document.querySelector(`[data-cell-id="${step.state}"]`);
  const next = document.querySelector(`[data-cell-id="${step.next_state}"]`);
  if (active) active.classList.add("active", "ego-live");
  if (next) next.classList.add("next");
  if (step.rival_state) {
    const rival = document.querySelector(`[data-cell-id="${step.rival_state}"]`);
    if (rival) rival.classList.add("rival-live");
  }
  for (const id of step.automaton_cells ?? []) {
    const automaton = document.querySelector(`[data-cell-id="${id}"]`);
    if (automaton) automaton.classList.add("automaton-live");
  }
  for (const id of step.virus_cells ?? []) {
    const virus = document.querySelector(`[data-cell-id="${id}"]`);
    if (virus) virus.classList.add("virus-live");
  }
  for (const id of step.cleansed_cells ?? []) {
    const cleansed = document.querySelector(`[data-cell-id="${id}"]`);
    if (cleansed) cleansed.classList.add("cleanse-live");
  }

  simProgressEl.value = traceIndex + 1;
  const reward = step.reward >= 0 ? `+${step.reward.toFixed(2)}` : step.reward.toFixed(2);
  const rivalry = step.rival_state ? ` | Kage ${step.rival_action ?? "hold"} -> ${step.rival_state}, CA g${step.automaton_generation ?? 0}` : "";
  const virus = step.virus_cells ? ` | virus ${step.infection_count ?? step.virus_cells.length}, cleansed ${(step.cleansed_cells ?? []).length}${step.victory ? ", victory" : ""}` : "";
  simStateEl.textContent = `step ${step.step + 1}/${trace.length}: ${step.state} observes ${step.observation}, ${step.action} -> ${step.next_state}, r=${reward}, total=${step.cumulative_reward.toFixed(2)}${rivalry}${virus}`;
}

function renderMdp(game) {
  const concepts = game.concepts
    .map((concept) => `
      <article class="concept">
        <b>${escapeHtml(concept.label)}</b>
        <code>${escapeHtml(concept.id)}</code>
        <span>${escapeHtml(concept.role)}</span>
      </article>
    `)
    .join("");

  const sourceLines = [
    `agent_embodiment: agent_hikari, emergent_agent, hikari_state, agent_state, observation_space`,
    `agent_policy_binding: agent_hikari, hikari_state, hikari_policy`,
    `goal_stack: agent_hikari, goal_reach_exit, goal_avoid_pit, goal_minimize_cost, agent_goal`,
    `schema_action: emergent_agent, agent_state, mdp_observation, action_hyperedge, agent_goal, reward_signal`,
    `schema_policy: agent_state, mdp_observation, policy_selector, action_hyperedge, value_field`,
    `schema_trace: agent_trace, emergent_agent, agent_state, mdp_observation, action_hyperedge, reward_signal`,
    `concept annotation edges: ${game.counts.concept_edges}`,
  ];

  return `<div class="concept-grid">${concepts}</div><pre class="schema-lines">${sourceLines.join("\n")}</pre>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
