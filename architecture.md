# Agent Architecture: The Neural-Symbolic Model (Phase 6)

This document describes the full architecture of the Generative Agents system as of Phase 6. It reflects
all implemented design decisions and serves as the reference for future development.

---

## Core Philosophy

### 1. The Agent is a Neural Network
The `AgentBrain` is a `dspy.Module` with the same structure as a neural network.

| Neural Network | Agent |
|---|---|
| Weights | DSPy prompts + few-shot examples (learnable) |
| Layers | Cognitive steps (Retrieval → Plan → Act) |
| Forward Pass | Stateless function, no side effects |
| Optimizer | DSPy `BootstrapFewShot` / `MIPRO` |

### 2. Functional Purity
$$\text{ActionSignal} = \text{Brain}(\text{Percept}, \text{State})$$

- **Input**: `Percept` (what the agent senses now) + `AgentState` (snapshot of memory & status).
- **Output**: `ActionSignal` (instructions to move, speak, write to memory — applied *after* the pass).
- **No side effects inside the brain.** Writing to memory happens in `Agent._apply_action_signal()`.

### 3. Separation of Concerns
| Responsibility | Component | Files |
|---|---|---|
| Logic & Reasoning | `AgentBrain` (DSPy) | `agents/brain.py` |
| State & Execution | `Agent` (Python class) | `agents/agent.py` |
| Long-Term Memory | `MemorySystem` + Qdrant | `agents/memory/` |
| World Simulation | `SimulationEngine` | `simulation/engine.py` |

---

## Execution Model

### Simulation Loop
```
SimulationEngine.step()
  1. Sequential perception  → _calculate_percepts() for all agents
  2. Concurrent cognition   → ThreadPoolExecutor: agent.run_step() per agent (LLM I/O in parallel)
  3. Sequential execution   → _update_map_events() (no race conditions)
  4. Record state           → _record_round_update() for API
```

All LLM calls happen inside Step 2. This shrinks wall-clock time from `O(N × LLM_latency)` to `O(LLM_latency)` for N agents.

### Agent Step
```
agent.run_step(percept, maze, agents, time)
  1. perceive(percept)                   → update spatial WorldMap
  2. construct AgentState snapshot
  3. brain.forward(percept, state,        → 1-2 LLM calls total (see below)
       retrieve_fn=memory.retrieve,
       expand_fn=memory.get_context,
       maze=maze)                         → maze passed for address resolution
  4. _apply_action_signal(signal)         → write memories, update schedule, set action
  5. Execution.run(address)               → resolve address, pathfind, step one tile
  6. [conditional] MemoryConsolidator     → System 2 reflection (poignancy-triggered)
```

---

## AgentBrain: 4-Layer Forward Pass

```python
class AgentBrain(dspy.Module):
    def forward(percept, state, retrieve_fn, expand_fn, maze=None) -> ActionSignal:
        # Layer 1: Perception (0 LLM calls)
        filtered_events = perception(percept, state)

        # Layer 2: Retrieval (1 LLM call via ContextualReranker)
        memories = retrieval(state, filtered_events, retrieve_fn, expand_fn)

        # Layer 3: Planning (1 LLM call on new-day only, else 0)
        plan_signal = planning(state)
        
        # Layer 3.5: Replanning (1 LLM call IF event poignancy >= 0.8, else 0)
        replan_signal = replanning(state, filtered_events)
        if replan_signal: plan_signal.updated_daily_schedule = replan_signal.updated_daily_schedule

        # Layer 4: Actor (1 LLM call via ReActActor)
        action_signal = actor(state, plan_signal, maze=maze)

        return merge(plan_signal, action_signal)
```

### LLM Call Budget
| Tick | Calls / Agent | 25 agents |
|---|---|---|
| New Day | **2** (Planner + Actor) | ~50 |
| Normal | **1** (Actor only, Reranker batched) | ~25 |
| Interrupt | **+1** (Replanning Evaluator) | |

---

## Layer Details

### Layer 1: `SensoryProcessingLayer` (Perception)
- **File**: `agents/layers/perception.py`
- **Calls**: 0 LLM calls (heuristic filter + poignancy threshold)
- Filters `Percept.events` → `List[PerceivedEvent]` based on relevance and retention.

### Layer 2: `AssociativeMemoryLayer` (Retrieval)
- **File**: `agents/layers/retrieval.py`
- **Calls**: 1 batched LLM call (`ContextualReranker`)
- Three-stage pipeline:
  1. **Qdrant ANN** — `retrieve_fn(query, 20)` → top-20 semantic candidates
  2. **Graph expansion** — `expand_fn(mem.id, depth=1)` → BFS over payload edges for each candidate
  3. **ContextualReranker** — 1 LLM call that jointly ranks all candidates by contextual importance

### Layer 3: `PlanningLayer` (Day Planning)
- **File**: `agents/layers/planning.py`
- **Calls**: 1 LLM call on `NEW_DAY` / `FIRST_DAY`, else 0
- Uses `UnifiedDayPlanner` → single `dspy.ChainOfThought(UnifiedDayPlanSignature)` call.
- **Output**: structured `UnifiedDayPlan` (wake_up_hour + plan_narrative + hourly_slots list).
- Replaces: `WakeUpHourPredictor + DailyPlanGenerator + HourlyScheduler` (was 3–5 calls).

### Layer 4: `ActorLayer` (Decision & Action)
- **File**: `agents/layers/actor.py`
- **Calls**: 1 LLM call (`ReActActor`) — skipped if current action is still running and no interrupt
- **Interrupt threshold**: If any perceived event has `poignancy >= 0.7` (`INTERRUPT_POIGNANCY_THRESHOLD`), the agent re-evaluates even if its current action hasn't finished. Configurable at module level.
- **Action duration**: `DEFAULT_ACTION_DURATION = 10` minutes (configurable). Agents cycle through actions every ~60 ticks at 10-second tick resolution.
- **Location resolution**: `_resolve_address(activity, maze)` fuzzy-matches natural-language activities (e.g. `"Morning routine"`) to real maze addresses (e.g. `"the Ville:Hobbs Cafe:cafe"`) using `maze.address_tiles`. Falls back to word matching, then random.
- `ReActActor` picks one of 4 tools; Python dispatches (zero extra LLM calls):
  - `move_to(destination)` → resolve address via maze, build `Action`
  - `speak_to(target, opening_line)` → create chat `Action` + memory event
  - `wait()` → wander toward current schedule location
  - `update_action(activity)` → update background activity description
- **Object Tracking**: Outputs an `ObjectAction` natively if the address resolves to `World:Sector:Arena:Object` syntax, registering state directly on the maze tile via `entity_id`.
- Replaces: `TalkDecider + ReactionDecider + SectorSelector + ArenaSelector + ObjectSelector + EmojiMapper` (was 6+ calls).

### System 2: `MemoryConsolidator` (Reflection, offline)
- **File**: `agents/layers/reflection.py`
- **Trigger**: `working_memory.reflection_trigger_counter <= 0` (poignancy-weighted)
- **Role**: Compresses memories → generates insights → updates identity.
- Runs asynchronously after the main brain pass inside `agent.run_step()`.

---

## Execution (Motor Control)

### `Execution` — Pathfinding & Movement
- **File**: `agents/components/execution.py`
- **Calls**: 0 LLM calls (pure Python)
- Receives the `Action.address` (a maze address resolved by `ActorLayer`) and converts it to tile-by-tile movement:
  1. Look up `address` in `maze.address_tiles` (with colon-trimming fallback)
  2. `maze.find_path(current_tile, target_tile)` → A* shortest path
  3. Pop one tile per tick from `planned_path`
- **Logging**: `log_agent` calls emit path calculation and per-tick movement (`Moving: (x,y) → (x,y) [N steps left]`)
- **Path caching**: `action_path_set` flag prevents redundant re-pathing while walking. Resets when `planned_path` is empty or action changes.

---

## Memory Architecture

### Design Principle
> Vectors AND graph edges live in the same Qdrant store. No external JSON file. No numpy.

### `QdrantMemoryRepository`
- **File**: `agents/memory/repository.py`
- **Backend**: `QdrantClient(location=":memory:")` (tests) or `QdrantClient(path=...)` (persistent).
- **Payload schema per point**:
  ```python
  {
    "str_id":         str,   # original UUID
    "content":        str,   # memory text
    "memory_type":    str,   # "observation" | "chat" | "event" | "thought"
    "depth":          int,   # 0=raw, 1=reflection, 2=insight
    "created_ts":     float, # Unix timestamp
    "importance":     float,
    "entity_id":      str,   # Originating agent or object identifier
    "related_events": list,  # [{id: str, relation: str}] — graph edges
  }
  ```
- **Important API notes** (qdrant-client embedded, version-specific):
  - Use `QdrantClient(location=":memory:")` or `QdrantClient(path="/dir")` — not `QdrantClient(":memory:")`.
  - Use `.query_points()` — `.search()` is gRPC/server-mode only.
  - Use `collection_exists()` + `create_collection()` — `recreate_collection()` removed.
  - Local path uses file locking: always `client.close()` before re-opening the same path.

### `MemorySystem`
- **File**: `agents/memory/system.py`
- Thin facade over the database layer:
  - `add(event)` → embed + upsert to Qdrant
  - `retrieve(query, limit)` → ANN search
  - `get_context(node_id, depth)` → BFS over `related_events` payload edges

### `WorkingMemory`
- **File**: `agents/memory/working.py`
- Volatile short-term state: current action, daily schedule, chatting state, reflection counter.

### `WorldMap` (Spatial)
- **File**: `agents/memory/spatial.py`
- Internal tile map built from perceived `Percept.nearby_tiles`.

### Graph Traversal (end-to-end)
```
agent.run_step()
  └─► brain.forward(expand_fn=memory.get_context)
        └─► AssociativeMemoryLayer
              1. retrieve_fn(query, 20)         → Qdrant ANN top-20
              2. expand_fn(mem.id, depth=1)     → BFS over payload related_events
                    MemorySystem.get_context()
                    repo.get_related(id)        → reads payload field
              3. ContextualReranker(candidates) → 1 LLM call, top-10 ranked
```

---

## Intelligence Modules

All LLM logic lives in `intelligence/modules/`. Every `dspy.Module` is end-to-end optimizable.

### Active Modules

| File | Module | Calls | Role |
|---|---|---|---|
| `planning.py` | `UnifiedDayPlanner` | 1 (new-day) | Single-call day plan (wake + schedule) |
| `actor_react.py` | `ReActActor` | 1/tick | Tool-choice actor: move / speak / wait / update |
| `retrieval.py` | `ContextualReranker` | 1 batched | Contextual importance re-ranking |
| `dialogue.py` | `DialogueGenerator`, `DialogueSummarizer`, `DialogueMemoer`, `DialoguePlanner` | 1 each | Multi-turn chat _(not yet wired in main brain)_ |
| `dialogue.py` | `RelationshipSummarizer` | 1 | Reflection: relationship summary |
| `reflection.py` | `ReflectionPointGenerator`, `InsightGenerator`, `IdentityFormulator` | 1 each | System 2: memory compression + identity |

### Replaced with Heuristics (0 LLM calls)

| Removed | Replacement | File |
|---|---|---|
| `PoignanceRater` | `heuristic_poignance(event_type, description)` — keyword+type scoring | `perception.py` |
| `EmojiMapper` | `heuristic_emoji(action_description)` — keyword lookup table | `perception.py` |

### Removed (absorbed into ReActActor tools)

| Removed | Reason |
|---|---|
| `TalkDecider` | 1 call/tick | 0 (ReActActor) |
| `ReactionDecider` | 1 call/tick | 0 (ReActActor) |
| `SectorSelector+ArenaSelector+ObjectSelector` | 3 calls/move | 0 (ReActActor) |
| `EventParser` | 1 call/action | 0 (Removed S/P/O triples) |

---

## DSPy Optimization Path

Since every cognitive step is a `dspy.Module`, the entire brain can be end-to-end optimized against a simulation quality metric:

```python
from dspy.teleprompt import BootstrapFewShot
optimizer = BootstrapFewShot(metric=simulation_quality_metric)
optimized_brain = optimizer.compile(AgentBrain(), trainset=simulation_traces)
```

Optimizable parameters: contextual importance weights (`ContextualReranker`), tool selection policy (`ReActActor`), plan generation quality (`UnifiedDayPlanner`).

---

## Key Files Reference

```
src/generative_agents/
  agents/
    agent.py                     ← run_step, expand_fn + maze wiring
    brain.py                     ← AgentBrain (4-layer forward pass, maze passthrough)
    components/
      execution.py               ← Execution (pathfinding + movement logging)
    layers/
      perception.py              ← SensoryProcessingLayer
      retrieval.py               ← AssociativeMemoryLayer (3-stage)
      planning.py                ← PlanningLayer (UnifiedDayPlanner)
      actor.py                   ← ActorLayer (ReActActor + address resolution + interrupt)
      reflection.py              ← MemoryConsolidator (System 2)
    memory/
      repository.py              ← QdrantMemoryRepository (pure Qdrant)
      system.py                  ← MemorySystem (add/retrieve/get_context)
      working.py                 ← WorkingMemory (short-term volatile)
      spatial.py                 ← WorldMap (tile map)
  intelligence/
    modules/
      planning.py                ← UnifiedDayPlanner + legacy modules
      actor_react.py             ← ReActActor + tool definitions
      retrieval.py               ← ContextualReranker
      perception.py              ← PoignanceRater, EventParser, ...
      dialogue.py                ← TalkDecider, DialogueGenerator, ...
      reflection.py              ← ReflectionPointGenerator, InsightGenerator
      # spatial.py DELETED — SectorSelector/ArenaSelector/ObjectSelector replaced by ReActActor move_to tool
  persistence/
    database.py                  ← Thin DB façade (per-agent repo registry)
  simulation/
    engine.py                    ← SimulationEngine (concurrent agent execution)
    maze.py                      ← Tile map + A* pathfinding
  common/
    neural_types.py              ← AgentState, ActionSignal (dataclasses)
    events.py                    ← PerceivedEvent, Event, Action
    percept.py                   ← Percept dataclass
```
