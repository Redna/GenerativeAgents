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
       expand_fn=memory.get_context)
  4. _apply_action_signal(signal)        → write memories, update schedule, set action
  5. Execution.run(address)              → pathfind to next tile
  6. [conditional] MemoryConsolidator   → System 2 reflection (every N ticks)
```

---

## AgentBrain: 4-Layer Forward Pass

```python
class AgentBrain(dspy.Module):
    def forward(percept, state, retrieve_fn, expand_fn) -> ActionSignal:
        # Layer 1: Perception (0 LLM calls)
        filtered_events = perception(percept, state)

        # Layer 2: Retrieval (1 LLM call via ContextualReranker)
        memories = retrieval(state, filtered_events, retrieve_fn, expand_fn)

        # Layer 3: Planning (1 LLM call on new-day only, else 0)
        plan_signal = planning(state)

        # Layer 4: Actor (1 LLM call via ReActActor)
        action_signal = actor(state, plan_signal)

        return merge(plan_signal, action_signal)
```

### LLM Call Budget
| Tick | Calls / Agent | 25 agents |
|---|---|---|
| New Day | **2** (Planner + Actor) | ~50 |
| Normal | **1** (Actor only, Reranker batched) | ~25 |

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
- **Calls**: 1 LLM call (`ReActActor`)
- `ReActActor` picks one of 4 tools; Python dispatches (zero extra LLM calls):
  - `move_to(destination)` → build `Action` with address
  - `speak_to(target, opening_line)` → create chat `Action` + memory event
  - `wait()` → no-op, current action continues
  - `update_action(activity)` → update background activity description
- Replaces: `TalkDecider + ReactionDecider + SectorSelector + ArenaSelector + ObjectSelector + EventParser + EmojiMapper` (was 7+ calls).

### System 2: `MemoryConsolidator` (Reflection, offline)
- **File**: `agents/layers/reflection.py`
- **Trigger**: `working_memory.reflection_trigger_counter <= 0` (poignancy-weighted)
- **Role**: Compresses memories → generates insights → updates identity.
- Runs asynchronously after the main brain pass inside `agent.run_step()`.

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
    "subject":        str,
    "predicate":      str,
    "object_":        str,
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
| `perception.py` | `EventParser` | 1/dispatch | Triple extraction for move/update actions |
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
| `TalkDecider` | ReActActor `speak_to(target, opening_line)` |
| `ReactionDecider` | ReActActor `wait` / `speak_to` |
| `Contextualizer` | No callers — deleted |
| `spatial.py` (entire file) | `SectorSelector` + `ArenaSelector` + `ObjectSelector` → ReActActor `move_to(destination)` |

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
    agent.py                     ← run_step, expand_fn wiring
    brain.py                     ← AgentBrain (4-layer forward pass)
    layers/
      perception.py              ← SensoryProcessingLayer
      retrieval.py               ← AssociativeMemoryLayer (3-stage)
      planning.py                ← PlanningLayer (UnifiedDayPlanner)
      actor.py                   ← ActorLayer (ReActActor + dispatch)
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
