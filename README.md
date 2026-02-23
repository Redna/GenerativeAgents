# Generative Agents: The Neural-Symbolic Model

A modernized simulation environment demonstrating autonomous epistemic agents. This project explores the intersection of neural cognition (Large Language Models via DSPy) and symbolic execution (classic game loops, state machines, and discrete spatial pathfinding) to create a high-performance, responsive virtual world.

> **Attribution & Acknowledgments**  
> This project builds upon the foundational work presented in the Stanford/Google paper *Generative Agents: Interactive Simulacra of Human Behavior* by Joon Sung Park et al. We acknowledge the original open-source release available at [joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents). This codebase is distributed under the **Apache License 2.0**.

---

## 🧠 Architectural Overview

The core philosophy of this environment is **Functional Purity in Cognition**. It addresses latency and determinism challenges inherent in multi-agent architectures through several key design principles:

### 1. The Agent is a Neural Network
The agent's "Brain" is modeled as a declarative `dspy.Module`. It receives contextual inputs from what it physically perceives (`Percept`) and an internal snapshot of its memory (`AgentState`), passes them through cognitive layers (Retrieval → Planning → Action), and emits a pure `ActionSignal` free of mid-pass side effects.

### 2. High-Performance Execution Loop
Instead of freezing the environment while agents compute their next actions, the `SimulationEngine.step()` separates the agent lifecycle:
1. **Sequential Perception**: The physical state is snapshotted.
2. **Concurrent Cognition**: All agents execute their LLM chains in parallel using a `ThreadPoolExecutor`.
3. **Sequential Execution**: Agents take discrete spatial steps and write to the global state safely.

### 3. Unified Qdrant Memory Store
The memory architecture unifies **Vectors** and **Graph Edges** inside a single Qdrant vector database instance.
- **Semantic Search** fetches relevant memory embeddings.
- **Graph Expansion (BFS)** transverses relationship payload edges to discover causally connected thoughts and observations.
- A **Contextual Reranker** then uses a rapid LLM pass to surface the most vital memories for the current situation.

### 4. Asynchronous Ambient Mechanics
To maintain the illusion of a living, breathing world, complex interactions unfold asynchronously:
- **Non-Blocking Dialogues**: When agents initiate conversations, a background thread handles the ongoing dialogue generation, summarization, and physical broadcasting of the utterances to the world map.
- **The Overhearing Mechanic**: Uninvolved bystanders running their concurrent perception sweeps can detect broadcasted utterances, formatting them as third-person observations and committing them to memory. 
- **Dynamic World Engine**: A `WorldPhysicsSimulator` evaluates open-ended object interactions (e.g., "fixing the coffee machine") and seamlessly injects dynamic environmental consequences back into the simulation tile.

---

## 🛠️ Requirements
- **Python 3.13+**
- **DSPy**
- **Qdrant**

## 🚀 Getting Started

1. Ensure you have your LLM provider running (e.g., a local Ollama instance, vLLM, or the standard OpenAI API).
2. Sync the Python dependencies in the main repository using `uv`:
   ```bash
   uv sync
   ```
3. Run the main simulation loop:
   ```bash
   uv run python src/generative_agents/__main__.py
   ```
4. Start the frontend server:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## 📜 License
Licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).  
Based on the original [Generative Agents codebase](https://github.com/joonspk-research/generative_agents).
