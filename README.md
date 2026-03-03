# Agentic-AI-Survey
A dense project surveying the Agentic AI paradigm: architectures, methodologies, RAG systems, applications, challenges, and future directions. This repository consolidates knowledge from foundational presentations, peer-reviewed papers, and applied implementations.


## 📋 Table of Contents
1. [Overview & Motivation](#overview--motivation)
2. [What is Agentic AI?](#what-is-agentic-ai)
3. [Core Components of an AI Agent](#core-components-of-an-ai-agent)
4. [Key Characteristics](#key-characteristics)
5. [Agentic AI Strategies](#agentic-ai-strategies)
6. [Architectural Approaches & Methodologies](#architectural-approaches--methodologies)
7. [Agentic Workflow Patterns](#agentic-workflow-patterns)
8. [RAG Systems Taxonomy](#rag-systems-taxonomy)
9. [Agentic RAG Classification](#agentic-rag-classification)
10. [Large AI Model (LAM) Taxonomy](#large-ai-model-lam-taxonomy)
11. [Multi-Agent Orchestration Patterns](#multi-agent-orchestration-patterns)
12. [Training & Evaluation Techniques](#training--evaluation-techniques)
13. [Comparison Metrics](#comparison-metrics)
14. [Domain-Specific Applications](#domain-specific-applications)
15. [Security & Trust Threats](#security--trust-threats)
16. [Challenges & Limitations](#challenges--limitations)
17. [Case Studies](#case-studies)
18. [Tools & Frameworks](#tools--frameworks)
19. [References & Related Work](#references--related-work)


## Overview & Motivation
Most traditional AI systems are designed as supervised tools with predefined restrictions. They perform well on clearly delimited tasks but fail when confronted with dynamic, long-horizon, or parameter-free environments. The Agentic AI paradigm addresses this gap by enabling systems that **perceive, reason, act, and learn** in continuous feedback loops without constant human supervision.

### Classic failure examples:
- **Minecraft diamond acquisition** — requires long-term planning, tool use, adaptation, and exploration far beyond rule-based execution.
- **No Man's Sky** — traditional AI cannot generalize to 18 quintillion procedurally generated planets with unique terrain, creatures, and resources.


## What is Agentic AI?
Agentic AI refers to AI architectures designed to operate as autonomous or semi-autonomous agents capable of performing multi-step tasks, making decisions, and interacting with other agents or systems in a goal-directed manner. The term highlights the notion of **agency**: intentionality, autonomy, and purposeful behavior within defined boundaries.

| Property | Description |
|----------|-------------|
| **Autonomous** | Pursues multiple goals independently over extended time horizons |
| **Adaptive** | Responds to environmental change via real-time learning |
| **Self-Sufficient** | Executes high-level decisions without per-step human supervision |
| **Goal-Directed** | Maintains long-term objectives while handling dynamic sub-tasks |
| **Collaborative** | Can coordinate with other agents in multi-agent settings |


## Core Components of an AI Agent
```
┌─────────────────────────────────────────────────────────────┐
│                        AI AGENT                             │
│                                                             │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐   │
│   │   LLM    │   │ Planning │   │  Memory  │   │ Tools  │   │
│   │ (Reason) │   │(Decompose│   │(Short &  │   │(APIs,  │   │
│   │          │   │& Reflect)│   │Long-term)│   │Search) │   │
│   └──────────┘   └──────────┘   └──────────┘   └────────┘   │
└─────────────────────────────────────────────────────────────┘
```

| Component | Role | Examples |
|-----------|------|---------|
| **LLM Core** | Acts as reasoning engine; interprets input and generates responses | GPT-4, LLaMA 3, Claude |
| **Planning** | Breaks tasks into adaptive, strategic sub-steps; supports self-reflection and task decomposition | ReAct, Chain-of-Thought, MAP |
| **Memory (Short-Term)** | Tracks interaction context within a session | Conversation buffer, working memory |
| **Memory (Long-Term)** | Stores episodic and semantic knowledge across sessions | Vector DB, knowledge graphs |
| **Tool Use** | Leverages external functions for context-aware actions | APIs, vector search, code execution |
| **Self-Reflection** | Learns by critiquing and adjusting its own outputs | Iterative prompting, critic agents |


## Key Characteristics
| Characteristic | Description | Related Technique |
|----------------|-------------|-------------------|
| **Autonomy** | Pursues goals without step-by-step instructions | HRL, Goal-Oriented Design |
| **Adaptability** | Real-time learning from environment feedback | Meta-learning, curriculum learning |
| **Multi-Step Reasoning** | Chains logic across extended task horizons | Chain-of-Thought, Tree-of-Thought |
| **Tool Augmentation** | Dynamic use of external APIs and tools | ReAct, function calling |
| **Persistent Memory** | Retains and retrieves context across interactions | Episodic/semantic memory, RAG |
| **Inter-Agent Communication** | Structured exchange between specialized agents | OVON framework, JSON APIs |
| **Self-Improvement** | Iterative refinement through feedback loops | RL, meta-prompting, self-verification |


## Agentic AI Strategies
| Strategy | Description | Typical Use Cases |
|----------|-------------|-------------------|
| **Reinforcement Learning (RL)** | Agent improves by interacting with an environment and maximizing reward signals | Game playing, robotics, simulation |
| **Meta-Learning** | Learns to learn; stays flexible by generalizing from past experiences | Few-shot adaptation, rapid fine-tuning |
| **Hierarchical Goal Decomposition** | Complex tasks are broken into sub-goals for organized, step-by-step execution | Long-horizon planning, multi-stage workflows |
| **Curriculum Learning** | Progressive difficulty; builds skills from simple to complex tasks | Training agents in simulation |
| **Multi-Task Learning** | Trains agents for multiple goals simultaneously; enhances generalization | Versatile agents, shared representations |


## Architectural Approaches & Methodologies
| Approach | Description | Key References |
|----------|-------------|----------------|
| **Multi-Agent Systems (MAS)** | Multiple specialized agents collaborate to solve complex tasks | Park et al. 2023, Gosmar et al. 2025 |
| **Hierarchical RL (HRL)** | Nested policies; high-level goals delegate to low-level sub-policies | Acharya et al. 2025 |
| **Goal-Oriented Architecture** | Agent behavior driven by explicit goal representations | MAP (Mondal et al. 2023) |
| **Modular Agentic Architecture** | Planning via recurrent interaction of specialized LLM modules | MAP: Actor, Monitor, Predictor, Evaluator, Decomposer, Orchestrator |
| **Retrieval-Augmented Generation (RAG)** | Context-aware responses by accessing external knowledge at inference time | Singh et al. 2025 |
| **Instruction Fine-Tuning** | Improves agent handling of detailed, multi-step instructions | BDDTestAIGen (Paduraru et al. 2025) |
| **Self-Corrective / RL-Driven Agents** | RL policy guides meta-prompting agent to learn from errors | Amjad et al. 2025 (SOIRE, CORD) |

### Modular Agentic Planner (MAP) — Module Breakdown
| Module | Function |
|--------|----------|
| **Task Decomposer** | Converts high-level goals into ordered sub-goals |
| **Actor** | Proposes concrete actions given a state and sub-goal |
| **Monitor** | Gates actions against task constraints; provides feedback |
| **Predictor** | Forecasts next state given current state and proposed action |
| **Evaluator** | Estimates value of a predicted state for tree search |
| **Orchestrator** | Determines goal/sub-goal completion; emits final plan |


## Agentic Workflow Patterns
| Pattern | Description | Best For |
|---------|-------------|----------|
| **1. Prompt Chaining** | Sequential processing; each step feeds the next | Ordered tasks, multilingual content, structured outputs |
| **2. Routing** | Inputs are directed to the most appropriate specialized process | Customer support, model selection, expert routing |
| **3. Parallelization** | Multiple tasks run concurrently; results are aggregated (e.g., voting) | Content moderation, code analysis, speed-critical pipelines |
| **4. Orchestrator–Workers** | Central controller dynamically assigns tasks to worker agents | Complex, fluid, or unpredictable multi-step workflows |
| **5. Evaluator–Optimizer** | Iterative feedback loop refines outputs until quality criteria are met | Tasks with clear evaluation metrics, translation, summarization |


## RAG Systems Taxonomy
| RAG Type | Retrieval Method | Strengths | Limitations |
|----------|-----------------|-----------|-------------|
| **Naïve RAG** | Keyword / BM25 | Simple, fast, low overhead | Poor context, incoherence, no multi-hop reasoning |
| **Advanced RAG** | Dense vector / semantic search | Better relevance, multi-hop queries | Higher latency, more complex pipelines |
| **Modular RAG** | Hybrid (keyword + vector + tools) | Flexible components, customizable pipelines | Integration complexity |
| **Graph RAG** | Graph traversal + vector search | Multi-hop reasoning, structured knowledge | Graph construction cost, scalability |
| **Agentic RAG** | Dynamic agent-driven retrieval | Autonomous strategy selection, iterative refinement | Coordination complexity, compute overhead |

### Challenges of Traditional RAG
| Challenge | Description |
|-----------|-------------|
| **Contextual Integration** | Fragmented or generic outputs due to poor fusion of retrieved content |
| **Multi-Step Reasoning** | Inability to perform iterative reasoning for complex, multi-hop queries |
| **Scalability & Latency** | High data volume increases computational load; hinders real-time use |


## Agentic RAG Classification
| Type | Architecture | Key Feature | Use Case |
|------|-------------|-------------|----------|
| **1. Single-Agent Router** | Centralized agent selects best data sources | Real-time adaptability | General Q&A, document lookup |
| **2. Multi-Agent Agentic RAG** | Central coordinator + specialized retrieval agents | Parallel specialized retrieval | Complex enterprise queries |
| **3. Hierarchical Agentic RAG** | Top-tier agent delegates to sub-agents, integrates results | Multi-layer delegation | Large-scale knowledge bases |
| **4. Corrective Agentic RAG** | Agents retrieve, evaluate, and refine using internal + external sources | Accuracy-focused refinement | High-stakes decision support |
| **5. Adaptive Agentic RAG** | Lightweight classifier selects retrieval strategy by query complexity | Dynamic strategy selection | Mixed-complexity query volumes |
| **6. Agent-G (Graph RAG)** | Specialized retrieval agents + critic module + dynamic collaboration | Structured + unstructured data fusion | Knowledge graph–rich domains |
| **7. GeAR** | Graph-enhanced expansion + intelligent agents + LLM synthesis | Graph-based context enrichment | Knowledge-intensive NLP tasks |
| **8. Agentic Document Workflows** | Document orchestration + context maintenance + business output generation | Business-specific document intelligence | Finance, legal, healthcare docs |


## Large AI Model (LAM) Taxonomy
*(From: Jiang et al. 2025 — Agentic AI for 6G Communications)*
| Category | Subtypes | Primary Modality | Application Focus |
|----------|----------|-----------------|-------------------|
| **Large Language Models (LLMs)** | GPT-4, LLaMA 3, Claude, Gemini | Text | NLP, reasoning, code generation |
| **Large Vision Models (LVMs)** | ViT, SAM, CLIP | Image/Video | Visual understanding, segmentation |
| **Large Multimodal Models (LMMs)** | GPT-4V, Gemini Ultra | Text + Image + Audio | Cross-modal tasks |
| **Large Reasoning Models (LRMs)** | o1, o3, DeepSeek-R1 | Text + Logic | Mathematical/scientific reasoning |
| **Lightweight LAMs** | Llama 3.1 8B, Phi-3, Mistral 7B | Text | On-device, resource-constrained environments |


### Key LAM Building Blocks
| Component | Role |
|-----------|------|
| **Transformer** | Core sequence modeling architecture |
| **Vision Transformer (ViT)** | Image patch-based attention |
| **Variational AutoEncoder (VAE)** | Latent space compression and generation |
| **Diffusion Model** | High-quality generative synthesis |
| **Mixture of Experts (MoE)** | Sparse conditional compute; scalable routing |







