# Agentic-AI-Survey
A dense project surveying the Agentic AI paradigm: architectures, methodologies, RAG systems, applications, challenges, and future directions. This repository consolidates knowledge from foundational presentations, peer-reviewed papers, and applied implementations.
<br>

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
<br>

## Overview & Motivation
Most traditional AI systems are designed as supervised tools with predefined restrictions. They perform well on clearly delimited tasks but fail when confronted with dynamic, long-horizon, or parameter-free environments. The Agentic AI paradigm addresses this gap by enabling systems that **perceive, reason, act, and learn** in continuous feedback loops without constant human supervision.

### Classic failure examples:
- **Minecraft diamond acquisition** — requires long-term planning, tool use, adaptation, and exploration far beyond rule-based execution.
- **No Man's Sky** — traditional AI cannot generalize to 18 quintillion procedurally generated planets with unique terrain, creatures, and resources.
<br>

## What is Agentic AI?
Agentic AI refers to AI architectures designed to operate as autonomous or semi-autonomous agents capable of performing multi-step tasks, making decisions, and interacting with other agents or systems in a goal-directed manner. The term highlights the notion of **agency**: intentionality, autonomy, and purposeful behavior within defined boundaries.

| Property | Description |
|----------|-------------|
| **Autonomous** | Pursues multiple goals independently over extended time horizons |
| **Adaptive** | Responds to environmental change via real-time learning |
| **Self-Sufficient** | Executes high-level decisions without per-step human supervision |
| **Goal-Directed** | Maintains long-term objectives while handling dynamic sub-tasks |
| **Collaborative** | Can coordinate with other agents in multi-agent settings |
<br>

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
<br>

| Component | Role | Examples |
|-----------|------|---------|
| **LLM Core** | Acts as reasoning engine; interprets input and generates responses | GPT-4, LLaMA 3, Claude |
| **Planning** | Breaks tasks into adaptive, strategic sub-steps; supports self-reflection and task decomposition | ReAct, Chain-of-Thought, MAP |
| **Memory (Short-Term)** | Tracks interaction context within a session | Conversation buffer, working memory |
| **Memory (Long-Term)** | Stores episodic and semantic knowledge across sessions | Vector DB, knowledge graphs |
| **Tool Use** | Leverages external functions for context-aware actions | APIs, vector search, code execution |
| **Self-Reflection** | Learns by critiquing and adjusting its own outputs | Iterative prompting, critic agents |
<br>

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
<br>

## Agentic AI Strategies
| Strategy | Description | Typical Use Cases |
|----------|-------------|-------------------|
| **Reinforcement Learning (RL)** | Agent improves by interacting with an environment and maximizing reward signals | Game playing, robotics, simulation |
| **Meta-Learning** | Learns to learn; stays flexible by generalizing from past experiences | Few-shot adaptation, rapid fine-tuning |
| **Hierarchical Goal Decomposition** | Complex tasks are broken into sub-goals for organized, step-by-step execution | Long-horizon planning, multi-stage workflows |
| **Curriculum Learning** | Progressive difficulty; builds skills from simple to complex tasks | Training agents in simulation |
| **Multi-Task Learning** | Trains agents for multiple goals simultaneously; enhances generalization | Versatile agents, shared representations |
<br>

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
<br>

### Modular Agentic Planner (MAP) — Module Breakdown
| Module | Function |
|--------|----------|
| **Task Decomposer** | Converts high-level goals into ordered sub-goals |
| **Actor** | Proposes concrete actions given a state and sub-goal |
| **Monitor** | Gates actions against task constraints; provides feedback |
| **Predictor** | Forecasts next state given current state and proposed action |
| **Evaluator** | Estimates value of a predicted state for tree search |
| **Orchestrator** | Determines goal/sub-goal completion; emits final plan |
<br>

## Agentic Workflow Patterns
| Pattern | Description | Best For |
|---------|-------------|----------|
| **1. Prompt Chaining** | Sequential processing; each step feeds the next | Ordered tasks, multilingual content, structured outputs |
| **2. Routing** | Inputs are directed to the most appropriate specialized process | Customer support, model selection, expert routing |
| **3. Parallelization** | Multiple tasks run concurrently; results are aggregated (e.g., voting) | Content moderation, code analysis, speed-critical pipelines |
| **4. Orchestrator–Workers** | Central controller dynamically assigns tasks to worker agents | Complex, fluid, or unpredictable multi-step workflows |
| **5. Evaluator–Optimizer** | Iterative feedback loop refines outputs until quality criteria are met | Tasks with clear evaluation metrics, translation, summarization |
<br>

## RAG Systems Taxonomy
| RAG Type | Retrieval Method | Strengths | Limitations |
|----------|-----------------|-----------|-------------|
| **Naïve RAG** | Keyword / BM25 | Simple, fast, low overhead | Poor context, incoherence, no multi-hop reasoning |
| **Advanced RAG** | Dense vector / semantic search | Better relevance, multi-hop queries | Higher latency, more complex pipelines |
| **Modular RAG** | Hybrid (keyword + vector + tools) | Flexible components, customizable pipelines | Integration complexity |
| **Graph RAG** | Graph traversal + vector search | Multi-hop reasoning, structured knowledge | Graph construction cost, scalability |
| **Agentic RAG** | Dynamic agent-driven retrieval | Autonomous strategy selection, iterative refinement | Coordination complexity, compute overhead |
<br>

### Challenges of Traditional RAG
| Challenge | Description |
|-----------|-------------|
| **Contextual Integration** | Fragmented or generic outputs due to poor fusion of retrieved content |
| **Multi-Step Reasoning** | Inability to perform iterative reasoning for complex, multi-hop queries |
| **Scalability & Latency** | High data volume increases computational load; hinders real-time use |
<br>


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
<br>

## Large AI Model (LAM) Taxonomy
*(From: Jiang et al. 2025 — Agentic AI for 6G Communications)*
| Category | Subtypes | Primary Modality | Application Focus |
|----------|----------|-----------------|-------------------|
| **Large Language Models (LLMs)** | GPT-4, LLaMA 3, Claude, Gemini | Text | NLP, reasoning, code generation |
| **Large Vision Models (LVMs)** | ViT, SAM, CLIP | Image/Video | Visual understanding, segmentation |
| **Large Multimodal Models (LMMs)** | GPT-4V, Gemini Ultra | Text + Image + Audio | Cross-modal tasks |
| **Large Reasoning Models (LRMs)** | o1, o3, DeepSeek-R1 | Text + Logic | Mathematical/scientific reasoning |
| **Lightweight LAMs** | Llama 3.1 8B, Phi-3, Mistral 7B | Text | On-device, resource-constrained environments |
<br>


### Key LAM Building Blocks
| Component | Role |
|-----------|------|
| **Transformer** | Core sequence modeling architecture |
| **Vision Transformer (ViT)** | Image patch-based attention |
| **Variational AutoEncoder (VAE)** | Latent space compression and generation |
| **Diffusion Model** | High-quality generative synthesis |
| **Mixture of Experts (MoE)** | Sparse conditional compute; scalable routing |
<br>


## Multi-Agent Orchestration Patterns
| Pattern | Structure | Communication Mechanism | Best For |
|---------|-----------|------------------------|----------|
| **Sequential Pipeline** | Agent A → Agent B → Agent C | Output chaining | Hallucination mitigation, review workflows |
| **Parallel Specialized Agents** | Multiple agents work simultaneously | Aggregated results | Speed, diverse expertise |
| **Hierarchical Delegation** | Manager agent → worker agents | Task assignment + result integration | Large-scale complex tasks |
| **Critic–Generator Loop** | Generator + Critic in dialogue | Iterative feedback | Scientific reasoning, feature selection |
| **OVON NLP Framework** | Agents communicate via natural language JSON APIs | Universal NLP-based interfaces | Interoperable multi-system agents |
<br>

### Hallucination Mitigation via Multi-Agent Pipelines
*(From: Gosmar & Dahl 2025)*
| Agent Level | Role | LLM Used |
|-------------|------|----------|
| **Front-End Agent** | Generates initial response | Variable |
| **Second-Level Agent** | Detects unverified claims; adds disclaimers | Separate LLM |
| **Third-Level Agent** | Clarifies speculative content; refines factuality | Separate LLM |
| **Fourth-Level Evaluator** | Measures KPIs; quantifies hallucination scores | Dedicated evaluator LLM |
<br>

#### Hallucination KPIs
| KPI | Definition |
|-----|-----------|
| **Factual Claim Density** | Ratio of verifiable factual claims per response unit |
| **Factual Grounding References** | Count of claims anchored to identifiable sources |
| **Fictional Disclaimer Frequency** | How often speculative content is explicitly flagged |
| **Explicit Contextualization Score** | Degree to which context boundaries are demarcated |
<br>


## Training & Evaluation Techniques
| Technique | Description | Advantage |
|-----------|-------------|-----------|
| **Simulation-Based Training** | Safe, repeatable environments for policy learning | Risk-free exploration, scalable data generation |
| **Curriculum Learning** | Progressive task difficulty; builds skills incrementally | Faster convergence, better generalization |
| **Multi-Task Learning** | Joint training across multiple goal domains | Enhanced transfer and generalization |
| **Iterative Prompting** | Feedback + errors re-injected into LLM prompt per round | Self-correcting without gradient updates |
| **Self-Verification** | Separate agent validates task success and critiques failure | Reduces hallucination, improves task completion |
| **RL with Gymnasium** | Natural language action space; reward/penalty policy | Adaptive prompt optimization |
| **Ablation Studies** | Systematic component removal to measure individual contribution | Identifies critical components |
<br>


## Comparison Metrics
| Metric | Description | Scope |
|--------|-------------|-------|
| **Adaptability & Learning Speed** | How quickly the agent adjusts to changing environments with minimal performance loss | Generalization |
| **Goal Efficiency** | Achievement of objectives with minimum resource expenditure | Optimization |
| **Robustness** | Stability under environmental disruptions or adversarial inputs | Reliability |
| **Scalability** | Performance maintained as task complexity and volume grow | Production readiness |
| **Task Success Rate** | Proportion of tasks completed successfully | Performance |
| **Exact Match (EM)** | Ratio of perfectly matched extracted fields to total fields | Document extraction |
| **Cosine Similarity** | Semantic similarity between generated and ground-truth output | NLP quality |
| **Hallucination Score (THS)** | Composite score from multi-agent KPI evaluators | Trustworthiness |
| **TrueSkill Rating** | Bayesian skill-rating for comparative agent evaluation | Human-agent comparison |
<br>

## Domain-Specific Applications
| Domain | System / Approach | Key Capability | Source |
|--------|------------------|----------------|--------|
| **Open-Ended Game Environments** | VOYAGER (Minecraft) | Automatic curriculum, skill library, iterative prompting, self-verification | Wang et al. 2023 |
| **Social Simulation** | Generative Agents (Smallville) | Memory + reflection + planning; emergent social behaviors | Park et al. 2023 |
| **Healthcare Simulation** | Agent Hospital + MedAgent-Zero | Evolvable doctor agents; 10,000+ cases/day; self-evolution via case experience | Li et al. 2024 |
| **Document Intelligence** | Agentic Form Extraction (SOIRE/CORD) | RL-driven multi-agent extraction from invoices, receipts, purchase orders | Amjad et al. 2025 |
| **Social Media Analysis** | BEYONDWORDS | Thematic extraction from autism community tweets; CoT + LLM QA | Ghali et al. 2025 |
| **Climate Finance** | EW4All Financial Tracking AI | Agent-based RAG for EWS investment classification; 87% accuracy | Vaghefi et al. 2025 |
| **IoT / Real-Time Search** | IoT-ASE (SensorsConnect) | LLM + RAG for real-time IoT data search; 92% intent-retrieval accuracy | Elewah & Elgazzar 2025 |
| **Software Testing** | BDDTestAIGen | Agentic BDD test generation with human-in-the-loop and fine-tuned LLaMA 3.1 8B | Paduraru et al. 2025 |
| **Quantum Chemistry** | xChemAgents | Selector + Validator cooperative agents for explainable molecular property prediction | Polat et al. 2025 |
| **6G Communications** | LAM-based Agentic AI | Resource management, network optimization, intelligent routing in 6G | Jiang et al. 2025 |
| **Multi-Step NLP Planning** | MAP (Modular Agentic Planner) | Recurrent specialized LLM modules for graph traversal, Tower of Hanoi, StrategyQA | Mondal et al. 2023 |
| **Agentic Task Generation** | TaskCraft | Automated difficulty-scalable, multi-tool agentic task synthesis (~36k tasks) | OPPO AI Team 2025 |
<br>


## Security & Trust Threats
*(From: Khan et al. 2024 — Security Threats in Agentic AI Systems)*
| Threat Category | Description | Impact |
|----------------|-------------|--------|
| **Unauthorized Data Retrieval** | AI agent accesses sensitive DB records without proper authorization | Data breach, privacy violation |
| **Adversarial Manipulation** | Malicious actors exploit agent autonomy to extract or tamper data | Integrity compromise |
| **Data Leakage via NLP Queries** | Natural language DB queries unintentionally expose sensitive fields | Confidentiality risk |
| **Scalability-Induced Lapses** | Performance optimization under load leads to security trade-offs | Reliability and security degradation |
| **Attack Surface Expansion** | More autonomous agents = larger attack surface | System-wide vulnerability |
| **Prompt Injection** | Adversarial inputs embedded in retrieved content hijack agent behavior | Goal misdirection, data exfiltration |
| **Hallucination Exploitation** | Fabricated outputs used to mislead downstream processes or users | Misinformation, trust erosion |
<br>

### Ethical Dimensions
| Dimension | Concern |
|-----------|---------|
| **Privacy** | Agents accessing personal or confidential data at scale |
| **Consent** | Users unaware of what data agents retrieve or share |
| **Explainability** | Black-box agent decisions are difficult to audit |
| **Accountability** | Unclear responsibility when autonomous agents cause harm |
| **Bias Amplification** | Agents may inherit and propagate biases from training data |
<br>


## Challenges & Limitations
| Challenge | Description | Potential Mitigation |
|-----------|-------------|----------------------|
| **Coordination Complexity** | Managing multi-agent interactions requires advanced orchestration | Hierarchical controllers, standardized APIs (OVON) |
| **Computational Overhead** | Multiple agents increase resource demands | Lightweight models, sparse MoE, model distillation |
| **Scalability Limits** | Dynamic systems can be overwhelmed at high query volumes | Load balancing, adaptive retrieval strategies |
| **Memory Overload** | Agents may misinterpret or lose critical context over long sessions | Structured memory with recency/importance scoring |
| **Hallucination** | LLMs generate confident but incorrect outputs | Multi-agent verification pipelines, RAG grounding |
| **Reward Hacking** | RL agents find unintended ways to maximize rewards | Careful reward shaping, curriculum constraints |
| **Generalization** | Agents trained in one environment may fail in others | Meta-learning, domain-randomization |
| **Identity Consistency** | Agents may drift from their defined role or persona over time | Instruction tuning, persona anchoring |
| **Ethical & Alignment Risks** | Autonomous goal pursuit may conflict with human values | RLHF, constitutional AI, guardrails |
| **Evaluation Difficulty** | No unified benchmark captures all agentic capabilities | TaskCraft, GAIA, BrowseComp, HLE |
<br>




## Case Studies
### Case Study 1 — VOYAGER: Open-Ended Embodied Agent in Minecraft
*(Wang et al., 2023 — arXiv:2310.00194)*
| Component | Description |
|-----------|-------------|
| **Automatic Curriculum** | GPT-4 generates progressively harder tasks based on agent state, inventory, and task history |
| **Skill Library** | Executable, embedding-indexed code-based skills; continuously expanded |
| **Iterative Prompting** | Environment feedback + execution errors + self-verification re-injected into GPT-4 each round |
| **Self-Verification** | Separate GPT-4 agent confirms task success; critiques failure for next iteration |
| **LLM APIs** | gpt-4-0314, gpt-3.5-turbo-0301, text-embedding-ada-002 |
| **Simulation** | MineDojo + Mineflayer (JavaScript motor controls) |
<br>

**Ablation Results:**
| Removed Component | Performance Drop |
|-------------------|-----------------|
| Automatic Curriculum | −93% discovered items |
| Self-Verification | −73% discovered items |
| GPT-3.5 vs GPT-4 | GPT-4 produces 5.7× more unique items |
<br>


### Case Study 2 — Generative Agents: Interactive Simulacra of Human Behavior
*(Park et al., 2023)*
| Component | Description |
|-----------|-------------|
| **Memory & Retrieval** | Ranked by recency, importance, and relevance; top-k fed to LLM |
| **Reflection** | Converts multiple observations into abstract insights; triggered periodically |
| **Planning & Reacting** | Maintains and updates daily plans; balances long-term behavior with real-time responses |
| **Evaluation** | 100 U.S.-based human raters; TrueSkill rating system |
<br>

**Emergent Behaviors:** Information diffusion through natural conversation, relationship memory, autonomous task coordination (e.g., party organization without user prompting).
<br>


### Case Study 3 — Agent Hospital: Evolvable Medical Agents
*(Li et al. 2024)*
| Component | Description |
|-----------|-------------|
| **Agents** | 14 doctors + 4 nurses + resident agents (can become patients) |
| **Medical Pipeline** | Disease onset → triage → registration → consultation → examination → diagnosis → treatment → follow-up |
| **MedAgent-Zero** | Doctor agents self-evolve via case experience + medical record library + experience base |
| **Throughput** | Simulates 10,000+ cases/day (equivalent to 2 years of a human doctor's experience) |
| **LLM Backends** | GPT-3.5 and GPT-4 tested |
<br>


## Tools & Frameworks
| Tool / Framework | Category | Key Capability |
|-----------------|----------|----------------|
| **LangChain / LangGraph** | Orchestration | Chain-based agent pipelines; graph-based multi-agent workflows |
| **LlamaIndex** | RAG | Document indexing, retrieval, and agent integration |
| **CrewAI** | Multi-Agent | Role-based agent crews for collaborative task execution |
| **AutoGen / AG2** | Multi-Agent | Conversational agent orchestration; code-executing agents |
| **OpenAI Swarm** | Multi-Agent | Lightweight multi-agent coordination framework |
| **Semantic Kernel** | Orchestration | Microsoft's AI orchestration SDK; plugin-based architecture |
| **Hugging Face Transformers + Qdrant** | RAG + Embeddings | Open-source LLMs + high-performance vector database |
| **Amazon Bedrock** | Cloud RAG | Managed agentic RAG on AWS infrastructure |
| **Vertex AI (Google)** | Cloud RAG | Managed agentic RAG on Google Cloud |
| **IBM Watson** | Enterprise AI | Agentic RAG for enterprise knowledge management |
| **Neo4j + Vector DBs** | Graph + RAG | Graph-enhanced retrieval with structured knowledge |
| **Gymnasium (OpenAI)** | RL Training | Natural language action space RL framework |
| **MineDojo / Mineflayer** | Simulation | Minecraft-based agent simulation environments |
| **OVON Framework** | Interoperability | Universal NLP-based API for inter-agent communication |
| **TaskCraft** | Benchmarking | Automated generation of 36k+ agentic tasks with trajectories |
<br>


## References & Related Work
| # | Title | Authors | Year | Source | Key Topic |
|---|-------|---------|------|--------|-----------|
| 1 | Agentic AI: Autonomous Intelligence for Complex Goals — A Comprehensive Survey | Acharya, Kuppan, Divya | 2025 | Journal | Agentic AI survey |
| 2 | Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG | Singh, Ehtesham, Kumar, Khoei | 2025 | arXiv | Agentic RAG taxonomy |
| 3 | VOYAGER: An Open-Ended Embodied Agent with Large Language Models | Wang et al. | 2023 | NeurIPS | Embodied open-ended learning |
| 4 | Generative Agents: Interactive Simulacra of Human Behavior | Park et al. | 2023 | UIST | Social simulation |
| 5 | Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents | Li et al. | 2024 | arXiv | Healthcare agents |
| 6 | An Agentic System with Reinforcement-Learned Subsystem Improvements for Parsing Form-Like Documents | Amjad, Sthapit, Syed | 2025 | arXiv:2505.13504 | Document intelligence + RL |
| 7 | Hallucination Mitigation using Agentic AI Natural Language-Based Frameworks | Gosmar, Dahl | 2025 | arXiv:2501.13946 | Hallucination mitigation |
| 8 | BEYONDWORDS: Agentic Generative AI based Social Media Themes Extractor | Ghali et al. | 2025 | arXiv:2503.01880 | Social media NLP |
| 9 | TaskCraft: Automated Generation of Agentic Tasks | OPPO AI Team | 2025 | arXiv:2506.10055 | Agentic benchmarking |
| 10 | From Large AI Models to Agentic AI: A Tutorial on Future Intelligent Communications | Jiang et al. | 2025 | arXiv:2505.22311 | 6G + Agentic AI |
| 11 | Security Threats in Agentic AI System | Khan, Sarkar, Mahata, Jose | 2024 | arXiv:2410.14728 | Security & privacy |
| 12 | Improving Planning with Large Language Models: A Modular Agentic Architecture | Mondal, Webb, Momennejad | 2023 | arXiv:2310.00194 | Planning architectures |
| 13 | AI for Climate Finance: Agentic Retrieval and Multi-Step Reasoning for EWS Investments | Vaghefi et al. | 2025 | arXiv:2504.05104 | Climate finance RAG |
| 14 | Agentic Search Engine for Real-Time IoT Data | Elewah, Elgazzar | 2025 | arXiv:2503.12255 | IoT + RAG |
| 15 | xChemAgents: Agentic AI for Explainable Quantum Chemistry | Polat et al. | 2025 | arXiv:2505.20574 | Scientific agents |
| 16 | Agentic AI for Behavior-Driven Development Testing Using LLMs | Paduraru, Zavelca, Stefanescu | 2025 | ICAART / Scitepress | Software testing |
