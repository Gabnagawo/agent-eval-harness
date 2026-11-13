# State of the Art in Tool-Using AI Agent Evaluation

**Author:** Peter (Research Agent)  
**Project:** `agent-eval-harness`  
**Date:** November 2026  
**Status:** Complete  

---

## 1. Executive Summary

Autonomous AI agents that interact with external environments via tools and function calling represent a fundamental paradigm shift from static text generation. Evaluating tool-using agents requires moving beyond classical NLP benchmarks (e.g., MMLU, GSM8K) toward interactive, multi-turn, stateful evaluation frameworks.

This research document analyzes the state of the art in tool-using agent evaluation, examining five landmark benchmarks and frameworks:
1. **$\tau$-bench (tau-bench)** — Dynamic Tool-Agent-User interaction in real-world policy domains (Sierra / Princeton).
2. **BFCL (Berkeley Function Calling Leaderboard)** — Comprehensive multi-language function calling and AST/executable evaluation (Gorilla / UC Berkeley).
3. **AgentBench** — Multi-environment operating system, database, and web interaction suite (Tsinghua / OpenBMB).
4. **SWE-bench & SWE-bench Verified** — Real-world software issue resolution via unit test suites (Princeton / OpenAI).
5. **Inspect AI** — Secure, extensible agent evaluation framework with sandboxing and model-graded scoring (UK AI Safety Institute).

It also examines the mathematical foundations of the **$\text{pass}^k$** reliability metric (in contrast to $\text{pass@k}$), catalogs seven critical pitfalls in agent benchmarking, and provides five concrete, architectural recommendations for our local evaluation harness (`agent-eval-harness`).

---

## 2. Benchmark Profiles & Evaluation Methodologies

### 2.1 $\tau$-bench ($\tau$-bench)

* **Origin & Authors:** Sierra Research & Princeton University (Shunyu Yao, Noah Shinn, Jeffrey Zhao, et al., June 2024).
* **Paper:** [$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045)
* **Repository:** [github.com/sierra-research/tau-bench](https://github.com/sierra-research/tau-bench)
* **Domains:** Customer service in **Retail** and **Airline** environments (expanded in $\tau^2$ and $\tau^3$ to telecom and duplex multimodal interactions).

#### Core Architecture & Environment Setup
$\tau$-bench models realistic enterprise workflows where an agent interacts simultaneously with:
1. A **user** (simulated by an LLM with hidden profile variables and dynamic preferences),
2. A suite of **domain-specific tools/APIs** (e.g., searching databases, updating bookings, calculating refund eligibility),
3. A set of **strict domain policies** (written natural-language business rules governing when actions are permissible).

Unlike static conversational benchmarks, the user does not supply all necessary information upfront. The agent must converse to elicit required details, verify constraints against policy documents, call backend tools to read/modify relational databases, and communicate the resolution.

#### How $\tau$-bench Measures Success
* **Deterministic Database State Verification:** Evaluation does not rely on subjective text judging. At the end of the conversation, the final state of the backend database is compared against the annotated ground-truth target state.
* **Dual Compliance (Outcome + Policy):** A task is successful if and only if the final database matches the goal state **and** the agent did not violate business policies (e.g., executing a non-refundable cancellation or modifying a flight without collecting the fee).
* **Reliability Metric ($\text{pass}^k$):** Introduced the $\text{pass}^k$ metric to quantify whether an agent succeeds across all $k$ consecutive independent trials, exposing steep performance degradation in stochastic models.

---

### 2.2 Berkeley Function Calling Leaderboard (BFCL)

* **Origin & Authors:** Gorilla Project, UC Berkeley (Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, Shishir G. Patil, Joseph E. Gonzalez, 2024).
* **Paper / Documentation:** [The Berkeley Function Calling Leaderboard: From Tool Use to Agentic Evaluation of Large Language Models](https://gorilla.cs.berkeley.edu/leaderboard.html)
* **Repository:** [github.com/ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla)
* **Scope:** 2,000+ question-function pairs across multiple programming languages (Python, Java, JavaScript), REST APIs, and SQL.

#### Core Architecture & Categories
BFCL evaluates the raw capability of LLMs to generate valid, syntactically and semantically correct function calls from natural-language instructions and JSON schemas. It assesses six distinct capability tiers:
1. **Simple Function Calling:** Single tool invocation with standard scalar arguments.
2. **Multiple Function Calling:** Selecting the single correct tool from a pool of candidate tools.
3. **Parallel Function Calling:** Invoking multiple distinct tools or multiple instances of the same tool in a single turn.
4. **Parallel Multiple Function Calling:** Selecting and invoking multiple tools concurrently from a large tool catalog.
5. **Relevance Detection / Abstention:** Correctly identifying when *no* available tool matches the user's intent, abstaining rather than hallucinating an invalid call.
6. **Multi-Turn Function Calling (BFCL v2/v3):** State-dependent, multi-step tool execution where subsequent tool inputs depend on the outputs of prior tool calls.

#### How BFCL Measures Success
* **Abstract Syntax Tree (AST) Matching:** Evaluates generated code/JSON calls by parsing them into an AST. It verifies function names, argument names, argument data types, default parameter handling, and nested structures without requiring fragile string equality.
* **Executable Verification:** For live and mocked API scenarios, BFCL executes the emitted function call against actual or simulated endpoints and verifies that the resulting response or environment state matches the ground-truth execution trace.
* **Abstention Scoring:** Measures true negative rates (precision/recall on tool invocation necessity).

---

### 2.3 AgentBench

* **Origin & Authors:** Tsinghua University, Zhipu AI, Renmin University, Ohio State, UC Berkeley (Xiao Liu, Hao Yu, Hanchen Zhang, et al., 2023; published at ICLR 2024).
* **Paper:** [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688)
* **Repository:** [github.com/THUDM/AgentBench](https://github.com/THUDM/AgentBench)
* **Scope:** 8 diverse interactive environments spanning text, code, and web interfaces.

#### Core Architecture & Environments
AgentBench constructs standardized multi-turn interaction loops across eight distinct operational environments:
1. **Operating System (OS):** Executing bash commands in a containerized Linux environment to accomplish system administration tasks.
2. **Database (DB):** Formulating and running SQL queries over complex relational databases to extract or update information.
3. **Knowledge Graph (KG):** Querying structured knowledge graphs (e.g., Freebase) via Cypher/SPARQL.
4. **Digital Card Game:** Strategic multi-turn decision-making in competitive games.
5. **Lateral Thinking Puzzles:** Deductive questioning under constrained "yes/no" feedback.
6. **ALFWorld:** Multi-room text-based embodied household tasks (object manipulation, navigation).
7. **WebShop:** Searching, exploring, comparing, and purchasing products in a simulated e-commerce platform.
8. **Mind2Web:** Browser-based complex web navigation and form interaction.

#### How AgentBench Measures Success
* **Environment-Specific Success Rate (SR):**
  * OS: Verified by executing validation bash scripts checking final file system state, processes, and exit codes.
  * DB: Verified by matching the query output table against gold query results.
  * ALFWorld: Verified by reaching the formal boolean goal condition of the underlying simulator.
  * WebShop: Continuous reward metric based on product attribute overlap and final purchase match.
* **Failure Mode Taxonomy:** Decomposes agent failures into programmatic categories:
  * Format/Syntax Error (inability to generate parsable tool calls),
  * Context Limit Exceeded (failure to manage context size across extended turns),
  * Task Limit Exceeded (step budget exhaustion without completing the goal),
  * Incorrect Final State / Wrong Answer.

---

### 2.4 SWE-bench & SWE-bench Verified

* **Origin & Authors:** Princeton University (Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan, ICLR 2024) & OpenAI (SWE-bench Verified, August 2024).
* **Paper:** [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) | [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)
* **Repository:** [github.com/princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench)
* **Scope:** 2,294 real-world GitHub issue-and-pull-request pairs from 12 major Python repositories (e.g., `django/django`, `sympy/sympy`, `scikit-learn/scikit-learn`, `pytest-dev/pytest`).

#### Core Architecture & Execution Setup
SWE-bench provides agents with a natural-language issue description and an entire code repository checked out at the base commit immediately preceding the fix. The agent is given an execution environment (bash, file editor, search tools) and must produce a unified git patch (`git diff`).

#### How SWE-bench Measures Success
Success is strictly evaluated by executing the repository's native unit test suite inside isolated Docker containers using two explicit test subsets:
1. **$\text{FAIL\_TO\_PASS}$:** Tests created or modified in the original pull request that reproduce the bug (they fail on the base commit). The agent's patch **must cause all these tests to pass**.
2. **$\text{PASS\_TO\_PASS}$:** Existing unit tests that passed before the pull request. The agent's patch **must leave all these tests passing**, verifying that no regressions or breaking changes were introduced.

A task instance is scored as resolved ($1$) if and only if:
$$\text{Resolved} = \left(\bigwedge_{t \in \text{FAIL\_TO\_PASS}} t == \text{PASS}\right) \land \left(\bigwedge_{t \in \text{PASS\_TO\_PASS}} t == \text{PASS}\right)$$

* **SWE-bench Verified:** Addresses test flakiness and under-specified issues by human-validating 500 instances with confirmed reproducible test environments and unambiguous descriptions.

---

### 2.5 Inspect AI

* **Origin & Authors:** UK AI Safety Institute (UK AISI) / UK Department for Science, Innovation and Technology (DSIT), 2024.
* **Documentation:** [inspect.ai-safety-institute.org.uk](https://inspect.ai-safety-institute.org.uk/)
* **Repository:** [github.com/UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai)
* **Scope:** Open-source, enterprise-grade evaluation framework for frontier LLMs, autonomous agents, cybersecurity, and safety evaluations.

#### Core Architecture & Primitives
Inspect AI structures evaluations around three clean, orthogonal primitives:
1. **Dataset:** A collection of `Sample` objects containing input prompts, target outputs, metadata, and optional sandbox environments.
2. **Solver (Plan):** A composable pipeline of steps executed by the agent (e.g., system message injection, chain-of-thought, tool-calling loops, sub-agent spawning). Solvers interact with external systems through sandboxed tool protocols.
3. **Scorer:** Modular grading components that evaluate the solver's final output or environment state. Scorers range from deterministic exact/regex matches to model-graded multi-attribute evaluators (`model_graded_qa`).

#### Key Architectural Capabilities
* **Containerized Sandboxing:** Transparent Docker/Podman integration that isolates code execution, file operations, and bash tools inside disposable sandbox containers.
* **First-Class Agent Tool Loop:** Native support for tool definitions with JSON schema validation, automatic tool result capture, and step limits.
* **Observable Evaluation Traces:** Comprehensive recording of complete interaction trajectories (tool names, raw JSON arguments, stdout/stderr, token counts, per-step latencies) accessible via the `inspect view` web visualizer.

---

## 3. Metric Formulations & Mathematical Definitions

| Metric | Category | Primary Reference / Source | Mathematical Definition |
|---|---|---|---|
| **Success Rate (SR)** | Outcome | AgentBench (Liu et al., 2024) | $\text{SR} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{task}_i \text{ resolved})$ |
| **$\text{pass@k}$** | Capability | HumanEval (Chen et al., 2021) | $\text{pass@k} = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$ |
| **$\text{pass}^k$** | Reliability | $\tau$-bench (Yao et al., 2024) | $\text{pass}^k = \mathbb{E}\left[\frac{\binom{c}{k}}{\binom{n}{k}}\right]$ |
| **Valid Call Rate** | Schema | BFCL (Yan et al., 2024) | $\text{VCR} = \frac{\sum \text{schema-valid calls}}{\sum \text{total attempted calls}}$ |
| **Tool Precision** | Selection | Classical Information Retrieval | $\text{Precision} = \frac{|\mathcal{T}_{\text{called}} \cap \mathcal{T}_{\text{expected}}|}{|\mathcal{T}_{\text{called}}|}$ |
| **Tool Recall** | Selection | Classical Information Retrieval | $\text{Recall} = \frac{|\mathcal{T}_{\text{called}} \cap \mathcal{T}_{\text{expected}}|}{|\mathcal{T}_{\text{expected}}|}$ |
| **Step Ratio** | Efficiency | WebShop / AgentBench | $\text{Step Ratio} = \frac{\text{Actual Steps}}{\text{Optimal Steps}} \quad (\text{successful runs only})$ |
| **Forbidden Action Rate** | Safety | Inspect AI / AI Safety Institute | $\text{FAR} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{forbidden calls} > 0)$ |

---

### Detailed Mathematical Formulations

#### 3.1 The $\text{pass@k}$ vs. $\text{pass}^k$ Duality

The distinction between $\text{pass@k}$ and $\text{pass}^k$ is the single most critical conceptual distinction in modern agent evaluation.

##### $\text{pass@k}$ (At Least One Success — Capability Potential)
Introduced in code generation (HumanEval, Chen et al., 2021 [arXiv:2107.03374]), $\text{pass@k}$ measures the probability that an agent solves a task **at least once** when allowed $k$ independent generation attempts. Given $n$ total evaluated samples ($n \ge k$) with $c$ correct samples, the unbiased minimum-variance estimator is:

$$\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

* **Operational meaning:** "If an automated test harness or human selector filters out all incorrect attempts, what is the chance that the best solution succeeds?"
* **Limitation for Agents:** Autonomous agents in production operate without an external oracle picking the successful trajectory. A customer service agent or database admin cannot take 3 random actions and rely on the customer to discard the 2 destructive runs.

##### $\text{pass}^k$ (All $k$ Successes — Operational Reliability)
Introduced by Sierra Research in $\tau$-bench (Yao et al., 2024 [arXiv:2406.12045]), $\text{pass}^k$ (pronounced *pass-to-the-k* or *pass-hat-k*) measures the probability that an agent successfully completes a task on **all $k$ consecutive, independent trials**:

$$\text{pass}^k = \frac{\binom{c}{k}}{\binom{n}{k}}$$

When evaluating exactly $k$ runs per task ($n = k$):
$$\text{pass}^k = \begin{cases} 1 & \text{if } c = k \\ 0 & \text{if } c < k \end{cases}$$

The benchmark-level $\text{pass}^k$ score is the mean over all $M$ tasks:
$$\text{pass}^k_{\text{benchmark}} = \frac{1}{M} \sum_{m=1}^{M} \text{pass}^k_m$$

##### Analytical Comparison for Single-Trial Success Probability $p$
Assuming Bernoulli trials with independent success probability $p$:
* Theoretical $\text{pass@k} = 1 - (1 - p)^k$
* Theoretical $\text{pass}^k = p^k$

| Single Run $p$ | $\text{pass@3}$ (Best-of-3) | $\text{pass}^3$ (Reliability-3) | $\text{pass}^5$ (Reliability-5) | Operational Implication |
|---|---|---|---|---|
| **$0.90$** | $99.9\%$ | $72.9\%$ | $59.0\%$ | High single-run capability, moderate repeatability |
| **$0.70$** | $97.3\%$ | $34.3\%$ | $16.8\%$ | Appears competent under $\text{pass@3}$, fails 2/3 of deployments |
| **$0.50$** | $87.5\%$ | $12.5\%$ | $3.1\%$ | Severe reliability collapse in production |
| **$0.30$** | $65.7\%$ | $2.7\%$ | $0.2\%$ | Functionally unusable autonomously |

Reporting $\text{pass@k}$ for agents creates a profound false sense of security. $\text{pass}^k$ is the true measure of whether an agent can be trusted in production workflows.

---

#### 3.2 Tool Call Accuracy & Selection Metrics

##### Valid Call Rate (VCR)
$$\text{VCR} = \frac{\sum_{t=1}^{T} \mathbb{I}(\text{call}_t \in \mathcal{T}_{\text{catalogue}} \land \text{validate}(\text{args}_t, \text{schema}) == \text{Valid})}{\sum_{t=1}^{T} 1}$$
* Penalizes hallucinations where an agent invents fictitious tool names or produces arguments violating JSON schema types, required fields, or enum constraints.

##### Tool Precision and Tool Recall
Given the set of distinct, valid tools invoked during a trial $\mathcal{T}_{\text{called}}$ and the minimal expected set of tools necessary to resolve the task $\mathcal{T}_{\text{expected}}$:
$$\text{Precision} = \frac{|\mathcal{T}_{\text{called}} \cap \mathcal{T}_{\text{expected}}|}{|\mathcal{T}_{\text{called}}|}, \quad \text{Recall} = \frac{|\mathcal{T}_{\text{called}} \cap \mathcal{T}_{\text{expected}}|}{|\mathcal{T}_{\text{expected}}|}$$
* **High Recall, Low Precision:** Indicates exploratory wandering, shotgun tool execution, or excessive reliance on extraneous tools.
* **Low Recall:** Indicates failure to identify the required tools to fulfill the user's objective.

---

#### 3.3 Trajectory Efficiency & Cost

##### Step Ratio
For trials that successfully meet the task goal:
$$\text{Step Ratio} = \frac{S_{\text{actual}}}{S_{\text{optimal}}}$$
Where $S_{\text{optimal}} = \text{len}(\text{reference\_solution})$.
* A ratio of $1.0$ indicates an optimal, direct path.
* Ratios $> 1.0$ quantify trajectory bloat, unnecessary retries, redundant searches, and exploratory dithering.
* Must be computed **strictly on successful runs**; failed runs that abort early after 1 step would otherwise falsely appear "efficient".

---

#### 3.4 Safety & Guardrail Metrics

##### Forbidden Action Rate (FAR) & Safe Goal Rate
$$\text{FAR} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{forbidden\_actions}_i > 0)$$
$$\text{Safe} = \mathbb{I}(\text{forbidden\_actions} == 0)$$
$$\text{Success} = \text{Goal Met} \land \text{Safe}$$
* In safety-critical agent evaluation, goal achievement without safety is failure. If an agent resolves a customer support ticket by executing an unauthorized administrative override or deleting user files, the trial is categorized as an overall failure.

---

## 4. Common Pitfalls in Tool-Using Agent Evaluation

Through analysis of the literature and empirical benchmark designs, seven critical failure modes emerge in agent evaluation suites:

### Pitfall 1: Conflating $\text{pass@k}$ with Production Capability
Many evaluation suites inherit the $\text{pass@k}$ metric from code generation benchmarks. Because sampling temperature $T > 0$ introduces variance, an agent that succeeds 1 out of 5 times achieves a $\text{pass@5}$ of over $67\%$. In production, agents do not run in parallel with a magical oracle selecting the successful output; every failed run causes user friction, corrupted databases, or wasted compute. Benchmarks must evaluate and highlight $\text{pass}^k$.

### Pitfall 2: Flaky, Mutable Environments & State Bleed
Running evaluations against live external services, shared test databases, or unseeded random environments introduces non-determinism. If Trial 1 creates an issue `QA-10` and Trial 2 runs without resetting the database, Trial 2 encounters an unexpected ID `QA-11` or state conflict. Benchmarks must guarantee **byte-for-byte reproducible snapshots** via deep copy initialization and deterministic seeded random generators.

### Pitfall 3: Trajectory Over-Specification (Enforcing the "How" over the "What")
Early evaluation frameworks checked whether an agent called exact tool sequences (e.g., `assert calls == [search, get, update]`). If an intelligent agent solves the problem by calling `filter` instead of `search`, or checks an item's status before updating it, trajectory-matching tests fail valid solutions. The gold standard (established by $\tau$-bench, SWE-bench, and AgentBench) is **declarative final state verification**: assert that the environment reached the required state, irrespective of the intermediate discovery path.

### Pitfall 4: Neglecting Side Effects & Forbidden Boundaries
Standard benchmarks reward task completion while completely blind to collateral damage. For instance, an agent tasked with cleaning up old logs that executes `rm -rf /` or deletes active database records may achieve its objective while causing catastrophic side effects. An evaluation harness must define explicit **forbidden actions** (destructive calls without proper confirmation, unauthorized resource access) and track violations as a first-class metric.

### Pitfall 5: Prompt Contamination & Tool Schema Leakage
When benchmarks use static public datasets, models memorize task answers and schema structures. Furthermore, harness developers often leak task hints in error messages (e.g., returning `"Error: Expected user ID to be 42"` instead of `"Error: User not found"`). Tool error responses must mirror realistic production APIs without acting as an oracle guide.

### Pitfall 6: Uncalibrated "LLM-as-a-Judge" Evaluators
Relying on an uncalibrated LLM judge to grade agent actions introduces severe noise:
* **Position bias:** Favoring the first or second presented option.
* **Verbosity bias:** Grading longer, articulate explanations higher regardless of whether the tool actually solved the issue.
* **Self-preference bias:** LLM judges rating outputs generated by their own family higher than competing models.
Deterministic assertions (state snapshots, JMESPath queries, regex/exact matches) must be the primary evaluation mechanism, reserving LLM judges strictly for nuanced natural language responses with hand-labeled human correlation.

### Pitfall 7: Hidden Context Bloat & Distorted Cost Accounting
In tool-calling protocols, the JSON schemas of all available tools are injected into the model's system prompt on **every single step** of the multi-turn interaction. If an agent has access to 20 detailed tools, each turn consumes 2,000+ prompt tokens before the model generates a single action. Harnesses that only count output tokens or ignore schema overhead report severely distorted cost and latency metrics.

---

## 5. Five Concrete Recommendations for `agent-eval-harness`

Based on the state-of-the-art benchmarks ($\tau$-bench, BFCL, AgentBench, SWE-bench, Inspect AI) and tailored specifically to the architecture of `agent-eval-harness` (defined in `TASK-agent-eval-harness.md` and `CONTRACTS.md`), here are five concrete, actionable recommendations:

### Recommendation 1: Decouple State Verification from Trajectory Exploration via Declarative JMESPath Checkers
* **Why:** In accordance with $\tau$-bench and SWE-bench best practices, avoid rigid trajectory assertions. Agents (especially when comparing Baseline vs. Guarded) should be free to explore, verify preconditions, or inspect files.
* **How in our harness:** Maintain the design in `CONTRACTS.md` where `success` assertions operate exclusively as JMESPath queries over the `World.snapshot()` dictionary (e.g., `path: tracker.issues."QA-7".status, op: equals, value: resolved`). Use `reference_solution` solely to define `optimal_steps` for efficiency calculations, never as a required execution sequence.

### Recommendation 2: Implement Dual-Faceted Evaluation (`success = goal_met AND safe`) with Destructive Tool Guardrails
* **Why:** As demonstrated by Inspect AI safety benchmarks, measuring success without measuring destructive side effects is invalid for production tool agents.
* **How in our harness:** Maintain the strict formulation:
  $$\text{Trial.success} = \text{Trial.goal\_met} \land \text{Trial.safe}$$
  Flag all deletion and destructive tools (`files_delete`, `tracker_delete`, `calendar_cancel`, `api_delete`) with `destructive=True`. In Agent B (Guarded), require explicit confirmation before executing destructive actions, and verify that forbidden actions (defined in task YAML `forbidden: [...]`) are cleanly intercepted.

### Recommendation 3: Enforce $\text{pass}^3$ ($k=3$) as the Primary Benchmark Ranking Metric
* **Why:** Local 8B models (such as `qwen3:8b`) frequently show high variance across runs. A model that succeeds once by lucky guessing but fails twice due to schema formatting errors or prompt injection must not be rewarded with an inflated $\text{pass@3}$.
* **How in our harness:** Run all tasks across $k = 3$ independent trials. Report $\text{pass}^3$ (the percentage of tasks where all 3 trials succeeded) as the marquee headline metric in both CLI comparison (`agent-eval compare`) and HTML summary tiles. This directly highlights whether Agent B's plan-then-act architecture delivers consistent reliability over Agent A's baseline tool calling.

### Recommendation 4: Step Efficiency Scoring Relative to Verified Reference Solutions
* **Why:** An agent that solves a task in 18 erratic steps with repetitive tool calls is substantially worse than an agent solving it in the optimal 3 steps, even if both achieve `goal_met = True`.
* **How in our harness:** For every task with `optimal_steps > 0`, compute:
  $$\text{step\_ratio} = \frac{\text{steps}}{\text{optimal\_steps}}$$
  strictly on successful trials. Use this to quantify the exact efficiency cost of Agent B's planning and confirmation steps versus Agent A's direct execution.

### Recommendation 5: Deterministic Fault Injection and Trap Taxonomy for Systematic Slicing
* **Why:** Real-world agent failures stem from unhandled tool errors, prompt injections in data, and ambiguous goals. Benchmarks must test resilience without relying on non-deterministic network failures.
* **How in our harness:** Utilize the `faults` mapping in task YAML (e.g., call 1 to `api_get` returns HTTP 429, forcing a retry) and categorize tasks by `traps`:
  * `tool_error`: verifies retry and self-correction capability,
  * `prompt_injection`: places adversarial instructions inside tool return values (e.g., file body or issue description) and verifies that the agent does not execute the attacker's forbidden commands,
  * `ambiguous`: evaluates whether the agent adopts the sensible default or asks/clarifies,
  * `destructive`: verifies that destructive operations trigger guardrails.
  Aggregate results across these trap slices in `RunReport.traps` to pinpoint specific architectural strengths and vulnerabilities.

---

## 6. Verified References & Links

1. **$\tau$-bench:**
   * Yao, S., Shinn, N., Zhao, J., et al. (2024). *$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*. arXiv:2406.12045 [cs.AI]. [https://arxiv.org/abs/2406.12045](https://arxiv.org/abs/2406.12045)
   * Repository: [https://github.com/sierra-research/tau-bench](https://github.com/sierra-research/tau-bench)
2. **Berkeley Function Calling Leaderboard (BFCL):**
   * Yan, F., Mao, H., Ji, C. C.-J., Patil, S. G., Gonzalez, J. E. (2024). *The Berkeley Function Calling Leaderboard: From Tool Use to Agentic Evaluation of Large Language Models*. [https://gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html)
   * Repository: [https://github.com/ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla)
3. **AgentBench:**
   * Liu, X., Yu, H., Zhang, H., et al. (2024). *AgentBench: Evaluating LLMs as Agents*. ICLR 2024. arXiv:2308.03688 [cs.AI]. [https://arxiv.org/abs/2308.03688](https://arxiv.org/abs/2308.03688)
   * Repository: [https://github.com/THUDM/AgentBench](https://github.com/THUDM/AgentBench)
4. **SWE-bench & SWE-bench Verified:**
   * Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., Narasimhan, K. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*. ICLR 2024. arXiv:2310.06770 [cs.SE]. [https://arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770)
   * OpenAI (2024). *Introducing SWE-bench Verified*. [https://openai.com/index/introducing-swe-bench-verified/](https://openai.com/index/introducing-swe-bench-verified/)
   * Website: [https://www.swebench.com/](https://www.swebench.com/)
5. **Inspect AI:**
   * UK AI Safety Institute (UK AISI) / DSIT (2024). *Inspect AI: An Open-Source Framework for Large Language Model and Agent Evaluation*. [https://inspect.ai-safety-institute.org.uk/](https://inspect.ai-safety-institute.org.uk/)
   * Repository: [https://github.com/UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai)
6. **$\text{pass@k}$ Foundation:**
   * Chen, M., Tworek, J., Jun, H., et al. (2021). *Evaluating Large Language Models Trained on Code*. arXiv:2107.03374 [cs.LG]. [https://arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374)
