# Agentic Time-Series Analysis Prototype

## Overview

This project is a small local prototype of a structured agentic workflow for time-series analysis.

The system uses two different LLMs with separate roles:

* **GPT** acts as a data retrieval/planning agent.
* **Claude** acts as a time-series analysis and feedback agent.

The workflow retrieves real financial time-series data, analyzes it, determines whether additional historical context is needed, and can automatically perform a second retrieval and revised analysis.

The prototype is inspired by *Structured Agentic Workflows for Financial Time-Series Modeling with LLMs and Reflective Feedback*.

The goal is not to reproduce the full TS-Agent framework. Instead, this implementation demonstrates several of its core ideas in a smaller and more understandable system:

* specialized agent roles
* external tool use
* structured agent-to-agent communication
* iterative feedback
* stateful workflow execution
* logging and traceability
* modular architecture

\---

## Architecture

```text
User Task
    |
    v
GPT Retriever Agent
    |
    | structured retrieval request
    v
Financial Data Tool (yfinance)
    |
    | time-series observations
    v
Claude Analyst Agent
    |
    | needs more data?
    |
    +---- No ----> Final Analysis
    |
    +---- Yes
            |
            v
      Financial Data Tool
            |
            v
      Expanded Dataset
            |
            v
      Claude Second-Pass Analysis
            |
            v
       Final Analysis

All workflow decisions and results are saved to JSON execution logs.
```

\---

## Current Workflow

### 1\. User Task

The prototype currently begins with a natural-language request such as:

```text
Analyze NVIDIA's recent stock-price behavior.
Determine whether the recent movement looks unusual
and whether more historical context would be useful.
```

### 2\. Retriever Agent

The GPT-based Retriever Agent determines:

* the stock ticker
* an appropriate initial historical period
* why that data is needed

Example:

```json
{
  "symbol": "NVDA",
  "period": "3mo",
  "reason": "Three months provides enough recent context to assess the current movement."
}
```

### 3\. Data Retrieval Tool

A Python tool uses `yfinance` to retrieve daily closing-price observations.

The LLM does not generate the financial data itself. Instead:

1. the Retriever Agent decides what data is needed
2. Python executes the retrieval
3. the resulting time-series data is passed to the Analyst Agent

### 4\. Analyst Agent

Claude receives:

* the original user task
* the Retriever Agent's structured request
* the retrieved time-series observations

It returns a structured analysis including:

* overall trend
* unusual movements
* whether more historical data is required
* the requested expanded time period
* the reasoning behind the request

### 5\. Feedback Loop

If Claude determines that the initial dataset is insufficient, the orchestrator automatically performs another data retrieval using the longer requested period.

The expanded dataset is then returned to Claude for a second-pass analysis.

This allows the final analysis to incorporate information discovered during the first pass rather than relying on a fixed one-shot pipeline.

### 6\. Execution Logging

Each workflow run is saved as a timestamped JSON file in the `logs/` directory.

The log records:

* user task
* model assignments
* Retriever Agent output
* first retrieval metadata
* first analysis
* feedback decision
* second retrieval metadata
* final analysis
* workflow status
* errors, if any

Failed runs are logged as well.

\---

## Project Structure

```text
agentic-timeseries/
|
├── main.py
├── agents.py
├── tools.py
├── logger.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
├── logs/
└── .venv/
```

### `main.py`

Acts as the local orchestrator.

It:

* controls the order of execution
* passes information between components
* manages the feedback loop
* maintains workflow state
* records final results and errors

### `agents.py`

Contains the LLM-based agents.

Current model assignments:

```text
Retriever Agent: GPT
Analyst Agent: Claude
```

It also handles:

* structured JSON parsing
* Markdown code-fence cleanup
* basic model-response errors
* API-call failures

### `tools.py`

Contains external tools available to the workflow.

Currently:

```text
fetch\_stock\_data()
```

This function retrieves financial time-series data using `yfinance`.

### `logger.py`

Saves execution traces to timestamped JSON files in the `logs/` directory.

\---

## Setup

Create and activate a Python virtual environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file containing:

```text
OPENAI\_API\_KEY=your\_openai\_api\_key
ANTHROPIC\_API\_KEY=your\_anthropic\_api\_key
```

API keys should never be committed to version control.

Run the prototype:

```bash
python main.py
```

\---

## Relationship to TS-Agent

The prototype implements a simplified subset of concepts from the TS-Agent paper.

|TS-Agent concept|Current prototype|
|-|-|
|Structured workflow|Local Python orchestrator|
|Planner/model-selection logic|GPT Retriever Agent|
|External resources|Financial data tool|
|Execution feedback|Claude can request expanded data|
|Iterative refinement|Second-pass analysis|
|Memory/context|Workflow state passed between stages|
|Auditability|JSON execution logs|
|Modular architecture|Separate agents, tools, logging, and orchestration|

Several major TS-Agent components are **not yet implemented**, including:

* Case Bank
* Financial Time-Series Code Base
* Refinement Knowledge Bank
* automated forecasting-model selection
* automated code refinement
* hyperparameter optimization
* execution-based model training
* multiple iterative refinement cycles

\---

## Current Limitations

This is an early prototype.

Current limitations include:

* only financial stock-price data is supported
* the tool currently retrieves closing prices only
* there is no statistical forecasting model yet
* the analysis is primarily LLM-based
* feedback is limited to requesting a larger historical window
* only one optional refinement pass is performed
* agent outputs can vary between runs
* model reasoning may not always be numerically or statistically rigorous
* financial conclusions are demonstrations of workflow behavior, not investment advice

\---

## Possible Next Steps

Possible extensions include:

* supporting multiple time-series datasets
* adding economic, epidemiological, or environmental data sources
* allowing multiple feedback iterations
* adding quantitative statistical features before LLM analysis
* comparing homogeneous and heterogeneous agent configurations
* testing different Retriever/Analyst model combinations
* introducing forecasting models
* evaluating whether feedback improves analysis quality
* measuring consistency and failure rates across repeated runs
* adding a model or knowledge bank inspired by TS-Agent
* investigating how agent specialization affects time-series reasoning

\---

## Research Direction

The current implementation is intended as a starting point for exploring structured multi-agent workflows for time-series analysis.

Longer-term research questions could include:

* Does specialization between LLM agents improve performance?
* Do heterogeneous model combinations behave differently from single-model systems?
* When does reflective feedback improve time-series analysis?
* How should agents decide when additional data is necessary?
* How reliable are agentic workflows across repeated runs?
* What forms of memory or structured feedback produce the most useful refinement?

