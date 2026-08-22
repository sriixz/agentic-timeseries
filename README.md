\# Agentic Time-Series Analysis Prototype



\## Overview



This project is a small local prototype of a structured agentic workflow for time-series analysis.



The system uses two different LLMs with separate roles:



\* \*\*GPT\*\* acts as a data retrieval/planning agent.

\* \*\*Claude\*\* acts as a time-series analysis and feedback agent.



The workflow retrieves real financial time-series data, analyzes it, determines whether additional historical context is needed, and can automatically perform a second retrieval and revised analysis.



The prototype is inspired by the structured agentic workflow described in \*Structured Agentic Workflows for Financial Time-Series Modeling with LLMs and Reflective Feedback\*.



The goal of this implementation is not to reproduce the full TS-Agent framework. Instead, it demonstrates several of its core ideas in a small and understandable system:



\* specialized agent roles

\* external tool use

\* structured agent-to-agent communication

\* iterative feedback

\* stateful workflow execution

\* logging and traceability

\* modular architecture



\## Architecture



```text

User Task

&#x20;   |

&#x20;   v

GPT Retriever Agent

&#x20;   |

&#x20;   | structured retrieval request

&#x20;   v

Financial Data Tool

(yfinance)

&#x20;   |

&#x20;   | time-series observations

&#x20;   v

Claude Analyst Agent

&#x20;   |

&#x20;   | needs more data?

&#x20;   |

&#x20;   +---- No ----> Final Analysis

&#x20;   |

&#x20;   +---- Yes

&#x20;           |

&#x20;           v

&#x20;     Financial Data Tool

&#x20;           |

&#x20;           v

&#x20;     Expanded Dataset

&#x20;           |

&#x20;           v

&#x20;     Claude Second-Pass Analysis

&#x20;           |

&#x20;           v

&#x20;      Final Analysis



All workflow decisions and results

are saved to JSON execution logs.

```



\## Current Workflow



\### 1. User task



The prototype currently begins with a natural-language request such as:



```text

Analyze NVIDIA's recent stock-price behavior.

Determine whether the recent movement looks unusual

and whether more historical context would be useful.

```



\### 2. Retriever Agent



The GPT-based Retriever Agent determines:



\* the stock ticker

\* an appropriate initial historical period

\* why that data is needed



Example:



```json

{

&#x20; "symbol": "NVDA",

&#x20; "period": "3mo",

&#x20; "reason": "Three months provides enough recent context..."

}

```



\### 3. Data Retrieval Tool



A Python tool uses `yfinance` to retrieve daily closing-price observations.



The LLM does not generate the financial data itself. It decides what data is needed, and Python executes the retrieval.



\### 4. Analyst Agent



Claude receives:



\* the original user task

\* the Retriever Agent's request

\* the retrieved time-series observations



It returns a structured analysis including:



\* overall trend

\* unusual movements

\* whether more historical data is required

\* the requested expanded time period

\* the reasoning behind the request



\### 5. Feedback Loop



If Claude determines that the initial dataset is insufficient, the orchestrator automatically performs another data retrieval.



The expanded dataset is then returned to Claude for a second-pass analysis.



This allows the later analysis to incorporate information discovered during the first pass.



\### 6. Execution Logging



Each workflow run is saved as a timestamped JSON file in the `logs/` directory.



The log records:



\* user task

\* model assignments

\* Retriever Agent output

\* retrieval metadata

\* first analysis

\* feedback decision

\* second retrieval metadata

\* final analysis

\* workflow status

\* errors, if any



Failed runs are also logged.



\## Project Structure



```text

agentic-timeseries/

|

├── main.py

├── agents.py

├── tools.py

├── logger.py

├── requirements.txt

├── .env

├── logs/

└── .venv/

```



\### `main.py`



Acts as the local orchestrator.



It controls the order of execution, passes information between components, manages the feedback loop, and records workflow state.



\### `agents.py`



Contains the LLM-based agents.



Current model assignments:



```text

Retriever Agent: GPT

Analyst Agent: Claude

```



It also handles structured JSON parsing and basic model-response errors.



\### `tools.py`



Contains external tools available to the workflow.



Currently:



```text

fetch\_stock\_data()

```



retrieves financial time-series data using `yfinance`.



\### `logger.py`



Saves execution traces to JSON files in the `logs/` directory.



\## Setup



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



\## Relationship to TS-Agent



The prototype implements a simplified subset of concepts from the TS-Agent paper.



| TS-Agent concept              | Current prototype                              |

| ----------------------------- | ---------------------------------------------- |

| Structured workflow           | Local Python orchestrator                      |

| Planner/model-selection logic | GPT Retriever Agent                            |

| External resources            | Financial data tool                            |

| Execution feedback            | Claude can request expanded data               |

| Iterative refinement          | Second-pass analysis                           |

| Memory/context                | Workflow state passed between stages           |

| Auditability                  | JSON execution logs                            |

| Modular architecture          | Separate agents, tools, logging, orchestration |



Several major TS-Agent components are \*\*not yet implemented\*\*, including:



\* Case Bank

\* Financial Time-Series Code Base

\* Refinement Knowledge Bank

\* automated forecasting-model selection

\* automated code refinement

\* hyperparameter optimization

\* execution-based model training

\* multiple iterative refinement cycles



\## Current Limitations



This is an early prototype.



Current limitations include:



\* only financial stock-price data is supported

\* the tool currently retrieves closing prices only

\* there is no statistical forecasting model yet

\* analysis is primarily LLM-based

\* feedback is limited to requesting a larger historical window

\* only one optional refinement pass is performed

\* agent outputs depend on model reasoning and may vary between runs

\* financial conclusions are demonstrations of workflow behavior, not investment advice



\## Possible Next Steps



Possible extensions include:



\* supporting multiple time-series datasets

\* adding economic, epidemiological, or environmental data sources

\* allowing multiple feedback iterations

\* adding quantitative statistical features before LLM analysis

\* comparing homogeneous and heterogeneous agent configurations

\* testing different Retriever/Analyst model combinations

\* introducing forecasting models

\* evaluating whether feedback improves analysis quality

\* measuring consistency and failure rates across repeated runs

\* adding a model or knowledge bank inspired by TS-Agent

\* investigating how agent specialization affects time-series reasoning



\## Research Direction



The current implementation is intended as a starting point for exploring structured multi-agent workflows for time-series analysis.



A longer-term research direction could investigate questions such as:



\* Does specialization between LLM agents improve performance?

\* Do heterogeneous model combinations behave differently from single-model systems?

\* When does reflective feedback improve time-series analysis?

\* How should agents decide when additional data is necessary?

\* How reliable are agentic workflows across repeated runs?

\* What forms of memory or structured feedback produce the most useful refinement?



