# LLM Prompt / Code Generation Log

This file documents the main prompts and instructions used to generate and refine the code in this repository.

The prompts below are reconstructed summaries rather than verbatim chat transcripts. They capture the main intent, constraints, and requested functionality for each file.

---

## `main.py`

### Purpose
Orchestrate the initial stock-analysis workflow using Apple (`AAPL`) as the example.

### Representative prompt
> Create a simple local multi-agent time-series analysis prototype.  
> Use GPT as a retrieval/planning agent to decide what stock data is needed, retrieve the data through a Python tool using `yfinance`, and pass the results to Claude for analysis.  
> If Claude requests more historical context, automatically retrieve the larger period and perform a second analysis.  
> Keep the orchestration simple and log the workflow.

---

## `main_nvda.py`

### Purpose
Provide a second financial demonstration using NVIDIA.

### Representative prompt
> Create a second version of the stock workflow using NVIDIA (`NVDA`) so that the same agentic workflow can be demonstrated on another stock without changing the original Apple example.

---

## `agents.py`

### Purpose
Define the GPT and Claude agents used by the financial and FluSight workflows.

### Initial financial-agent prompt
> Implement separate agent functions:
>
> - a GPT retriever/planner that returns a structured request containing the stock ticker, historical period, and reason for retrieving that data
> - a Claude analyst that receives the user task and retrieved observations and returns a structured analysis
> - allow the analyst to request additional historical data
> - add a second-pass analyst for the expanded dataset
>
> Require structured JSON-like outputs so Python can control the workflow.

### FluSight extension prompt
> Extend the agent module for CDC FluSight analysis.
>
> Add a GPT planner that selects one of:
>
> - `latest_season`
> - `cross_season`
> - `all_seasons`
>
> Then have Claude analyze only the deterministic summary produced for that scope.
>
> Claude may request additional detail for one valid season, followed by a second-pass analysis.

### Validation / repair prompt
> Add a semantic and structural validation layer after Claude produces a FluSight analysis.
>
> Detect unsupported claims such as:
>
> - unsupported percentages
> - unsupported causal or mechanistic explanations
> - geographic claims not supported by the supplied summary
> - structural output violations
>
> If violations are found, give Claude the original output, authoritative Python-generated facts, and the exact violations and allow one constrained repair attempt.
>
> Validate the repaired output again before continuing.

---

## `tools.py`

### Purpose
Provide deterministic data retrieval and preprocessing functions.

### Financial-tool prompt
> Create a Python data tool that uses `yfinance` to retrieve historical closing-price data for a requested ticker and time period.  
> Keep data retrieval separate from the LLM reasoning.

### FluSight-tool prompt
> Add deterministic preprocessing functions for the CDC FluSight hospitalization dataset.
>
> The dataset contains:
>
> - `date`
> - `location`
> - `location_name`
> - `value`
> - `weekly_rate`
>
> Compute dataset metadata, national seasonal peaks, jurisdiction peak dates, dominant peak timing, and early/typical/late classifications.
>
> Do not send the entire raw dataset to the LLM. Compute structured summaries in Python first.

### Trend-analysis extension prompt
> Add functions that summarize national hospitalization trends using the `value` column as the primary measure.
>
> Include:
>
> - seasonal peak counts and dates
> - latest available observation
> - week-over-week change
> - difference from the observed seasonal peak
> - comparison of the latest available season's peak with past seasons
>
> Keep `weekly_rate` available as a secondary normalized measure.

---

## `logger.py`

### Purpose
Save reproducible execution traces.

### Representative prompt
> Create a simple JSON logger that saves each workflow execution with a timestamp.
>
> Include major workflow decisions, agent outputs, retrieved data summaries, feedback decisions, final output, generated plots, and errors.

---

## `main_flu.py`

### Purpose
Orchestrate the complete FluSight agentic workflow.

### Representative prompt
> Build a FluSight orchestration script with the following flow:
>
> 1. load deterministic dataset metadata
> 2. ask GPT to select an analysis scope
> 3. generate only the Python summary required for that scope
> 4. send the summary to Claude
> 5. validate the response and repair it once if needed
> 6. optionally retrieve detailed information for one season
> 7. perform a second-pass Claude analysis
> 8. generate plots
> 9. save the complete run to a JSON log
>
> Avoid passing all raw FluSight rows directly to the LLM.

---

## `test_flu.py`

### Purpose
Test deterministic FluSight preprocessing without using LLM API calls.

### Representative prompt
> Create a local test script for the deterministic FluSight functions.
>
> Print dataset metadata, season summaries, national peaks, and timing statistics so the Python calculations can be checked independently of the agents.

---

## `test_flu_scopes.py`

### Purpose
Test planner-controlled analytical scopes.

### Representative prompt
> Create tests using representative natural-language tasks for:
>
> - latest-season analysis
> - cross-season analysis
> - all-season analysis
>
> Verify that the planner selects the correct scope and that the downstream workflow obeys that scope.

---

## `evaluate_flu.py`

### Purpose
Evaluate structural/control behavior of the agentic workflow.

### Representative prompt
> Build a small evaluation harness for the FluSight workflow.
>
> Evaluate several representative tasks and check:
>
> - planner scope accuracy
> - feedback scope compliance
> - required output keys
> - output-length compliance
> - successful workflow completion
>
> Keep this separate from full scientific/natural-language factual evaluation.

---

## `visualizations.py`

### Purpose
Generate deterministic FluSight visualizations.

### Representative prompt
> Create Matplotlib visualizations from Python-computed FluSight data.
>
> Include:
>
> - national weekly hospitalization rate over time
> - early/typical/late jurisdiction counts across seasons
> - jurisdiction peak dates for a selected season
>
> Generate the plots deterministically without asking an LLM to create or interpret the figure.

---

## `flu_trend_analysis.py`

### Purpose
Analyze national hospitalization-count trends.

### Representative prompt
> Create a standalone deterministic analysis script focused on the FluSight `value` column because it represents actual hospitalization counts.
>
> Compare recent flu seasons using national observations.
>
> For each season calculate:
>
> - peak hospitalization count
> - peak date
> - latest value
> - previous week's value
> - week-over-week absolute and percentage change
> - change from the observed seasonal peak
>
> Compare the latest available season's peak against previous complete seasons.
>
> Generate a multi-season hospitalization-count plot and save CSV summaries.
>
> Do not use an LLM for this analysis or plot generation.

---

## `lstm_forecast.py`

### Purpose
Create a manually configured NeuralForecast LSTM baseline.

### Representative prompt
> Build a standalone NeuralForecast LSTM experiment using U.S. national FluSight `weekly_rate`.
>
> Use:
>
> - training data through September 2025
> - held-out test data from October 2025 through May 2026
> - 4-week forecast horizon
> - 12-week input window
> - standard scaling
> - hidden size 64
> - 300 training steps
>
> Fit the model once before the test period.
>
> Evaluate rolling 1-, 2-, 3-, and 4-week forecasts with `refit=False` so model weights remain fixed during the held-out period.
>
> Report MAE and RMSE separately by forecast horizon and generate plots.

---

## `autolstm_forecast.py`

### Purpose
Automatically tune an LSTM using NeuralForecast AutoLSTM and Optuna.

### Representative prompt
> Add an AutoLSTM experiment that is directly comparable with the manual LSTM baseline.
>
> Keep:
>
> - the same `weekly_rate` target
> - the same training cutoff
> - the same held-out test period
> - the same 4-week forecasting horizon
>
> Use NeuralForecast `AutoLSTM` with `backend="optuna"`.
>
> Use only pre-test observations for hyperparameter selection and reserve a temporal validation window inside the training data.
>
> Because the time series is relatively short, constrain the input-size search to feasible values.
>
> Tune:
>
> - `input_size`
> - `encoder_hidden_size`
> - `encoder_n_layers`
> - `context_size`
> - `decoder_hidden_size`
> - `learning_rate`
> - `max_steps`
> - `batch_size`
> - `random_seed`
>
> Run a small initial search of five Optuna trials.
>
> After selecting the best configuration, instantiate a standard LSTM with those parameters, fit it once on pre-test data, and evaluate it with fixed weights using `refit=False`.
>
> Save trial results, the best configuration, forecasts, metrics, and plots.

---

## `compare_lstm_models.py`

### Purpose
Compare the manual LSTM and AutoLSTM results.

### Representative prompt
> Create a deterministic comparison script that loads the saved manual-LSTM and AutoLSTM metric CSV files.
>
> Merge them by forecast horizon and compare:
>
> - MAE
> - RMSE
> - percentage improvement or degradation from the manual model to AutoLSTM
>
> Generate separate MAE and RMSE comparison plots.
>
> Do not use an LLM or agent for this comparison.

---

## Notes on LLM usage

LLMs were primarily used as software-development assistants to generate and iteratively refine code based on the specifications above.

The repository contains two different types of LLM usage:

1. **Development-time LLM assistance**
   - generating Python implementations
   - debugging errors
   - modifying architecture
   - writing tests and documentation

2. **Runtime LLM agents**
   - GPT planner/retriever
   - Claude analyst
   - Claude repair/second-pass analysis

Deterministic numerical analysis, model training, forecast evaluation, plotting, and metric calculations are executed by Python rather than generated by the runtime LLM agents.
