# Agentic Informer Forecasting Workflow



## 1. Goal



The goal of this experiment is to compare a manually configured Informer forecasting model against an agent-assisted Informer tuning workflow for influenza hospitalization forecasting.



The agent does not generate forecasts directly. Instead, it acts as an experiment-orchestration and refinement layer around a deterministic NeuralForecast + Optuna pipeline.



The basic workflow is:



**agent proposes search space → Optuna trains/evaluates candidate models → deterministic validation metrics are computed → best configuration is selected → agent reviews validation results → agent proposes a refined search → refined candidates are evaluated**



The final forecasting model remains an Informer throughout the experiment.



---



## 2. Forecasting task



Target:



- Weekly influenza hospitalization rate (`weekly_rate`)



Forecast horizons:



- 1 week ahead

- 2 weeks ahead

- 3 weeks ahead

- 4 weeks ahead



Primary model:



- NeuralForecast Informer



Exogenous feature:



- Emergency-department influenza visit proportion



Because the installed Informer implementation does not support historical exogenous variables directly, ED visits are included as a leakage-safe four-week-lagged future exogenous variable.



For target week \\(t\\), the model receives:



`ED(t - 4)`



This ensures that for forecast horizons 1 through 4, the ED value associated with each target week would already have been observable at the forecast origin.



---



## 3. Manual Informer workflow



The manually selected Informer configuration used:



- input size: 12 weeks

- hidden size: 64

- attention heads: 4

- encoder layers: 2

- decoder layers: 1

- dropout: 0.05

- learning rate: 0.0001

- batch size: 32

- maximum training steps: 1000

- random seed: 2

- standard scaling



This configuration was selected manually based on prior experiments rather than by an automated search over the full Informer architecture space.



The model was trained on historical observations and evaluated using rolling forecasts with model weights fixed during evaluation (`refit=False`).



---



## 4. Agentic round 1



### 4.1 Agent role



The configuration agent receives a natural-language description of the forecasting task and a deterministic schema defining which Informer hyperparameters may be searched.



The agent outputs structured JSON describing:



- experiment metadata

- Optuna search-space definitions



The output is validated deterministically before it is converted into executable Python.



The LLM does not directly execute arbitrary model code.



### 4.2 Round-1 search space



The initial agent-generated search included:



- input size: 8, 12, 16, or 24

- hidden size: 32, 64, or 128

- convolution hidden size: 32, 64, or 128

- attention heads: 2, 4, or 8

- encoder layers: 1–3

- decoder layers: 1–2

- dropout: 0.0–0.2

- learning rate: 0.00005–0.005 on a log scale

- maximum steps: 500 or 1000

- batch size: 16 or 32

- random seed: initially searched from 1–10

- scaler: fixed to standard



Optuna evaluated eight candidate configurations.



### 4.3 Validation objective



Each Optuna candidate was evaluated using a 12-week validation window at the end of the pre-test training data.



For each configuration, rolling forecasts were generated for horizons 1 through 4.



The objective supplied to Optuna was:



**mean validation RMSE across horizons 1–4**



Each horizon therefore received equal weight during model selection.



### 4.4 Round-1 selected configuration



The first agentic search selected:



- input size: 8

- hidden size: 32

- convolution hidden size: 128

- attention heads: 4

- encoder layers: 1

- decoder layers: 2

- dropout: approximately 0.150

- learning rate: approximately 0.00198

- maximum steps: 500

- batch size: 16

- random seed: 4

- scaler: standard



This was substantially different from the manually configured Informer.



In particular, the agent-selected configuration used:



- a shorter observation window,

- a smaller hidden representation,

- fewer encoder layers,

- more decoder layers,

- substantially more dropout,

- a much higher learning rate,

- a smaller batch size,

- and half as many training steps.



---



## 5. Held-out evaluation



After the best configuration was selected using validation data, the configuration was frozen.



The manual and agent-selected models were then evaluated on the held-out 2025–26 forecasting period.



The held-out results were not used by the configuration agent during round-1 selection.



### Manual vs round-1 agentic Informer



| Horizon | Manual RMSE | Agent RMSE | RMSE reduction |

|---|---:|---:|---:|

| 1 week | 2.0679 | 1.1593 | 43.9% |

| 2 weeks | 3.9799 | 2.0644 | 48.1% |

| 3 weeks | 7.0193 | 3.0561 | 56.5% |

| 4 weeks | 7.3947 | 5.0476 | 31.7% |



Mean RMSE across horizons:



- Manual Informer: 5.1154

- Agent-selected Informer: 2.8319

- Relative reduction: 44.6%



The first agentic search therefore produced lower held-out RMSE at all four forecast horizons.



---



## 6. Agentic refinement round



A second refinement agent was added so that the workflow would contain an explicit feedback step rather than stopping after a single Optuna search.



The refinement agent received:



- the round-1 winning configuration,

- the validation RMSE for every completed round-1 Optuna trial,

- the horizon-specific validation RMSE values,

- and the original search space.



It did **not** receive held-out 2025–26 test performance.



This separation was intended to prevent the agent from adapting its search directly to the final evaluation period.



The refinement agent then proposed a narrower search space around the region that performed well during round 1.



The random seed was fixed to 4 during the refined search so that seed selection would not act as another tuning dimension.



---



## 7. Round-2 result



The refined search selected:



- input size: 8

- hidden size: 64

- convolution hidden size: 128

- attention heads: 2

- encoder layers: 1

- decoder layers: 1

- dropout: approximately 0.084

- learning rate: approximately 0.000718

- maximum steps: 500

- batch size: 16

- random seed: 4

- scaler: standard



Validation mean RMSE improved slightly from approximately:



- round 1: 0.0436

- round 2: 0.0422



However, the refined configuration generalized worse on the held-out forecasting period.



Mean held-out RMSE:



- Manual Informer: approximately 5.11

- Round-1 agentic Informer: approximately 2.83

- Round-2 refined Informer: approximately 3.92



The round-2 model still outperformed the manual configuration, but it did not outperform the first agentic winner.



---



## 8. Current interpretation



The results indicate that the agentic framework can identify Informer configurations that substantially outperform the manually selected configuration.



However, the second refinement round reveals an important limitation in the current validation design.



The existing 12-week validation window occurs mainly during July through September, when influenza hospitalization activity is relatively low.



This creates a mismatch between:



- the environment used to choose and refine hyperparameters, and

- the active-season forecasting conditions that dominate the held-out evaluation period.



A model can therefore improve validation RMSE without improving active-season generalization.



This suggests that the main bottleneck is no longer simply the hyperparameter search space. The validation procedure itself needs to better represent the forecasting conditions of interest.



---



## 9. Separation of responsibilities



### LLM agent



The agent is responsible for:



- interpreting the forecasting task,

- proposing a valid hyperparameter search space,

- reviewing deterministic validation results,

- identifying promising or weak regions of the search,

- and proposing a refined search space.



### Optuna



Optuna is responsible for:



- sampling concrete configurations from the approved search space,

- running candidate experiments,

- tracking validation performance,

- and identifying the configuration with the lowest validation objective.



### Deterministic Python pipeline



The forecasting pipeline is responsible for:



- loading and preprocessing the data,

- constructing leakage-safe ED features,

- training Informer models,

- generating forecasts,

- calculating MAE and RMSE,

- enforcing train/validation/test separation,

- validating agent-generated configuration schemas,

- and saving experiment outputs.



The LLM is therefore an orchestration and refinement component rather than the numerical forecasting model itself.



---



## 10. Proposed next validation experiment



The current temporal validation window will be replaced or supplemented with a state-based validation procedure.



A proposed design is:



1\. Use approximately 10 states as a validation subset.

2\. Keep the remaining jurisdictions separate for evaluation.

3\. Shuffle the state assignment.

4\. Repeat the procedure approximately five times.

5\. Aggregate validation performance across the five state splits.

6\. Allow the refinement agent to observe only validation-fold results.

7\. Preserve a final evaluation set that is not used to guide agent refinement.



This experiment will test whether the agentic tuning result remains robust when validation contains realistic active-season trajectories rather than primarily low-activity summer observations.



---



## 11. Current conclusion



The first agentic Informer experiment demonstrates that automated search-space generation and model selection can materially improve forecast performance relative to the manually selected configuration.



The refinement experiment also demonstrates why the evaluation architecture surrounding the agent matters.



A stronger agentic forecasting framework therefore requires both:



- effective model/configuration search, and

- a validation procedure that accurately represents the deployment conditions the model will face.

