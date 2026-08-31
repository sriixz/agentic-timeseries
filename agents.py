import json

from anthropic import Anthropic
from openai import OpenAI


openai_client = OpenAI()
claude_client = Anthropic()


def parse_json_response(response_text):
    cleaned = response_text.strip()

    # Remove Markdown code fences like ```json ... ```
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]

    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]

    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError as error:
        raise ValueError(
            "Model returned invalid JSON.\n\n"
            f"Raw response:\n{response_text}"
        ) from error


# -------------------------------------------------------------------
# STOCK WORKFLOW AGENTS
# -------------------------------------------------------------------

def run_retriever_agent(user_task):
    retriever_prompt = f"""
You are a data retrieval agent.

Your job is to decide what stock-price data is needed
to answer the user's task.

User task:
{user_task}

Return ONLY valid JSON in this exact format:

{{
  "symbol": "stock ticker",
  "period": "one of: 5d, 1mo, 3mo, 6mo, 1y",
  "reason": "brief explanation of why this data is needed"
}}
"""

    try:
        response = openai_client.responses.create(
            model="gpt-5.4-mini",
            input=retriever_prompt,
        )

        return parse_json_response(response.output_text)

    except Exception as error:
        raise RuntimeError(
            f"Retriever agent failed: {error}"
        ) from error


def run_analyst_agent(user_task, retrieval_request, data):
    analyst_prompt = f"""
You are a time-series analysis agent.

User task:
{user_task}

The retrieval agent requested:

{json.dumps(retrieval_request, indent=2)}

Retrieved data:

{json.dumps(data, indent=2)}

Analyze the data.

Return ONLY valid JSON in this exact format:

{{
  "summary": "short summary of the overall trend",
  "unusual_movement": "description of any unusual movement",
  "needs_more_data": true,
  "requested_period": "one of: 5d, 1mo, 3mo, 6mo, 1y",
  "reason": "why more data is or is not needed"
}}

Rules:
- If more historical context would materially improve the analysis,
  set "needs_more_data" to true.
- If the current data is sufficient, set it to false.
- If needs_more_data is false, requested_period should stay the same
  as the current period.
"""

    try:
        message = claude_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1200,
            messages=[
                {
                    "role": "user",
                    "content": analyst_prompt,
                }
            ],
        )

        response_text = message.content[0].text

        return parse_json_response(response_text)

    except Exception as error:
        raise RuntimeError(
            f"Analyst agent failed: {error}"
        ) from error


def run_second_pass_analyst(
    user_task,
    first_analysis,
    expanded_data,
):
    second_analyst_prompt = f"""
You are a time-series analysis agent.

User task:
{user_task}

Your first analysis was:

{json.dumps(first_analysis, indent=2)}

You requested additional historical context.

Here is the expanded dataset:

{json.dumps(expanded_data, indent=2)}

Now revise your analysis using the larger dataset.

Return ONLY valid JSON in this exact format:

{{
  "summary": "revised summary",
  "unusual_movement": "revised assessment of unusual movement",
  "needs_more_data": false,
  "reason": "explain what the expanded history changed or clarified"
}}
"""

    try:
        message = claude_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1200,
            messages=[
                {
                    "role": "user",
                    "content": second_analyst_prompt,
                }
            ],
        )

        response_text = message.content[0].text

        return parse_json_response(response_text)

    except Exception as error:
        raise RuntimeError(
            f"Second-pass analyst failed: {error}"
        ) from error


# -------------------------------------------------------------------
# FLUSIGHT WORKFLOW AGENTS
# -------------------------------------------------------------------

def run_flu_planner_agent(user_task, dataset_summary):
    """
    GPT decides how the FluSight dataset should be analyzed.
    """

    planner_prompt = f"""
You are a planning agent for an epidemiological time-series analysis workflow.

The user wants to analyze CDC FluSight influenza hospitalization data.

USER TASK:
{user_task}

DATASET METADATA:
{json.dumps(dataset_summary, indent=2)}

Your job is NOT to perform the epidemiological analysis itself.

Instead, decide what analytical scope is most appropriate.

Available analysis scopes:

- "all_seasons"
- "latest_season"
- "cross_season"

Scope definitions:

- "latest_season":
  Use when the task explicitly focuses on the most recent season.

- "cross_season":
  Use when the task asks how patterns change across seasons or asks for
  comparisons across time.

- "all_seasons":
  Use when the task explicitly asks for detailed analysis of every season.

Consider:
- whether the task requires multiple seasons
- whether national seasonality matters
- whether comparisons across jurisdictions matter
- whether partial seasons should be treated cautiously

Return ONLY valid JSON with exactly this structure:

{{
  "analysis_scope": "one of: all_seasons, latest_season, cross_season",
  "focus": [
    "one or more short analytical goals"
  ],
  "reason": "brief explanation of why this scope is appropriate"
}}
"""

    try:
        response = openai_client.responses.create(
            model="gpt-5.4-mini",
            input=planner_prompt,
        )

        return parse_json_response(response.output_text)

    except Exception as error:
        raise RuntimeError(
            f"FluSight planner agent failed: {error}"
        ) from error


def run_flu_analyst_agent(
    user_task,
    planning_request,
    flu_summary,
):
    """
    Claude performs first-pass FluSight analysis.
    """

    analysis_scope = planning_request[
        "analysis_scope"
    ]

    dataset_summary = flu_summary[
        "dataset_summary"
    ]

    valid_seasons = dataset_summary[
        "seasons"
    ]

    latest_season = valid_seasons[-1]

    analyst_prompt = f"""
You are an epidemiological time-series analysis agent.

Your task is to interpret structured summaries derived from CDC FluSight
influenza hospitalization data.

USER TASK:
{user_task}

PLANNER DECISION:
{json.dumps(planning_request, indent=2)}

SELECTED ANALYSIS SCOPE:
{analysis_scope}

VALID SEASONS:
{json.dumps(valid_seasons, indent=2)}

LATEST SEASON:
{latest_season}

STRUCTURED FLUSIGHT SUMMARY:
{json.dumps(flu_summary, indent=2)}

AUTHORITATIVE DATA RULE:

All counts, dates, rates, jurisdiction lists, season labels,
timing groups, and peak classifications produced by Python
are authoritative.

Do NOT:
- recompute counts,
- infer missing group membership,
- reconstruct lists from memory,
- change dates,
- change rates,
- change thresholds,
- invent statistics,
- or claim a jurisdiction belongs to a timing group unless that
  membership is explicitly present in the supplied data.

SCOPE-COMPLIANCE RULE:

You must respect the planner's selected scope.

If analysis_scope == "latest_season":
- Focus only on the latest season.
- If more detail is needed, requested_season MUST equal:
  "{latest_season}"
- Do NOT request a prior season.
- Do NOT expand into cross-season analysis.

If analysis_scope == "cross_season":
- Compare seasons using only supplied cross-season data.
- You may request deeper detail for ONE valid season if useful.

If analysis_scope == "all_seasons":
- Analyze all supplied seasons.
- You may request deeper detail for ONE valid season if useful.

If needs_more_detail is false:
- requested_season MUST be null.

IMPORTANT DEFINITIONS:

- A flu season is labeled August through July.
- A season with "is_partial": true is incomplete.
- A jurisdiction is "early" only if its peak is MORE THAN 7 days
  before that season's dominant jurisdictional peak date.
- A jurisdiction is "late" only if its peak is MORE THAN 7 days
  after that season's dominant jurisdictional peak date.
- A jurisdiction within plus or minus 7 days is "typical".
- These classifications are relative to each season's dominant
  jurisdictional peak date.

SCIENTIFIC RULES:

1. Use only information contained in the supplied structured data.
2. Keep observed facts separate from possible explanations.
3. Do not attribute patterns to:
   - transmission dynamics
   - climate
   - demographics
   - population density
   - mobility
   - immunity
   - viral strains
   - healthcare access
   - reporting practices
   - public policy
   - post-pandemic effects
   unless those variables are explicitly present in the data.
4. Geographic statements must be descriptive only.
5. Do not introduce outside factual claims such as population rankings,
   regional epidemiological norms, or historical influenza behavior.
6. Do not describe incomplete seasons as complete.
7. Do not say a partial season captures a "tail", "residual period",
   or similar interpretation unless the supplied data establish that.
8. Do not call one season more synchronized than another unless the
   supplied counts support that statement.
9. Do not use terms such as "increasing", "decreasing", "trend", or
   "progression" unless the sequence of supplied values supports the claim.
10. If a possible explanation would require outside information,
    place it only in "hypotheses_requiring_external_data".

NUMERICAL CONSISTENCY RULES:

- If you mention a count, copy it from the structured data.
- If you list jurisdictions and state a count, the list length must match.
- Do not calculate new metrics such as coefficient of variation,
  correlation, significance, or percentage unless the exact calculation
  is trivial and directly supported by supplied values.
- If you calculate a simple percentage, ensure the numerator and
  denominator are explicitly present.
- Do not describe a rate mean as a coefficient of variation.
- Do not claim a range or duration unless the supplied dates support it.

OUTPUT LENGTH RULES:

- "supported_findings": at most 6 items
- "notable_geographic_patterns": at most 4 items
- "limitations": at most 4 items
- "hypotheses_requiring_external_data": at most 3 items
- Keep each list item concise.
- Do not restate the entire dataset.

Return ONLY valid JSON with exactly this structure:

{{
  "supported_findings": [
    "concise finding directly supported by the supplied data"
  ],
  "seasonality_summary": "concise descriptive summary",
  "spatiotemporal_summary": "concise descriptive summary of jurisdiction timing",
  "cross_season_changes": "concise comparison appropriate to the selected scope",
  "notable_geographic_patterns": [
    "descriptive geographic pattern supported by supplied data"
  ],
  "limitations": [
    "data or interpretation limitation"
  ],
  "hypotheses_requiring_external_data": [
    "question or hypothesis requiring outside data"
  ],
  "needs_more_detail": true,
  "requested_season": "valid season or null",
  "reason": "brief explanation of why more detail is or is not useful"
}}
"""

    try:
        message = claude_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=3000,
            messages=[
                {
                    "role": "user",
                    "content": analyst_prompt,
                }
            ],
        )

        response_text = message.content[0].text

        result = parse_json_response(response_text)

        # ----------------------------------------------------------
        # Deterministic post-response scope validation
        # ----------------------------------------------------------

        needs_more_detail = result.get(
            "needs_more_detail"
        )

        requested_season = result.get(
            "requested_season"
        )

        if not needs_more_detail:
            result["requested_season"] = None

        elif requested_season not in valid_seasons:
            raise ValueError(
                f"FluSight analyst requested invalid season: "
                f"{requested_season}"
            )

        elif (
            analysis_scope == "latest_season"
            and requested_season != latest_season
        ):
            raise ValueError(
                "FluSight analyst violated planner scope: "
                f"latest_season scope only allows "
                f"{latest_season}, but analyst requested "
                f"{requested_season}."
            )

        return result

    except Exception as error:
        raise RuntimeError(
            f"FluSight analyst agent failed: {error}"
        ) from error


def run_flu_second_pass_analyst(
    user_task,
    first_analysis,
    season_detail,
):
    """
    Claude performs a deeper second-pass analysis of one season
    using detailed jurisdiction-level peak information.
    """

    second_pass_prompt = f"""
You are performing a second-pass epidemiological time-series analysis.

The first analysis requested deeper detail for one influenza season.

USER TASK:
{user_task}

FIRST ANALYSIS:
{json.dumps(first_analysis, indent=2)}

DETAILED SEASON DATA:
{json.dumps(season_detail, indent=2)}

AUTHORITATIVE DATA RULE:

The Python-generated DETAILED SEASON DATA is authoritative.

Do NOT:
- recompute counts,
- infer missing jurisdiction membership,
- reconstruct timing groups,
- change dates,
- change rates,
- change peak classifications,
- introduce external factual claims,
- or invent numerical metrics.

If the first analysis conflicts with the detailed Python data,
follow the detailed Python data.

IMPORTANT DEFINITIONS:

- A jurisdiction is "early" only if its peak occurs MORE THAN 7 days
  before the dominant jurisdictional peak date.
- A jurisdiction is "late" only if its peak occurs MORE THAN 7 days
  after the dominant jurisdictional peak date.
- A jurisdiction within plus or minus 7 days is "typical".

SCIENTIFIC RULES:

1. Use only facts contained in FIRST ANALYSIS and DETAILED SEASON DATA.
2. Keep observations separate from explanations.
3. Geographic observations must remain descriptive.
4. Do not introduce outside facts such as population rankings,
   regional norms, or assumed influenza behavior.
5. Do not attribute patterns to mechanisms unless those variables
   are explicitly present.
6. If a possible explanation needs outside information, place it only
   in "hypotheses_requiring_external_data".
7. If you mention a group count and list its members, the count and
   list length must agree.
8. Do not create labels such as bimodal, trimodal, highly synchronized,
   or weakly synchronized unless the supplied distribution clearly
   supports that wording.
9. Do not invent significance, correlations, coefficients, or trends.
10. Prefer conservative descriptions over strong interpretation.

OUTPUT LENGTH RULES:

- "supported_findings": at most 6 items
- "notable_geographic_patterns": at most 4 items
- "limitations": at most 4 items
- "hypotheses_requiring_external_data": at most 3 items
- Keep each item concise.

Return ONLY valid JSON with exactly this structure:

{{
  "supported_findings": [
    "concise finding directly supported by detailed season data"
  ],
  "seasonality_summary": "concise refined seasonal interpretation",
  "spatiotemporal_summary": "concise jurisdiction-level timing interpretation",
  "cross_season_changes": "how this season relates to the first-pass comparison",
  "notable_geographic_patterns": [
    "descriptive geographic pattern supported by supplied data"
  ],
  "limitations": [
    "limitation of the current analysis"
  ],
  "hypotheses_requiring_external_data": [
    "question or hypothesis requiring outside data"
  ],
  "needs_more_detail": false,
  "requested_season": null,
  "reason": "brief explanation of what detailed data clarified"
}}
"""

    try:
        message = claude_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=3000,
            messages=[
                {
                    "role": "user",
                    "content": second_pass_prompt,
                }
            ],
        )

        response_text = message.content[0].text

        result = parse_json_response(response_text)

        # Second pass must always terminate the loop.
        result["needs_more_detail"] = False
        result["requested_season"] = None

        return result

    except Exception as error:
        raise RuntimeError(
            f"FluSight second-pass analyst failed: {error}"
        ) from error