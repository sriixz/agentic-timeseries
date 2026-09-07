import json
import re

from anthropic import Anthropic
from openai import OpenAI


openai_client = OpenAI()
claude_client = Anthropic()


# -------------------------------------------------------------------
# GENERAL HELPERS
# -------------------------------------------------------------------

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


def collect_numeric_values(data):
    """
    Recursively collect numeric values that were
    deterministically supplied by Python.

    Used by the FluSight semantic validator to check
    whether Claude invents percentages.
    """

    values = []

    if isinstance(data, dict):
        for value in data.values():
            values.extend(
                collect_numeric_values(value)
            )

    elif isinstance(data, list):
        for value in data:
            values.extend(
                collect_numeric_values(value)
            )

    elif isinstance(data, (int, float)):
        values.append(
            float(data)
        )

    return values


# -------------------------------------------------------------------
# FLUSIGHT SEMANTIC VALIDATION
# -------------------------------------------------------------------

BANNED_FLU_PHRASES = [
    "synchronized",
    "synchronization",
    "desynchronized",
    "epidemic spread",
    "geographic spread",
    "geographic progression",
    "epidemic progression",
    "earlier initiation",
    "earlier epidemic initiation",
    "transmission from",
    "wave propagation",
    "extended tail",
    "residual period",
    "seasonal resolution",
    "epidemic resolution",
    "western states",
    "southern states",
    "northeastern states",
    "midwestern states",
    "coastal states",
    "mainland states",
    "continental divide",
    "times higher",
    "times lower",
    "fold higher",
    "fold lower",
    "fold difference",
    "only the post-peak phase",
    "post-peak activity",
    "capturing only post-peak",
]


VALIDATED_FLU_FIELDS = [
    "supported_findings",
    "seasonality_summary",
    "spatiotemporal_summary",
    "cross_season_changes",
    "notable_geographic_patterns",
    "limitations",
]


def extract_validated_flu_text(result):
    """
    Build one text string from fields expected to
    contain descriptive findings.

    hypotheses_requiring_external_data is excluded
    because that field may discuss mechanisms that
    require outside data.
    """

    parts = []

    for field in VALIDATED_FLU_FIELDS:
        value = result.get(field)

        if isinstance(value, list):
            parts.extend(
                str(item)
                for item in value
            )

        elif value is not None:
            parts.append(
                str(value)
            )

    return "\n".join(parts)


def find_percentage_values(text):
    """
    Return numerical percentage values found in text.

    Example:
        "23.5%" -> 23.5
    """

    matches = re.findall(
        r"(-?\d+(?:\.\d+)?)\s*%",
        text,
    )

    return [
        float(match)
        for match in matches
    ]


def percentage_is_supported(
    percentage,
    authoritative_numbers,
    tolerance=0.01,
):
    """
    Check whether a percentage quoted by Claude was
    already supplied by Python.

    Accept exact values, sign-flipped magnitudes, and
    reasonable rounded versions such as 97.0% when
    Python supplied -97.14%.
    """

    for number in authoritative_numbers:
        exact_difference = abs(
            percentage - number
        )

        magnitude_difference = abs(
            abs(percentage)
            - abs(number)
        )

        # Exact or essentially exact match.
        if exact_difference <= tolerance:
            return True

        # Same magnitude with opposite sign.
        if magnitude_difference <= tolerance:
            return True

        # Allow one-decimal rounding.
        if round(
            abs(percentage),
            1,
        ) == round(
            abs(number),
            1,
        ):
            return True

        # Allow whole-number rounding.
        if round(
            abs(percentage)
        ) == round(
            abs(number)
        ):
            return True

    return False


def validate_flu_analysis_language(
    result,
    authoritative_data,
):
    """
    Deterministically inspect Claude's descriptive
    FluSight output for known failure modes.

    Returns a list of human-readable violations.
    """

    text = extract_validated_flu_text(
        result
    )

    lower_text = text.lower()

    violations = []

    # --------------------------------------------------------------
    # Unsupported interpretive language
    # --------------------------------------------------------------

    for phrase in BANNED_FLU_PHRASES:
        if phrase in lower_text:
            violations.append(
                f'unsupported phrase: "{phrase}"'
            )

    # --------------------------------------------------------------
    # Additional risky wording
    # --------------------------------------------------------------

    risky_patterns = {
        r"\bprogression\b":
            'unsupported term: "progression"',
        r"\bpropagation\b":
            'unsupported term: "propagation"',
        r"\binitiation\b":
            'unsupported term: "initiation"',
        r"\btransmission\b":
            'unsupported term: "transmission"',
        r"\bresolution\b":
            'unsupported term: "resolution"',
        r"\brapid(?:ly)?\b":
            'unsupported qualitative term: "rapid"',
    }

    for pattern, message in risky_patterns.items():
        if re.search(
            pattern,
            lower_text,
        ):
            violations.append(
                message
            )

    # --------------------------------------------------------------
    # New ratios / fold changes
    # --------------------------------------------------------------

    ratio_patterns = [
        r"\b\d+(?:\.\d+)?\s*x\b",
        r"\b\d+(?:\.\d+)?\s*times\b",
        r"\b\d+(?:\.\d+)?[- ]fold\b",
    ]

    for pattern in ratio_patterns:
        matches = re.findall(
            pattern,
            lower_text,
        )

        for match in matches:
            violations.append(
                "unsupplied ratio or fold-change: "
                f'"{match}"'
            )

    # --------------------------------------------------------------
    # New percentages
    # --------------------------------------------------------------

    authoritative_numbers = (
        collect_numeric_values(
            authoritative_data
        )
    )

    percentages = (
        find_percentage_values(
            text
        )
    )

    for percentage in percentages:
        if not percentage_is_supported(
            percentage,
            authoritative_numbers,
        ):
            violations.append(
                "percentage not supplied by Python: "
                f"{percentage}%"
            )

    # Remove duplicates while preserving order.
    violations = list(
        dict.fromkeys(
            violations
        )
    )

    return violations


def validate_flu_output_structure(result):
    """
    Check deterministic output-length constraints.
    """

    violations = []

    list_limits = {
        "supported_findings": 6,
        "notable_geographic_patterns": 4,
        "limitations": 4,
        "hypotheses_requiring_external_data": 3,
    }

    for field, maximum in list_limits.items():
        value = result.get(field)

        if not isinstance(value, list):
            violations.append(
                f'"{field}" must be a list'
            )
            continue

        if len(value) > maximum:
            violations.append(
                f'"{field}" contains {len(value)} items; '
                f"maximum is {maximum}"
            )

    return violations


def repair_flu_analysis(
    user_task,
    original_result,
    authoritative_data,
    violations,
):
    """
    Give Claude one constrained repair attempt when
    deterministic semantic or structural validation
    finds problems.
    """

    repair_prompt = f"""
You are repairing a structured epidemiological
time-series analysis.

The previous analysis violated deterministic
output rules.

USER TASK:
{user_task}

AUTHORITATIVE PYTHON DATA:
{json.dumps(authoritative_data, indent=2)}

PREVIOUS ANALYSIS:
{json.dumps(original_result, indent=2)}

DETECTED VIOLATIONS:
{json.dumps(violations, indent=2)}

Rewrite the analysis so that every detected violation
is removed.

STRICT REPAIR RULES:

- Preserve the exact JSON structure.
- Use only facts explicitly present in the Python data.
- Copy numerical values from Python.
- Do not calculate new percentages.
- Do not calculate ratios or fold changes.
- Do not introduce regional group labels.
- Do not introduce geographic labels based on outside
  knowledge such as continental regions or divides.
- Do not interpret jurisdiction peak timing as evidence
  of transmission, spread, initiation, propagation,
  progression, or epidemic movement.
- Do not use synchronization terminology.
- Do not describe unobserved portions of partial seasons.
- For partial seasons, refer only to observed data.
- Do not call an observed period "post-peak activity"
  unless the supplied data prove that the true full-season
  peak occurred before the first available observation.
- Prefer direct counts and dates over interpretive labels.
- Do not imply a monotonic increase or decrease across
  seasons unless every intermediate season supports
  that direction.
- Keep hypotheses requiring outside information only in
  "hypotheses_requiring_external_data".
- Do not remove required JSON keys.

OUTPUT LENGTH LIMITS:

- supported_findings: maximum 6 items
- notable_geographic_patterns: maximum 4 items
- limitations: maximum 4 items
- hypotheses_requiring_external_data: maximum 3 items

Return ONLY valid JSON with exactly this structure:

{{
  "supported_findings": [
    "concise finding directly supported by the supplied data"
  ],
  "seasonality_summary": "concise descriptive summary",
  "spatiotemporal_summary": "concise descriptive summary of jurisdiction timing",
  "cross_season_changes": "concise comparison appropriate to the selected scope",
  "notable_geographic_patterns": [
    "descriptive jurisdiction timing pattern supported by supplied data"
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

    message = claude_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=3000,
        messages=[
            {
                "role": "user",
                "content": repair_prompt,
            }
        ],
    )

    response_text = (
        message.content[0].text
    )

    return parse_json_response(
        response_text
    )


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

        return parse_json_response(
            response.output_text
        )

    except Exception as error:
        raise RuntimeError(
            f"Retriever agent failed: {error}"
        ) from error


def run_analyst_agent(
    user_task,
    retrieval_request,
    data,
):
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
- If more historical context would materially improve
  the analysis, set "needs_more_data" to true.
- If the current data is sufficient, set it to false.
- If needs_more_data is false, requested_period should
  stay the same as the current period.
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

        response_text = (
            message.content[0].text
        )

        return parse_json_response(
            response_text
        )

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

        response_text = (
            message.content[0].text
        )

        return parse_json_response(
            response_text
        )

    except Exception as error:
        raise RuntimeError(
            f"Second-pass analyst failed: {error}"
        ) from error


# -------------------------------------------------------------------
# FLUSIGHT WORKFLOW AGENTS
# -------------------------------------------------------------------

def run_flu_planner_agent(
    user_task,
    dataset_summary,
):
    """
    GPT decides how the FluSight dataset should be analyzed.
    """

    planner_prompt = f"""
You are a planning agent for an epidemiological
time-series analysis workflow.

The user wants to analyze CDC FluSight influenza
hospitalization data.

USER TASK:
{user_task}

DATASET METADATA:
{json.dumps(dataset_summary, indent=2)}

Your job is NOT to perform the epidemiological
analysis itself.

Instead, decide what analytical scope is most appropriate.

Available analysis scopes:

- "all_seasons"
- "latest_season"
- "cross_season"

Scope definitions:

- "latest_season":
  Use when the task explicitly focuses on the most
  recent season.

- "cross_season":
  Use when the task asks how patterns change across
  seasons or asks for comparisons across time.

- "all_seasons":
  Use when the task explicitly asks for detailed
  analysis of every season.

Consider:

- whether the task requires multiple seasons
- whether national seasonality matters
- whether current-season trends should be compared
  with past-season trends
- whether hospitalization counts matter
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

        return parse_json_response(
            response.output_text
        )

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

Your task is to interpret structured summaries derived
from CDC FluSight influenza hospitalization data.

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
timing groups, peak classifications, hospitalization trends,
and current-vs-past comparisons produced by Python
are authoritative.

Do NOT:

- recompute counts
- infer missing group membership
- reconstruct lists from memory
- change dates
- change hospitalization values
- change rates
- change thresholds
- invent statistics
- claim a jurisdiction belongs to a timing group unless that
  membership is explicitly present in the supplied data

METRIC INTERPRETATION RULE:

The FluSight dataset contains two related but distinct
hospitalization measures.

1. RAW HOSPITALIZATION COUNT

The field "value" represents the weekly number of
hospital admissions.

For:

- US national trend analysis
- current-season trend interpretation
- past-season trend interpretation
- national peak magnitude
- comparisons of national peak magnitude across seasons
- weekly changes in US hospitalization activity

treat raw hospitalization count ("value") as the
PRIMARY metric.

When fields such as:

- latest_hospital_admissions
- previous_week_hospital_admissions
- weekly_change_in_admissions
- peak_hospital_admissions
- change_from_peak
- percent_change_from_peak

are supplied, prioritize those fields when describing
national trends.

2. WEEKLY RATE

The field "weekly_rate" is a normalized hospitalization
measure.

Use weekly_rate primarily for:

- comparing jurisdictions with different population sizes
- jurisdiction peak-rate statistics
- early / typical / late timing-group rate comparisons
- identifying jurisdictions with high or low normalized
  peak rates

Do NOT replace national hospitalization counts with
weekly_rate when describing how many admissions occurred.

Do NOT treat raw state hospitalization counts as directly
comparable measures of intensity across differently sized
jurisdictions unless the supplied data explicitly supports
that comparison.

CURRENT VS PAST SEASON RULE:

If the supplied data include:

- "national_season_trends"
- "current_season_trend"
- "current_peak_vs_past_seasons"

use those fields to explicitly describe how the current
season compares with previous seasons when allowed by
the selected scope.

Do not calculate additional comparisons if Python has
already supplied the comparison.

Do not imply that peak counts increased or decreased
monotonically across several seasons unless every
intermediate season supports that direction.

If one season decreases before another later season
increases, describe those values separately.

PARTIAL-SEASON RULE:

A season with:

"is_partial": true

is incomplete.

For a partial season:

- describe only what is observed in the available data
- call its reported peak the "observed peak" or
  "maximum in the available data"
- do not infer where an unobserved full-season peak
  would have occurred
- do not infer what happened outside the observed dates
- do not describe missing portions as onset, tail,
  residual activity, resolution, or post-peak activity
- do not call the observed period post-peak unless the
  supplied data establish that the true full-season peak
  occurred before the first available observation

SCOPE-COMPLIANCE RULE:

You must respect the planner's selected scope.

If analysis_scope == "latest_season":

- Focus only on the latest season.
- You may describe current-season national count trends.
- If historical comparisons are supplied, use them only
  to contextualize the latest season.
- If more detail is needed, requested_season MUST equal:
  "{latest_season}"
- Do NOT request a prior season.

If analysis_scope == "cross_season":

- Compare seasons using only supplied cross-season data.
- Explicitly consider current-season versus past-season
  hospitalization-count trends when available.
- You may request deeper detail for ONE valid season.

If analysis_scope == "all_seasons":

- Analyze all supplied seasons.
- Include hospitalization-count trends where supplied.
- You may request deeper detail for ONE valid season.

If needs_more_detail is false:

- requested_season MUST be null.

IMPORTANT DEFINITIONS:

- A flu season is labeled August through July.
- A season with "is_partial": true is incomplete.
- A jurisdiction is "early" only if its peak is MORE THAN
  7 days before that season's dominant jurisdictional
  peak date.
- A jurisdiction is "late" only if its peak is MORE THAN
  7 days after that dominant date.
- A jurisdiction within plus or minus 7 days is "typical".

DESCRIPTIVE LANGUAGE RULE:

Describe timing patterns directly using the quantities
provided by Python.

Prefer statements like:

- "37 jurisdictions peaked on January 3"
- "47 jurisdictions were in the typical timing group"
- "12 jurisdictions peaked more than 7 days before the
  dominant date"

Do NOT translate these observations into stronger
epidemiological concepts.

Do NOT use terms such as:

- synchronized
- synchronization
- desynchronized
- epidemic spread
- geographic spread
- progression
- initiation
- propagation
- extended tail
- residual period
- resolution

when describing observed findings.

GEOGRAPHIC LABEL RULE:

Do not introduce geographic region labels unless those
labels are explicitly supplied by Python.

Do NOT create labels based on outside geographic knowledge,
including:

- western states
- southern states
- northeastern states
- midwestern states
- coastal states
- mainland states
- continental divide
- east/west regional groupings

You may name individual jurisdictions exactly as supplied.

SCIENTIFIC RULES:

1. Use only supplied structured data.

2. Keep observations separate from possible explanations.

3. Do not attribute observed patterns to:

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

   unless those variables are explicitly present.

4. External mechanisms may appear only under
   "hypotheses_requiring_external_data".

5. Do not describe incomplete seasons as complete.

6. Do not infer unobserved portions of partial seasons.

7. Do not calculate synchronization or dispersion metrics.

8. Do not infer directional movement between jurisdictions.

9. Distinguish hospitalization count from weekly rate.

10. For national hospitalization burden, use counts.

11. Do not imply a monotonic trend across seasons unless
    every intermediate value supports that direction.

NUMERICAL CONSISTENCY RULES:

- Copy counts and dates from Python.
- Prefer Python-supplied percentages.
- Do not calculate new percentages.
- Do not calculate ratios.
- Do not calculate fold changes.
- Do not calculate correlations.
- Do not calculate significance.
- Do not calculate coefficients of variation.
- Do not create new synchronization metrics.

OUTPUT CONTENT RULE:

When permitted by scope and available in the data,
address BOTH:

1. current-season national hospitalization-count trends
2. comparison with past-season national trends

Jurisdiction timing should be described using dates
and group counts rather than epidemiological mechanisms.

OUTPUT LENGTH RULES:

- "supported_findings": at most 6 items
- "notable_geographic_patterns": at most 4 items
- "limitations": at most 4 items
- "hypotheses_requiring_external_data": at most 3 items
- Keep each item concise.

Return ONLY valid JSON with exactly this structure:

{{
  "supported_findings": [
    "concise finding directly supported by the supplied data"
  ],
  "seasonality_summary": "concise descriptive summary",
  "spatiotemporal_summary": "concise descriptive summary of jurisdiction timing",
  "cross_season_changes": "concise comparison appropriate to the selected scope",
  "notable_geographic_patterns": [
    "descriptive jurisdiction timing pattern supported by supplied data"
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

        response_text = (
            message.content[0].text
        )

        result = parse_json_response(
            response_text
        )

        # ----------------------------------------------------------
        # Deterministic semantic + structural validation
        # ----------------------------------------------------------

        violations = (
            validate_flu_analysis_language(
                result,
                flu_summary,
            )
        )

        violations.extend(
            validate_flu_output_structure(
                result
            )
        )

        if violations:
            print(
                "\n--- FLUSIGHT SEMANTIC "
                "VALIDATION ---"
            )

            print(
                "Violations detected:"
            )

            for violation in violations:
                print(
                    f"- {violation}"
                )

            print(
                "\nAttempting one constrained "
                "repair pass..."
            )

            result = repair_flu_analysis(
                user_task,
                result,
                flu_summary,
                violations,
            )

            repaired_violations = (
                validate_flu_analysis_language(
                    result,
                    flu_summary,
                )
            )

            repaired_violations.extend(
                validate_flu_output_structure(
                    result
                )
            )

            if repaired_violations:
                raise ValueError(
                    "FluSight analyst still violated "
                    "validation after repair: "
                    + "; ".join(
                        repaired_violations
                    )
                )

            print(
                "Semantic repair passed."
            )

        # ----------------------------------------------------------
        # Deterministic scope validation
        # ----------------------------------------------------------

        needs_more_detail = result.get(
            "needs_more_detail"
        )

        requested_season = result.get(
            "requested_season"
        )

        if not needs_more_detail:
            result[
                "requested_season"
            ] = None

        elif requested_season not in valid_seasons:
            raise ValueError(
                "FluSight analyst requested invalid season: "
                f"{requested_season}"
            )

        elif (
            analysis_scope
            == "latest_season"
            and requested_season
            != latest_season
        ):
            raise ValueError(
                "FluSight analyst violated planner scope: "
                "latest_season scope only allows "
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
    Claude performs a deeper second-pass analysis
    of one season.
    """

    second_pass_prompt = f"""
You are performing a second-pass epidemiological
time-series analysis.

USER TASK:
{user_task}

FIRST ANALYSIS:
{json.dumps(first_analysis, indent=2)}

DETAILED SEASON DATA:
{json.dumps(season_detail, indent=2)}

The Python-generated detailed data are authoritative.

Use raw hospitalization counts for national trends.

Use weekly_rate primarily for normalized
cross-jurisdiction comparisons.

Do not:

- invent counts
- change dates
- calculate new percentages
- calculate ratios or fold changes
- introduce regional labels
- introduce geographic groupings based on outside knowledge
- infer transmission or spread
- infer geographic progression
- use synchronization terminology
- infer unobserved portions of partial seasons
- describe a partial observed period as post-peak activity
  unless the supplied data prove the true full-season peak
  occurred before the first available observation
- imply a monotonic increase or decrease unless all
  supplied intermediate values support it

Describe jurisdiction timing directly with supplied
dates and group counts.

If external information would be required to explain
a pattern, place it only in
"hypotheses_requiring_external_data".

OUTPUT LENGTH LIMITS:

- supported_findings: maximum 6 items
- notable_geographic_patterns: maximum 4 items
- limitations: maximum 4 items
- hypotheses_requiring_external_data: maximum 3 items

Return ONLY valid JSON with exactly this structure:

{{
  "supported_findings": [
    "concise finding directly supported by detailed season data"
  ],
  "seasonality_summary": "concise refined seasonal interpretation",
  "spatiotemporal_summary": "concise jurisdiction-level timing interpretation",
  "cross_season_changes": "how this season relates to the first-pass comparison",
  "notable_geographic_patterns": [
    "descriptive jurisdiction timing pattern supported by supplied data"
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

        response_text = (
            message.content[0].text
        )

        result = parse_json_response(
            response_text
        )

        # ----------------------------------------------------------
        # Semantic + structural validation
        # ----------------------------------------------------------

        violations = (
            validate_flu_analysis_language(
                result,
                season_detail,
            )
        )

        violations.extend(
            validate_flu_output_structure(
                result
            )
        )

        if violations:
            print(
                "\n--- SECOND-PASS SEMANTIC "
                "VALIDATION ---"
            )

            print(
                "Violations detected:"
            )

            for violation in violations:
                print(
                    f"- {violation}"
                )

            print(
                "\nAttempting one constrained "
                "repair pass..."
            )

            result = repair_flu_analysis(
                user_task,
                result,
                season_detail,
                violations,
            )

            repaired_violations = (
                validate_flu_analysis_language(
                    result,
                    season_detail,
                )
            )

            repaired_violations.extend(
                validate_flu_output_structure(
                    result
                )
            )

            if repaired_violations:
                raise ValueError(
                    "FluSight second-pass analyst "
                    "still violated validation: "
                    + "; ".join(
                        repaired_violations
                    )
                )

            print(
                "Second-pass semantic repair passed."
            )

        # Second pass must always terminate the loop.
        result[
            "needs_more_detail"
        ] = False

        result[
            "requested_season"
        ] = None

        return result

    except Exception as error:
        raise RuntimeError(
            f"FluSight second-pass analyst failed: {error}"
        ) from error