import json

from openai import OpenAI
from anthropic import Anthropic


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
            input=retriever_prompt
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
                    "content": analyst_prompt
                }
            ]
        )

        response_text = message.content[0].text

        return parse_json_response(response_text)

    except Exception as error:
        raise RuntimeError(
            f"Analyst agent failed: {error}"
        ) from error


def run_second_pass_analyst(user_task, first_analysis, expanded_data):
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
                    "content": second_analyst_prompt
                }
            ]
        )

        response_text = message.content[0].text

        return parse_json_response(response_text)

    except Exception as error:
        raise RuntimeError(
            f"Second-pass analyst failed: {error}"
        ) from error