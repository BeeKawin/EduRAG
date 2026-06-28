from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import (
    LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_SITE_URL,
)
from retrieval.rag_chain import get_rag_chain

from .eval_config import (
    BASELINE_MODEL,
    BASELINE_PROVIDER,
    JUDGE_MAX_RETRIES,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    JUDGE_TIMEOUT_SECONDS,
    LIVE_QUERY_DIR,
)
from .judge import normalize_answer_type
from .run_eval import _format_context


@dataclass
class LiveAnswerScores:
    correctness: int
    groundedness: int | None
    completeness: int
    clarity: int
    type_alignment: int
    target_answer_type: str
    detected_answer_type: str
    overall_band: int
    rationale: str


@dataclass
class LiveQueryScores:
    rag_scores: LiveAnswerScores
    baseline_scores: LiveAnswerScores
    winner: str
    comparison_rationale: str


def _iso_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _openrouter_headers() -> dict[str, str]:
    headers = {}
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        headers["X-OpenRouter-Title"] = OPENROUTER_APP_NAME
    return headers


def _get_openrouter_llm(model: str, temperature: float, max_tokens: int = 1024, timeout: int | None = None):
    from langchain_community.chat_models.openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "api_key": OPENROUTER_API_KEY,
        "base_url": OPENROUTER_BASE_URL,
        "default_headers": _openrouter_headers(),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
        kwargs["max_retries"] = JUDGE_MAX_RETRIES
    return ChatOpenAI(**kwargs)


def _get_baseline_llm(model: Optional[str] = None):
    if BASELINE_PROVIDER == "openrouter":
        return _get_openrouter_llm(model or BASELINE_MODEL, temperature=0.3)
    raise ValueError(
        f"Unsupported BASELINE_PROVIDER='{BASELINE_PROVIDER}'. Use openrouter for live query evaluation."
    )


def _get_judge_llm():
    if JUDGE_PROVIDER == "openrouter":
        return _get_openrouter_llm(
            JUDGE_MODEL,
            temperature=0,
            max_tokens=1024,
            timeout=JUDGE_TIMEOUT_SECONDS,
        )
    raise ValueError(
        f"Unsupported JUDGE_PROVIDER='{JUDGE_PROVIDER}'. Use openrouter for live query evaluation."
    )


def _target_language(language: Optional[str]) -> str:
    if not language:
        return "Match the user question language"
    normalized = language.strip().upper()
    if normalized in {"TH", "THAI"}:
        return "TH"
    if normalized in {"EN", "ENGLISH"}:
        return "EN"
    return "Match the user question language"


def _build_baseline_prompt(
    question: str,
    preferred_answer_type: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    answer_type = normalize_answer_type(preferred_answer_type)
    return f"""
You are an educational tutor for Thai high-school students.

Answer the user's question using your general knowledge only.
Do not claim that you used retrieved curriculum context or citations.

Target language: {_target_language(language)}
Target answer type: {answer_type}

Answer type meanings:
- quick-answer: short and direct.
- homework-help: step-by-step with formulas, substitutions, calculations, and final answer.
- concept-focused: explain meanings, relationships, and why ideas work.
- exam-prep: concise revision with key formulas, common exam points, and common mistakes.
- general: balanced, clear, educational answer.

Question:
{question}
""".strip()


def _build_baseline_messages(
    question: str,
    preferred_answer_type: Optional[str] = None,
    language: Optional[str] = None,
) -> list:
    answer_type = normalize_answer_type(preferred_answer_type)
    system = f"""
You are an educational tutor for Thai high-school students.
Answer the user's question directly.
Do not return safety labels, moderation labels, metadata, or policy classifications.
Do not claim that you used retrieved curriculum context or citations.

Target language: {_target_language(language)}
Target answer type: {answer_type}

Answer type meanings:
- quick-answer: short and direct.
- homework-help: step-by-step with formulas, substitutions, calculations, and final answer.
- concept-focused: explain meanings, relationships, and why ideas work.
- exam-prep: concise revision with key formulas, common exam points, and common mistakes.
- general: balanced, clear, educational answer.
""".strip()
    return [SystemMessage(content=system), HumanMessage(content=question)]


def _looks_like_non_answer(text: str) -> bool:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return True
    exact_non_answers = {
        "user safety: safe",
        "safe",
        "unsafe",
        "i cannot provide json for this request.",
    }
    if normalized in exact_non_answers:
        return True
    non_answer_markers = [
        "user safety:",
        "assistant safety:",
        "safety:",
        "content safety:",
        "moderation:",
        "policy:",
    ]
    return any(normalized.startswith(marker) for marker in non_answer_markers)


def generate_baseline_answer(
    question: str,
    preferred_answer_type: Optional[str] = None,
    language: Optional[str] = None,
    baseline_model: Optional[str] = None,
) -> str:
    llm = _get_baseline_llm(baseline_model)
    response = llm.invoke(_build_baseline_messages(question, preferred_answer_type, language))
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Live judge output did not contain a JSON object")
    return json.loads(text[start : end + 1])


def _coerce_score(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid boolean score: {value}")
    if isinstance(value, (int, float)):
        numeric = int(round(value))
    elif isinstance(value, str) and value.strip():
        numeric = int(round(float(value.strip())))
    else:
        raise ValueError(f"Invalid score: {value}")
    return max(1, min(5, numeric))


def calculate_live_overall_band(
    correctness: int,
    completeness: int,
    clarity: int,
    type_alignment: int,
) -> int:
    """Calculate a fair live-comparison band using metrics applicable to both answers."""
    overall_band = round(
        2.25
        * (
            correctness * (7 / 15)
            + completeness * (1 / 5)
            + clarity * (1 / 5)
            + type_alignment * (2 / 15)
        )
        - 1.25
    )
    overall_band = max(1, min(10, overall_band))

    if correctness <= 1:
        overall_band = min(overall_band, 3)
    elif correctness <= 2:
        overall_band = min(overall_band, 5)
    if completeness <= 2:
        overall_band = min(overall_band, 7)
    if type_alignment <= 2:
        overall_band = min(overall_band, 8)
    return overall_band


def _score_from_payload(
    payload: dict[str, Any],
    target_answer_type: str,
    *,
    include_groundedness: bool,
) -> LiveAnswerScores:
    if "type_alignment" not in payload and "safety" in payload:
        payload["type_alignment"] = payload["safety"]

    scores: dict[str, int] = {}
    for key in ["correctness", "completeness", "clarity", "type_alignment"]:
        try:
            scores[key] = _coerce_score(payload.get(key))
        except Exception as exc:
            raise ValueError(f"Invalid live judge score for '{key}': {payload.get(key)}") from exc

    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        raise ValueError("Live judge rationale must be a string")

    groundedness = None
    if include_groundedness:
        try:
            groundedness = _coerce_score(payload.get("groundedness"))
        except Exception as exc:
            raise ValueError(
                f"Invalid live judge score for 'groundedness': {payload.get('groundedness')}"
            ) from exc

    return LiveAnswerScores(
        correctness=scores["correctness"],
        groundedness=groundedness,
        completeness=scores["completeness"],
        clarity=scores["clarity"],
        type_alignment=scores["type_alignment"],
        target_answer_type=target_answer_type,
        detected_answer_type=normalize_answer_type(payload.get("detected_answer_type")),
        overall_band=calculate_live_overall_band(
            scores["correctness"],
            scores["completeness"],
            scores["clarity"],
            scores["type_alignment"],
        ),
        rationale=rationale.strip(),
    )


def parse_live_judge_response(text: str, preferred_answer_type: Optional[str] = None) -> LiveQueryScores:
    payload = _extract_json_object(text)
    target_answer_type = normalize_answer_type(preferred_answer_type)
    rag_scores = _score_from_payload(
        dict(payload.get("rag_scores") or {}), target_answer_type, include_groundedness=True
    )
    baseline_scores = _score_from_payload(
        dict(payload.get("baseline_scores") or {}), target_answer_type, include_groundedness=False
    )

    winner = "tie"
    if rag_scores.overall_band > baseline_scores.overall_band:
        winner = "rag"
    elif baseline_scores.overall_band > rag_scores.overall_band:
        winner = "baseline"

    comparison_rationale = payload.get("comparison_rationale", "")
    if not isinstance(comparison_rationale, str):
        raise ValueError("comparison_rationale must be a string")

    return LiveQueryScores(
        rag_scores=rag_scores,
        baseline_scores=baseline_scores,
        winner=winner,
        comparison_rationale=comparison_rationale.strip(),
    )


def _fallback_live_scores(
    preferred_answer_type: Optional[str],
    error: Exception,
    raw_output: str = "",
    rag_answer: str = "",
    baseline_answer: str = "",
) -> LiveQueryScores:
    target_answer_type = normalize_answer_type(preferred_answer_type)
    rationale = (
        "Live judge output was malformed, so these are neutral placeholder scores. "
        f"Parser error: {error}"
    )
    if raw_output.strip():
        rationale += f" Raw judge output preview: {raw_output.strip()[:180]}"

    rag_is_non_answer = _looks_like_non_answer(rag_answer)
    baseline_is_non_answer = _looks_like_non_answer(baseline_answer)

    rag_metric = 1 if rag_is_non_answer else 3
    baseline_metric = 1 if baseline_is_non_answer else 3
    rag_band = calculate_live_overall_band(rag_metric, rag_metric, rag_metric, rag_metric)
    baseline_band = calculate_live_overall_band(
        baseline_metric, baseline_metric, baseline_metric, baseline_metric
    )
    rag_rationale = rationale
    baseline_rationale = rationale
    if rag_is_non_answer:
        rag_rationale += " RAG answer looked like a non-answer/safety label."
    if baseline_is_non_answer:
        baseline_rationale += " Baseline answer looked like a non-answer/safety label."

    rag_scores = LiveAnswerScores(
        correctness=rag_metric,
        groundedness=rag_metric,
        completeness=rag_metric,
        clarity=rag_metric,
        type_alignment=rag_metric,
        target_answer_type=target_answer_type,
        detected_answer_type="general",
        overall_band=rag_band,
        rationale=rag_rationale,
    )
    baseline_scores = LiveAnswerScores(
        correctness=baseline_metric,
        groundedness=None,
        completeness=baseline_metric,
        clarity=baseline_metric,
        type_alignment=baseline_metric,
        target_answer_type=target_answer_type,
        detected_answer_type="general",
        overall_band=baseline_band,
        rationale=baseline_rationale,
    )
    winner = "tie"
    if rag_band > baseline_band:
        winner = "rag"
    elif baseline_band > rag_band:
        winner = "baseline"
    return LiveQueryScores(
        rag_scores=rag_scores,
        baseline_scores=baseline_scores,
        winner=winner,
        comparison_rationale=(
            "The RAG and baseline answers were generated, but the judge output was malformed and did not return valid JSON. "
            "Comparison scores are neutral placeholders; use a stronger judge model for reliable live evaluation."
        ),
    )


def _build_live_judge_prompt(
    question: str,
    context: str,
    rag_answer: str,
    baseline_answer: str,
    preferred_answer_type: Optional[str],
    language: Optional[str],
) -> str:
    target_answer_type = normalize_answer_type(preferred_answer_type)
    return f"""
You are an expert evaluator for educational answers.

Compare two answers to the same student question:
1. RAG answer: generated using retrieved curriculum context.
2. Baseline answer: generated without retrieval.

Judge both answers against the question, the retrieved curriculum context, and the selected response preferences.
Return ONLY one valid JSON object. The first character must be "{" and the last character must be "}".
Do not include markdown, code fences, prose, headings, or comments.

Use this JSON structure:
{{
  "rag_scores": {{
    "correctness": 1,
    "groundedness": 1,
    "completeness": 1,
    "clarity": 1,
    "type_alignment": 1,
    "detected_answer_type": "general",
    "rationale": "Short reason."
  }},
  "baseline_scores": {{
    "correctness": 1,
    "groundedness": null,
    "completeness": 1,
    "clarity": 1,
    "type_alignment": 1,
    "detected_answer_type": "general",
    "rationale": "Short reason."
  }},
  "comparison_rationale": "Short comparison."
}}

Scoring rubric:
- correctness: factual and mathematical accuracy.
- groundedness: support from retrieved curriculum context; score this only for the RAG answer and always return null for the baseline.
- completeness: covers the important ideas needed to answer.
- clarity: clear for high-school learners.
- type_alignment: matches the target answer type.

Applicable scores must be integers from 1 to 5; baseline groundedness must be null.
Never use 0. If an answer is extremely poor, use 1.

Important scoring behavior:
- Score each metric independently.
- If retrieved context is missing or weak, groundedness may be 1, but correctness, clarity, completeness, and type_alignment can still be higher if the answer deserves it.
- Do not give all-1 scores just because one metric is weak.
- Baseline groundedness is not applicable because the baseline does not receive retrieved context. Never penalize it for this.
- Compare the answers and choose the winner using correctness, completeness, clarity, and type_alignment only. RAG groundedness is diagnostic and must not influence the winner.

Target language: {_target_language(language)}
Target answer type: {target_answer_type}

Question:
{question}

Retrieved curriculum context:
{context}

RAG answer:
{rag_answer}

Baseline answer:
{baseline_answer}
""".strip()


def judge_live_answers(
    question: str,
    context: str,
    rag_answer: str,
    baseline_answer: str,
    preferred_answer_type: Optional[str] = None,
    language: Optional[str] = None,
) -> LiveQueryScores:
    prompt = _build_live_judge_prompt(
        question,
        context,
        rag_answer,
        baseline_answer,
        preferred_answer_type,
        language,
    )
    response = _get_judge_llm().invoke(prompt)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(str(item) for item in content)
    text = str(content)
    try:
        return parse_live_judge_response(text, preferred_answer_type)
    except Exception as exc:
        return _fallback_live_scores(preferred_answer_type, exc, text, rag_answer, baseline_answer)


def _live_query_path(timestamp: str) -> Path:
    return LIVE_QUERY_DIR / f"{timestamp[:8]}.jsonl"


def _append_live_result(row: dict[str, Any], timestamp: str) -> Path:
    path = _live_query_path(timestamp)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _winner_from_scores(scores: LiveQueryScores) -> str:
    if scores.rag_scores.overall_band > scores.baseline_scores.overall_band:
        return "rag"
    if scores.baseline_scores.overall_band > scores.rag_scores.overall_band:
        return "baseline"
    return scores.winner


def run_live_query_evaluation(
    message: str,
    subject: Optional[str] = None,
    grade: Optional[str] = None,
    preferred_answer_type: Optional[str] = None,
    language: Optional[str] = None,
    source: Optional[str] = None,
    top_k: int = 6,
    baseline_model: Optional[str] = None,
) -> dict[str, Any]:
    timestamp = _iso_ts()
    target_answer_type = normalize_answer_type(preferred_answer_type)
    chain = get_rag_chain()

    context_docs = chain.get_context_docs(
        message,
        subject=subject,
        grade=grade,
        top_k=top_k,
        language=language,
        source=source,
    )
    context = _format_context(context_docs)

    rag_answer = chain.ask(
        message,
        subject=subject,
        grade=grade,
        top_k=top_k,
        preferred_answer_type=target_answer_type,
        language=language,
        source=source,
    )
    baseline_answer = generate_baseline_answer(
        message,
        preferred_answer_type=target_answer_type,
        language=language,
        baseline_model=baseline_model,
    )
    scores = judge_live_answers(
        message,
        context,
        rag_answer,
        baseline_answer,
        preferred_answer_type=target_answer_type,
        language=language,
    )

    row = {
        "timestamp": timestamp,
        "question": message,
        "preferences": {
            "subject": subject,
            "grade": grade,
            "preferred_answer_type": target_answer_type,
            "language": language,
            "source": source,
            "top_k": top_k,
        },
        "models": {
            "rag_model": LLM_MODEL,
            "baseline_provider": BASELINE_PROVIDER,
            "baseline_model": baseline_model or BASELINE_MODEL,
            "judge_provider": JUDGE_PROVIDER,
            "judge_model": JUDGE_MODEL,
        },
        "retrieved_sources": context_docs,
        "rag_answer": rag_answer,
        "baseline_answer": baseline_answer,
        "rag_scores": asdict(scores.rag_scores),
        "baseline_scores": asdict(scores.baseline_scores),
        "winner": _winner_from_scores(scores),
        "comparison_rationale": scores.comparison_rationale,
    }
    saved_path = _append_live_result(row, timestamp)
    row["saved_result_path"] = str(saved_path)
    return row
