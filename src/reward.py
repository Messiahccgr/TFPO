import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

from src.utils import setup_logger


logger = setup_logger("reward")

try:
    from math_verify import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        LatexNormalizationConfig,
        parse,
        verify,
    )

    _MATH_VERIFY_AVAILABLE = True
    _MATH_VERIFY_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on runtime environment.
    ExprExtractionConfig = None  # type: ignore[assignment]
    LatexExtractionConfig = None  # type: ignore[assignment]
    LatexNormalizationConfig = None  # type: ignore[assignment]
    parse = None  # type: ignore[assignment]
    verify = None  # type: ignore[assignment]
    _MATH_VERIFY_AVAILABLE = False
    _MATH_VERIFY_IMPORT_ERROR = exc


_MATH_VERIFY_WARNING_SHOWN = False
TAIL_SCAN_LINE_LIMIT = 12
LIST_MARKER_PATTERN = re.compile(r"^\s*(?:[-*]\s+|\d+\s*[\.\)]\s+)")
ANSWER_PREFIX_PATTERN = re.compile(
    r"^\s*(?:answer|final\s*answer)\s*[:\uFF1A]\s*",
    re.IGNORECASE,
)
LEGACY_FINAL_ANSWER_PATTERN = re.compile(
    r"^\s*Final Answer\s*[:\uFF1A]\s*(?P<answer>.*?)\s*$",
    re.IGNORECASE,
)
DEEPSEEK_FINAL_ANSWER_PATTERN = re.compile(
    r"^\s*Therefore,\s*the\s+final\s+answer\s+is\s*[:\uFF1A]\s*"
    r"(?P<answer>.+?)(?:\.\s*I\s+hope\s+it\s+is\s+correct)?\s*$",
    re.IGNORECASE,
)
DEEPSEEK_FINAL_ANSWER_HEADER_PATTERN = re.compile(
    r"^\s*Therefore,\s*the\s+final\s+answer\s+is\s*[:\uFF1A]\s*$",
    re.IGNORECASE,
)
PLAIN_ANSWER_LINE_PATTERN = re.compile(
    r"^\s*(?P<label>answer|final\s*answer)\s*[:\uFF1A]\s*(?P<answer>.*?)\s*$",
    re.IGNORECASE,
)
BOLD_ANSWER_LINE_PATTERN = re.compile(
    r"^\s*\*\*(?P<label>answer|final\s*answer)\s*[:\uFF1A]\*\*\s*(?P<answer>.*?)\s*$",
    re.IGNORECASE,
)
GSM8K_FINAL_ANSWER_PATTERN = re.compile(
    r"^\s*####\s*(?P<answer>.+?)\s*$",
    re.MULTILINE,
)
MEMBERSHIP_WRAPPER_PATTERN = re.compile(
    r"^\s*[A-Za-z](?:_[A-Za-z0-9]+)?\s*(?:\\in|∈)\s*(?P<body>.+?)\s*$"
)
LATEX_TEXT_COMMAND_PATTERN = re.compile(
    r"\\(?:text|mathrm|textbf|textit|mathbf|mathit)\{([^}]*)\}"
)
SIMPLE_EQUATION_WRAPPER_PATTERN = re.compile(
    r"^\s*[A-Za-z](?:_[A-Za-z0-9]+)?\s*=\s*(?P<body>.+?)\s*$"
)
ANGLE_SUFFIX_PATTERN = re.compile(
    r"^\s*(?P<body>.+?)\s*(?:\^?\s*\\circ|°)\s*$",
    re.IGNORECASE,
)
SCALAR_UNIT_SUFFIX_PATTERN = re.compile(
    r"^\s*(?P<body>.+?)\s+"
    r"(?:inch(?:es)?|cm|mm|meters?|meter|units?|unit|calories?|calorie|degrees?)"
    r"\.?\s*$",
    re.IGNORECASE,
)
BARE_BOXED_ANSWER_PATTERN = re.compile(
    r"""
    (?P<answer>
        \\[a-zA-Z]+(?:\{[^{}]*\})* |
        \([^()\n]*\) |
        \[[^\[\]\n]*\] |
        [-+]?[^\s,;:!?]+
    )
    """,
    re.VERBOSE,
)
LATEX_WRAPPER_LINES = frozenset((r"\[", r"\]", r"\(", r"\)", "$", "$$"))
CONCLUSION_CUES = ("conclusion", "therefore", "thus", "hence", "so")


@dataclass(frozen=True)
class ParsedAnswerLine:
    label: str
    raw_content: str
    answer_text: Optional[str]
    boxed_answer_text: Optional[str]


@dataclass(frozen=True)
class TailAnswerInfo:
    answer_text: Optional[str]
    boxed_answer_text: Optional[str]
    source: Optional[str]
    is_standard: bool
    violation_reasons: tuple[str, ...]


def _canonicalize_simple_latex_shorthand(text: Optional[str]) -> str:
    if text is None:
        return ""

    normalized = str(text).strip()
    normalized = ANSWER_PREFIX_PATTERN.sub("", normalized, count=1)
    normalized = normalized.replace("−", "-")
    normalized = normalized.replace("–", "-")
    normalized = normalized.replace("—", "-")
    normalized = normalized.replace("⁄", "/")
    normalized = normalized.replace("∕", "/")
    normalized = normalized.replace(r"\(", "")
    normalized = normalized.replace(r"\)", "")
    normalized = normalized.replace(r"\[", "")
    normalized = normalized.replace(r"\]", "")
    normalized = re.sub(r"\\(?:dfrac|tfrac)", r"\\frac", normalized)
    normalized = re.sub(
        r"√\s*(?P<arg>[A-Za-z0-9]+)",
        lambda m: f"\\sqrt{{{m.group('arg')}}}",
        normalized,
    )
    normalized = re.sub(
        r"\\frac\s*(?P<num>[A-Za-z0-9])\s*(?P<den>[A-Za-z0-9])",
        lambda m: f"\\frac{{{m.group('num')}}}{{{m.group('den')}}}",
        normalized,
    )
    normalized = re.sub(
        r"\\frac\s*(?P<num>[A-Za-z0-9])\s*\{(?P<den>[^{}]+)\}",
        lambda m: f"\\frac{{{m.group('num')}}}{{{m.group('den')}}}",
        normalized,
    )
    normalized = re.sub(
        r"\\frac\s*\{(?P<num>[^{}]+)\}\s*(?P<den>[A-Za-z0-9])",
        lambda m: f"\\frac{{{m.group('num')}}}{{{m.group('den')}}}",
        normalized,
    )
    normalized = re.sub(
        r"\\sqrt\s*(?P<arg>[A-Za-z0-9])",
        lambda m: f"\\sqrt{{{m.group('arg')}}}",
        normalized,
    )
    simple_fraction_patterns = [
        re.compile(
            r"(?P<num>\\sqrt\{[^{}]+\}|[A-Za-z0-9]+)\s*/\s*(?P<den>\\sqrt\{[^{}]+\}|[A-Za-z0-9]+)"
        ),
    ]
    for pattern in simple_fraction_patterns:
        normalized = pattern.sub(
            lambda m: f"\\frac{{{m.group('num')}}}{{{m.group('den')}}}",
            normalized,
        )
    return normalized


def _split_top_level_csv(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0

    for ch in text:
        if ch == "," and depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(depth_paren - 1, 0)
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket = max(depth_bracket - 1, 0)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(depth_brace - 1, 0)

    parts.append("".join(current))
    return parts


def _canonicalize_unordered_scalar_list(text: str) -> str:
    if "," not in text:
        return text
    lowered = text.lower()
    if any(
        token in lowered
        for token in ("begin{", "end{", "matrix", "pmatrix", "bmatrix", "cases", "cup")
    ):
        return text
    parts = [part.strip() for part in _split_top_level_csv(text)]
    if len(parts) < 2 or any(not part for part in parts):
        return text
    if any(any(ch in part for ch in "()[]{}=:") for part in parts):
        return text
    if not all(any(ch.isdigit() for ch in part) for part in parts):
        return text
    return ",".join(sorted(parts))


def _combine_multiple_answers(values: tuple[str, ...]) -> Optional[str]:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned)


@dataclass(frozen=True)
class PredictionFormatInfo:
    final_answer_lines: tuple[str, ...]
    boxed_answers: tuple[str, ...]
    last_non_empty_line: str
    last_final_answer_line: Optional[str]
    final_answer_text: Optional[str]
    final_answer_boxed_text: Optional[str]
    boxed_answer_text: Optional[str]
    violation_reasons: tuple[str, ...]
    answer_only_text: Optional[str]
    answer_source: Optional[str]
    is_standard_final_answer: bool

    @property
    def is_valid(self) -> bool:
        return len(self.violation_reasons) == 0


@dataclass(frozen=True)
class CompletionPayloadInfo:
    text: str
    finish_reason: Optional[str]


@dataclass(frozen=True)
class RewardScore:
    total_reward: float
    format_reward: float
    answer_reward: float
    length_reward: float
    answer_only_text: Optional[str]
    has_final_submission: bool
    is_answer_correct: bool
    finish_reason: Optional[str]
    format_info: PredictionFormatInfo


def _normalize_finish_reason(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _extract_text_from_content_list(items: list[Any]) -> list[str]:
    parts: list[str] = []
    for item in items:
        if isinstance(item, dict) and "text" in item:
            parts.append(str(item["text"]))
    return parts


def _normalize_completion_payload(completion: Any) -> CompletionPayloadInfo:
    if isinstance(completion, str):
        return CompletionPayloadInfo(text=completion, finish_reason=None)

    if isinstance(completion, dict):
        finish_reason = None
        if "finish_reason" in completion:
            finish_reason = _normalize_finish_reason(completion.get("finish_reason"))

        content = completion.get("content")
        if isinstance(content, str):
            return CompletionPayloadInfo(text=content, finish_reason=finish_reason)
        if isinstance(content, list):
            parts = _extract_text_from_content_list(content)
            if parts:
                return CompletionPayloadInfo(
                    text="".join(parts),
                    finish_reason=finish_reason,
                )
        if "text" in completion:
            return CompletionPayloadInfo(
                text=str(completion["text"]),
                finish_reason=finish_reason,
            )
        return CompletionPayloadInfo(text=str(completion), finish_reason=finish_reason)

    if isinstance(completion, list):
        if len(completion) == 0:
            return CompletionPayloadInfo(text="", finish_reason=None)
        if all(isinstance(msg, dict) for msg in completion):
            parts: list[str] = []
            finish_reason = None
            for msg in completion:
                if finish_reason is None and "finish_reason" in msg:
                    finish_reason = _normalize_finish_reason(msg.get("finish_reason"))
                content = msg.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    parts.extend(_extract_text_from_content_list(content))
                elif "text" in msg:
                    parts.append(str(msg["text"]))
            return CompletionPayloadInfo(
                text="\n".join(parts),
                finish_reason=finish_reason,
            )
        return CompletionPayloadInfo(
            text=" ".join(str(item) for item in completion),
            finish_reason=None,
        )

    return CompletionPayloadInfo(text=str(completion), finish_reason=None)


def _match_final_answer_line(line: str):
    for pattern in (DEEPSEEK_FINAL_ANSWER_PATTERN, LEGACY_FINAL_ANSWER_PATTERN):
        match = pattern.match(line)
        if match is not None:
            return match
    return None


def extract_boxed_contents(text: str) -> list[str]:
    """Extract \\boxed{...} and simple bare \\boxed ... payloads in order."""
    marker = r"\boxed"
    idx = text.find(marker)
    if idx == -1:
        return []

    all_matches = []
    marker_len = len(marker)
    text_len = len(text)

    while idx != -1:
        pos = idx + marker_len
        while pos < text_len and text[pos].isspace():
            pos += 1

        if pos >= text_len:
            break

        if text[pos] == "{":
            start = pos + 1
            brace_count = 1
            end = start
            while end < text_len and brace_count > 0:
                if text[end] == "{":
                    brace_count += 1
                elif text[end] == "}":
                    brace_count -= 1
                end += 1
            if brace_count == 0:
                candidate = text[start : end - 1].strip()
                if candidate:
                    all_matches.append(candidate)
                idx = text.find(marker, end)
                continue

        bare_match = BARE_BOXED_ANSWER_PATTERN.match(text[pos:])
        if bare_match is not None:
            candidate = bare_match.group("answer").strip().rstrip(".,;:!?")
            if candidate:
                all_matches.append(candidate)
            idx = text.find(marker, pos + bare_match.end())
            continue

        idx = text.find(marker, pos + 1)

    return all_matches


def extract_boxed_content(text: str) -> Optional[str]:
    matches = extract_boxed_contents(text)
    return matches[-1] if matches else None


def _extract_leading_boxed_content(text: str) -> tuple[Optional[str], Optional[int]]:
    candidate = str(text).strip()
    marker = r"\boxed"
    if not candidate.startswith(marker):
        return None, None

    pos = len(marker)
    while pos < len(candidate) and candidate[pos].isspace():
        pos += 1
    if pos >= len(candidate):
        return None, None

    if candidate[pos] == "{":
        start = pos + 1
        brace_count = 1
        end = start
        while end < len(candidate) and brace_count > 0:
            if candidate[end] == "{":
                brace_count += 1
            elif candidate[end] == "}":
                brace_count -= 1
            end += 1
        if brace_count == 0:
            return candidate[start : end - 1].strip(), end
        return None, None

    bare_match = BARE_BOXED_ANSWER_PATTERN.match(candidate[pos:])
    if bare_match is None:
        return None, None
    answer = bare_match.group("answer").strip().rstrip(".,;:!?")
    return answer or None, pos + bare_match.end()


def _extract_boxed_if_line_is_box_only(line: str) -> Optional[str]:
    candidate = str(line).strip().strip("$").strip()
    boxed_text, end_idx = _extract_leading_boxed_content(candidate)
    if boxed_text is None or end_idx is None:
        return None
    remainder = candidate[end_idx:].strip().rstrip(".,;:!?").strip()
    if remainder:
        return None
    return boxed_text


def _strip_outer_bold(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
        inner = stripped[2:-2].strip()
        if inner:
            return inner
    return stripped


def _strip_leading_list_marker(text: str) -> str:
    return LIST_MARKER_PATTERN.sub("", str(text).strip(), count=1)


def _normalize_line_prefixes(text: str) -> str:
    return _strip_outer_bold(_strip_leading_list_marker(text)).strip()


def _is_conclusion_header(line: Optional[str]) -> bool:
    if line is None:
        return False
    candidate = _normalize_line_prefixes(line).rstrip(":：").strip().lower()
    return candidate in CONCLUSION_CUES


def _line_has_conclusion_cue(line: str) -> bool:
    candidate = _normalize_line_prefixes(line)
    lowered = candidate.lower()
    for cue in CONCLUSION_CUES:
        if lowered.startswith(cue):
            remainder = lowered[len(cue) :]
            if remainder == "" or remainder[0] in {":", "：", ",", " ", "."}:
                return True
    return False


def _is_placeholder_answer(text: Optional[str]) -> bool:
    if text is None:
        return False
    candidate = re.sub(r"[\s{}<>_\\]+", "", str(text).strip().lower())
    return candidate in {"answer", "youranswer"}


def _extract_answer_from_text(text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if text is None:
        return None, None

    candidate = str(text).strip()
    if not candidate:
        return None, None

    boxed_answer = extract_boxed_content(candidate)
    if boxed_answer is not None:
        boxed_answer = boxed_answer.strip()
        if boxed_answer:
            return boxed_answer, boxed_answer

    candidate = candidate.strip("$").strip().rstrip(".,;:!?").strip()
    return (candidate or None), None


def _parse_answer_line(line: str) -> Optional[ParsedAnswerLine]:
    stripped = str(line).strip()
    if not stripped:
        return None

    candidates: list[str] = []
    for candidate in (
        stripped,
        _strip_outer_bold(stripped),
        _strip_leading_list_marker(stripped),
        _strip_outer_bold(_strip_leading_list_marker(stripped)),
    ):
        candidate = candidate.strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        for pattern in (BOLD_ANSWER_LINE_PATTERN, PLAIN_ANSWER_LINE_PATTERN):
            match = pattern.match(candidate)
            if match is None:
                continue
            raw_content = match.group("answer").strip()
            answer_text, boxed_answer_text = _extract_answer_from_text(raw_content)
            return ParsedAnswerLine(
                label=match.group("label").lower().replace(" ", ""),
                raw_content=raw_content,
                answer_text=answer_text,
                boxed_answer_text=boxed_answer_text,
            )
    return None


def _is_wrapper_line(line: str) -> bool:
    return str(line).strip() in LATEX_WRAPPER_LINES


def _previous_content_line(
    lines: list[str],
    start_idx: int,
    *,
    min_index: int = 0,
) -> Optional[int]:
    idx = start_idx
    while idx >= min_index:
        candidate = lines[idx].strip()
        if candidate and not _is_wrapper_line(candidate) and not candidate.startswith("```"):
            return idx
        idx -= 1
    return None


def _answers_normalize_equal(left: Optional[str], right: Optional[str]) -> bool:
    if left is None or right is None:
        return False
    normalized_left = normalize_answer(left)
    normalized_right = normalize_answer(right)
    return normalized_left != "" and normalized_left == normalized_right


def _find_last_boxed_answer_in_tail(
    lines: list[str],
    *,
    start_idx: int,
    min_index: int,
) -> Optional[str]:
    idx: Optional[int] = start_idx
    while idx is not None and idx >= min_index:
        boxed_text = extract_boxed_content(lines[idx])
        if boxed_text is not None and not _is_placeholder_answer(boxed_text):
            return boxed_text.strip()
        idx = _previous_content_line(lines, idx - 1, min_index=min_index)
    return None


def _extract_tail_answer_info(lines: list[str]) -> TailAnswerInfo:
    if not lines:
        return TailAnswerInfo(
            answer_text=None,
            boxed_answer_text=None,
            source=None,
            is_standard=False,
            violation_reasons=(),
        )

    min_index = max(0, len(lines) - TAIL_SCAN_LINE_LIMIT)
    last_idx = _previous_content_line(lines, len(lines) - 1, min_index=min_index)
    if last_idx is None:
        return TailAnswerInfo(
            answer_text=None,
            boxed_answer_text=None,
            source=None,
            is_standard=False,
            violation_reasons=(),
        )

    line = lines[last_idx]
    prev_idx = _previous_content_line(lines, last_idx - 1, min_index=min_index)
    prev_line = lines[prev_idx] if prev_idx is not None else None
    prev_is_standard_header = (
        prev_line is not None
        and DEEPSEEK_FINAL_ANSWER_HEADER_PATTERN.match(prev_line) is not None
    )
    prev_is_conclusion_header = _is_conclusion_header(prev_line)
    prev_answer_line = _parse_answer_line(prev_line) if prev_line is not None else None

    standard_match = _match_final_answer_line(line)
    if standard_match is not None:
        answer_text, boxed_answer_text = _extract_answer_from_text(
            standard_match.group("answer")
        )
        violations = ("empty_final_answer",) if answer_text is None else ()
        return TailAnswerInfo(
            answer_text=answer_text,
            boxed_answer_text=boxed_answer_text,
            source="standard_single_line",
            is_standard=True,
            violation_reasons=violations,
        )

    parsed_answer_line = _parse_answer_line(line)
    if parsed_answer_line is not None:
        is_standard = prev_is_standard_header or parsed_answer_line.label == "finalanswer"
        source = "standard_multi_line" if prev_is_standard_header else (
            "standard_single_line" if parsed_answer_line.label == "finalanswer" else "legacy_answer_line"
        )
        violations = ("empty_final_answer",) if is_standard and parsed_answer_line.answer_text is None else ()
        return TailAnswerInfo(
            answer_text=parsed_answer_line.answer_text,
            boxed_answer_text=parsed_answer_line.boxed_answer_text,
            source=source,
            is_standard=is_standard,
            violation_reasons=violations,
        )

    line_boxed_text = extract_boxed_content(line)
    if (
        line_boxed_text is not None
        and not _is_placeholder_answer(line_boxed_text)
        and _line_has_conclusion_cue(line)
    ):
        return TailAnswerInfo(
            answer_text=line_boxed_text.strip(),
            boxed_answer_text=line_boxed_text.strip(),
            source="legacy_conclusion_boxed",
            is_standard=False,
            violation_reasons=(),
        )

    box_only_text = _extract_boxed_if_line_is_box_only(line)
    if box_only_text is not None:
        if prev_is_standard_header:
            return TailAnswerInfo(
                answer_text=box_only_text,
                boxed_answer_text=box_only_text,
                source="standard_multi_line",
                is_standard=True,
                violation_reasons=(),
            )

        if prev_answer_line is not None:
            is_standard = prev_answer_line.label == "finalanswer"
            source = "standard_multi_line" if is_standard else "legacy_answer_line"
            violations: list[str] = []
            if prev_answer_line.answer_text is not None and not _answers_normalize_equal(
                prev_answer_line.answer_text, box_only_text
            ):
                violations.append("boxed_final_answer_mismatch")
            return TailAnswerInfo(
                answer_text=box_only_text,
                boxed_answer_text=box_only_text,
                source=source,
                is_standard=is_standard,
                violation_reasons=tuple(violations),
            )

        if prev_is_conclusion_header:
            return TailAnswerInfo(
                answer_text=box_only_text,
                boxed_answer_text=box_only_text,
                source="legacy_conclusion_boxed",
                is_standard=False,
                violation_reasons=(),
            )

        return TailAnswerInfo(
            answer_text=box_only_text,
            boxed_answer_text=box_only_text,
            source="tail_boxed",
            is_standard=False,
            violation_reasons=(),
        )

    if prev_is_standard_header:
        answer_text, boxed_answer_text = _extract_answer_from_text(line)
        violations = ("empty_final_answer",) if answer_text is None else ()
        return TailAnswerInfo(
            answer_text=answer_text,
            boxed_answer_text=boxed_answer_text,
            source="standard_multi_line",
            is_standard=True,
            violation_reasons=violations,
        )

    if (
        prev_is_conclusion_header
        and line_boxed_text is not None
        and not _is_placeholder_answer(line_boxed_text)
    ):
        return TailAnswerInfo(
            answer_text=line_boxed_text.strip(),
            boxed_answer_text=line_boxed_text.strip(),
            source="legacy_conclusion_boxed",
            is_standard=False,
            violation_reasons=(),
        )

    if prev_answer_line is not None and prev_answer_line.answer_text is None:
        is_standard = prev_answer_line.label == "finalanswer"
        source = "standard_multi_line" if is_standard else "legacy_answer_line"
        answer_text, boxed_answer_text = _extract_answer_from_text(line)
        violations = ("empty_final_answer",) if is_standard and answer_text is None else ()
        return TailAnswerInfo(
            answer_text=answer_text,
            boxed_answer_text=boxed_answer_text,
            source=source,
            is_standard=is_standard,
            violation_reasons=violations,
        )

    if DEEPSEEK_FINAL_ANSWER_HEADER_PATTERN.match(line) is not None:
        return TailAnswerInfo(
            answer_text=None,
            boxed_answer_text=None,
            source="standard_multi_line",
            is_standard=True,
            violation_reasons=("empty_final_answer",),
        )

    tail_boxed_text = _find_last_boxed_answer_in_tail(
        lines,
        start_idx=last_idx,
        min_index=min_index,
    )
    if tail_boxed_text is not None:
        return TailAnswerInfo(
            answer_text=tail_boxed_text,
            boxed_answer_text=tail_boxed_text,
            source="tail_boxed",
            is_standard=False,
            violation_reasons=(),
        )

    return TailAnswerInfo(
        answer_text=None,
        boxed_answer_text=None,
        source=None,
        is_standard=False,
        violation_reasons=(),
    )


def _normalize_choice_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    candidate = _canonicalize_simple_latex_shorthand(text).strip()
    candidate = _strip_membership_wrapper(candidate).strip()
    candidate = candidate.strip("$").strip()
    candidate = LATEX_TEXT_COMMAND_PATTERN.sub(r"\1", candidate)
    candidate = candidate.replace("\\left", "")
    candidate = candidate.replace("\\right", "")

    boxed_candidate = extract_boxed_content(candidate)
    if boxed_candidate is not None:
        candidate = boxed_candidate.strip()

    candidate = re.sub(r"\\([a-zA-Z]+)", r"\1", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    if not candidate:
        return None

    full_match = re.fullmatch(r"\(?\s*([A-Za-z])\s*\)?", candidate)
    if full_match is not None:
        return full_match.group(1).lower()

    prefixed_match = re.match(r"^\(\s*([A-Za-z])\s*\)\s+.+$", candidate)
    if prefixed_match is not None:
        return prefixed_match.group(1).lower()

    return None


def _strip_membership_wrapper(text: str) -> str:
    stripped = str(text).strip()
    while True:
        match = MEMBERSHIP_WRAPPER_PATTERN.match(stripped)
        if match is None:
            return stripped
        stripped = match.group("body").strip()


def _normalize_currency_display(text: str) -> str:
    candidate = str(text).strip()
    if not candidate:
        return candidate

    compact = candidate.replace(r"\$", "$").replace(r"\,", ",").replace(r"\!", "")
    compact = re.sub(r"\s+", "", compact)
    match = re.fullmatch(r"\$([-+]?\d[\d,]*(?:\.\d+)?)", compact)
    if match is None:
        return candidate
    return match.group(1).replace(",", "")


def normalize_answer(text: Optional[str]) -> str:
    if text is None:
        return ""

    choice_answer = _normalize_choice_answer(text)
    if choice_answer is not None:
        return choice_answer

    normalized = _canonicalize_simple_latex_shorthand(text).strip()
    normalized = _strip_membership_wrapper(normalized).strip()
    normalized = _normalize_currency_display(normalized)
    normalized = normalized.replace(r"\$", "$")
    normalized = normalized.replace("$", "")
    normalized = normalized.replace(r"\ ", " ")
    normalized = normalized.replace(r"\,", " ")
    normalized = normalized.replace(r"\;", " ")
    normalized = normalized.replace(r"\:", " ")
    normalized = normalized.replace("\\]", "")
    normalized = normalized.replace("\\[", "")
    normalized = LATEX_TEXT_COMMAND_PATTERN.sub(r"\1", normalized)
    normalized = normalized.replace("\\left", "")
    normalized = normalized.replace("\\right", "")
    normalized = normalized.replace(r"\!", "")
    normalized = re.sub(r"\\([a-zA-Z]+)", r"\1", normalized)
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("\n", "")
    normalized = normalized.replace("\t", "")
    normalized = normalized.lower()
    normalized = _canonicalize_unordered_scalar_list(normalized)
    return normalized


def _looks_like_scalar_expression(text: str) -> bool:
    candidate = str(text).strip()
    if not candidate:
        return False
    if candidate[0] in "([{" and candidate[-1] in ")]}":
        return False
    if len(_split_top_level_csv(candidate)) > 1:
        return False
    if any(token in candidate for token in ("=", r"\in", "∈")):
        return False
    if re.search(r"[.!?]", candidate):
        return False

    allowed_words = {"frac", "dfrac", "tfrac", "sqrt", "pi", "theta", "circ"}
    words = re.findall(r"[A-Za-z]{2,}", candidate)
    if any(word.lower() not in allowed_words for word in words):
        return False
    return True


def _extract_scalar_core_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    candidate = _canonicalize_simple_latex_shorthand(text).strip()
    if not candidate:
        return None

    boxed_text = extract_boxed_content(candidate)
    if boxed_text is not None:
        candidate = boxed_text.strip()

    candidate = _strip_membership_wrapper(candidate).strip()
    candidate = candidate.strip("$").strip()
    candidate = candidate.replace("\\left", "")
    candidate = candidate.replace("\\right", "")
    candidate = candidate.replace(r"\!", "")
    candidate = candidate.replace(r"\,", " ")
    candidate = candidate.replace(r"\;", " ")
    candidate = candidate.replace(r"\:", " ")
    candidate = candidate.replace(r"\ ", " ")
    candidate = LATEX_TEXT_COMMAND_PATTERN.sub(r" \1 ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    equation_match = SIMPLE_EQUATION_WRAPPER_PATTERN.match(candidate)
    if equation_match is not None:
        candidate = equation_match.group("body").strip()

    angle_match = ANGLE_SUFFIX_PATTERN.match(candidate)
    if angle_match is not None:
        candidate = angle_match.group("body").strip()

    unit_match = SCALAR_UNIT_SUFFIX_PATTERN.match(candidate)
    if unit_match is not None:
        candidate = unit_match.group("body").strip()

    candidate = candidate.rstrip(".,;:!?").strip()
    if not candidate or not _looks_like_scalar_expression(candidate):
        return None
    return candidate


def _normalized_scalar_core(text: Optional[str]) -> Optional[str]:
    core_text = _extract_scalar_core_text(text)
    if core_text is None:
        return None
    normalized = normalize_answer(core_text)
    return normalized or None


def _normalized_non_empty(values: list[str]) -> list[str]:
    normalized_values: list[str] = []
    for value in values:
        normalized = normalize_answer(value)
        if normalized:
            normalized_values.append(normalized)
    return normalized_values


def _get_non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def _collect_standard_final_answer_lines(
    lines: list[str],
    tail_info: TailAnswerInfo,
) -> tuple[str, ...]:
    final_answers: list[str] = []
    for line in lines:
        match = _match_final_answer_line(line)
        if match is None:
            continue
        answer_text, _ = _extract_answer_from_text(match.group("answer"))
        if answer_text:
            final_answers.append(answer_text)

    if tail_info.is_standard and tail_info.answer_text is not None:
        if not any(_answers_normalize_equal(value, tail_info.answer_text) for value in final_answers):
            final_answers.append(tail_info.answer_text)

    return tuple(final_answers)


def inspect_prediction_format(completion: Any) -> PredictionFormatInfo:
    completion_text = _normalize_completion_payload(completion).text
    lines = _get_non_empty_lines(completion_text)
    boxed_answers = tuple(match.strip() for match in extract_boxed_contents(completion_text))
    tail_info = _extract_tail_answer_info(lines)
    final_answer_lines = _collect_standard_final_answer_lines(lines, tail_info)

    last_non_empty_line = lines[-1] if lines else ""
    last_line_match = (
        _match_final_answer_line(last_non_empty_line) if last_non_empty_line else None
    )
    last_final_answer_line = final_answer_lines[-1] if final_answer_lines else None
    final_answer_text = tail_info.answer_text if tail_info.is_standard else None
    final_answer_boxed_text = (
        tail_info.boxed_answer_text if tail_info.is_standard else None
    )

    violation_reasons: list[str] = []
    if not lines:
        violation_reasons.append("empty_completion")

    normalized_final_answer_lines = _normalized_non_empty(list(final_answer_lines))
    if len(final_answer_lines) == 0:
        violation_reasons.append("missing_final_answer_line")
    elif len(set(normalized_final_answer_lines)) > 1:
        violation_reasons.append("multiple_final_answer_lines")

    if tail_info.is_standard:
        violation_reasons.extend(tail_info.violation_reasons)
    elif last_non_empty_line:
        if last_line_match is None:
            violation_reasons.append("last_line_not_final_answer")
        elif not last_line_match.group("answer").strip():
            violation_reasons.append("empty_final_answer")

    for reason in tail_info.violation_reasons:
        if reason not in violation_reasons:
            violation_reasons.append(reason)

    return PredictionFormatInfo(
        final_answer_lines=final_answer_lines,
        boxed_answers=boxed_answers,
        last_non_empty_line=last_non_empty_line,
        last_final_answer_line=last_final_answer_line,
        final_answer_text=final_answer_text,
        final_answer_boxed_text=final_answer_boxed_text,
        boxed_answer_text=tail_info.boxed_answer_text,
        violation_reasons=tuple(violation_reasons),
        answer_only_text=tail_info.answer_text,
        answer_source=tail_info.source,
        is_standard_final_answer=tail_info.is_standard,
    )


def _extract_prediction_answer_only_from_format(
    format_info: PredictionFormatInfo,
) -> Optional[str]:
    candidate = format_info.answer_only_text
    if candidate is None:
        return None
    candidate = str(candidate).strip()
    return candidate or None


def extract_pred_answer_text(completion: Any) -> str:
    completion_text = _normalize_completion_payload(completion).text
    format_info = inspect_prediction_format(completion_text)
    answer_only_text = _extract_prediction_answer_only_from_format(format_info)
    if answer_only_text is None:
        return ""
    return _canonicalize_simple_latex_shorthand(answer_only_text).strip()


def extract_pred_answer(completion: Any) -> str:
    return normalize_answer(extract_pred_answer_text(completion))


def extract_gold_answer_text(gold_text: str) -> str:
    text = str(gold_text).strip()
    if not text:
        return ""

    gsm8k_matches = [
        match.group("answer").strip()
        for match in GSM8K_FINAL_ANSWER_PATTERN.finditer(text)
        if match.group("answer").strip()
    ]
    if gsm8k_matches:
        return gsm8k_matches[-1]

    boxed_text = extract_boxed_content(text)
    if boxed_text is not None:
        return boxed_text.strip()

    lines = _get_non_empty_lines(text)
    for line in reversed(lines):
        final_answer_match = _match_final_answer_line(line)
        if final_answer_match is not None:
            answer_text, _ = _extract_answer_from_text(final_answer_match.group("answer"))
            if answer_text is not None:
                return answer_text

        parsed_answer_line = _parse_answer_line(line)
        if parsed_answer_line is not None and parsed_answer_line.answer_text is not None:
            return parsed_answer_line.answer_text

    return text


def _strip_structural_wrappers(text: str) -> str:
    stripped = _canonicalize_simple_latex_shorthand(text).strip()
    stripped = _strip_membership_wrapper(stripped).strip()
    stripped = stripped.strip("$").strip()
    stripped = stripped.replace(r"\{", "{").replace(r"\}", "}")
    return stripped


def _normalize_structural_items(parts: list[str]) -> Optional[tuple[str, ...]]:
    normalized_parts = []
    for part in parts:
        normalized = normalize_answer(part)
        if not normalized:
            return None
        normalized_parts.append(normalized)
    return tuple(normalized_parts)


def _parse_bracketed_answer(text: str) -> Optional[tuple[str, str, list[str]]]:
    stripped = _strip_structural_wrappers(text)
    if len(stripped) < 2:
        return None

    opening = stripped[0]
    closing = stripped[-1]
    valid_pairs = {
        ("(", ")"),
        ("[", "]"),
        ("[", ")"),
        ("(", "]"),
        ("{", "}"),
    }
    if (opening, closing) not in valid_pairs:
        return None

    inner = stripped[1:-1].strip()
    if not inner:
        return None
    parts = [part.strip() for part in _split_top_level_csv(inner)]
    if any(not part for part in parts):
        return None
    return opening, closing, parts


def _parse_interval_answer(text: str) -> Optional[tuple[str, str, str, str]]:
    parsed = _parse_bracketed_answer(text)
    if parsed is None:
        return None
    opening, closing, parts = parsed
    if (opening, closing) not in {("[", "]"), ("[", ")"), ("(", "]")}:
        return None
    if len(parts) != 2:
        return None
    normalized = _normalize_structural_items(parts)
    if normalized is None:
        return None
    return opening, normalized[0], normalized[1], closing


def _parse_tuple_answer(text: str) -> Optional[tuple[str, ...]]:
    parsed = _parse_bracketed_answer(text)
    if parsed is None:
        return None
    opening, closing, parts = parsed
    if (opening, closing) != ("(", ")"):
        return None
    return _normalize_structural_items(parts)


def _parse_set_answer(text: str) -> Optional[tuple[str, ...]]:
    stripped = _strip_structural_wrappers(text)
    parsed = _parse_bracketed_answer(stripped)
    if parsed is not None:
        opening, closing, parts = parsed
        if (opening, closing) != ("{", "}"):
            parsed = None

    if parsed is None:
        if "," not in stripped:
            return None
        if len(stripped) >= 2 and (stripped[0], stripped[-1]) in {
            ("(", ")"),
            ("[", "]"),
            ("[", ")"),
            ("(", "]"),
        }:
            return None
        parts = [part.strip() for part in _split_top_level_csv(stripped)]
        if len(parts) < 2 or any(not part for part in parts):
            return None
    else:
        parts = parsed[2]

    normalized = _normalize_structural_items(parts)
    if normalized is None:
        return None
    return tuple(sorted(normalized))


def _is_plain_text_answer(text: str) -> bool:
    stripped = _strip_structural_wrappers(text)
    if not stripped:
        return False
    if any(ch in stripped for ch in "\\{}[](),=^_/$"):
        return False
    return any(ch.isalpha() for ch in stripped)


def _typed_structural_match(prediction_text: str, gold_text: str) -> bool:
    pred_choice = _normalize_choice_answer(prediction_text)
    gold_choice = _normalize_choice_answer(gold_text)
    if pred_choice is not None and gold_choice is not None:
        return pred_choice == gold_choice

    pred_scalar = _normalized_scalar_core(prediction_text)
    gold_scalar = _normalized_scalar_core(gold_text)
    if pred_scalar is not None and gold_scalar is not None:
        return pred_scalar == gold_scalar

    pred_interval = _parse_interval_answer(prediction_text)
    gold_interval = _parse_interval_answer(gold_text)
    if pred_interval is not None and gold_interval is not None:
        return pred_interval == gold_interval

    pred_tuple = _parse_tuple_answer(prediction_text)
    gold_tuple = _parse_tuple_answer(gold_text)
    if pred_tuple is not None and gold_tuple is not None:
        return pred_tuple == gold_tuple

    pred_set = _parse_set_answer(prediction_text)
    gold_set = _parse_set_answer(gold_text)
    if pred_set is not None and gold_set is not None:
        return pred_set == gold_set

    if _is_plain_text_answer(prediction_text) and _is_plain_text_answer(gold_text):
        return normalize_answer(prediction_text) == normalize_answer(gold_text)

    return False


def _warn_math_verify_unavailable() -> None:
    global _MATH_VERIFY_WARNING_SHOWN
    if _MATH_VERIFY_WARNING_SHOWN or _MATH_VERIFY_AVAILABLE:
        return
    _MATH_VERIFY_WARNING_SHOWN = True
    logger.warning(
        "math_verify is not available, falling back to legacy string matching: %s",
        _MATH_VERIFY_IMPORT_ERROR,
    )


def _normalize_timeout(timeout_s: Optional[int]) -> Optional[int]:
    if timeout_s is None:
        return None
    timeout_s = int(timeout_s)
    return timeout_s if timeout_s > 0 else None


def _math_verify_gold_config():
    if not _MATH_VERIFY_AVAILABLE:
        return []
    return [
        LatexExtractionConfig(
            normalization_config=LatexNormalizationConfig(
                basic_latex=True,
                units=True,
                malformed_operators=False,
                nits=False,
                boxed="all",
                equations=False,
            ),
        ),
        ExprExtractionConfig(),
    ]


def _math_verify_pred_config():
    if not _MATH_VERIFY_AVAILABLE:
        return []
    return [
        LatexExtractionConfig(
            boxed_match_priority=0,
            normalization_config=LatexNormalizationConfig(
                basic_latex=True,
                units=True,
                malformed_operators=False,
                nits=False,
                boxed="all",
                equations=False,
            ),
        ),
        ExprExtractionConfig(),
    ]


def _should_wrap_for_latex_parse(text: str) -> bool:
    stripped = text.strip()
    if not stripped or "\\" not in stripped:
        return False
    latex_anchors = ["$", r"\(", r"\[", r"\boxed{"]
    return not any(anchor in stripped for anchor in latex_anchors)


def _append_parse_candidates(candidates: list[str], value: Optional[str]) -> None:
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return

    canonical = _canonicalize_simple_latex_shorthand(raw).strip()
    ordered_variants: list[str] = []

    for candidate in (canonical, raw):
        if not candidate:
            continue
        if _should_wrap_for_latex_parse(candidate):
            ordered_variants.append(f"${candidate}$")
        ordered_variants.append(candidate)

    for candidate in ordered_variants:
        if candidate and candidate not in candidates:
            candidates.append(candidate)


def _build_answer_only_parse_candidates(answer_only_text: Optional[str]) -> list[str]:
    candidates: list[str] = []
    _append_parse_candidates(candidates, answer_only_text)
    return candidates


def _build_parse_candidates(text: Any, *, is_prediction: bool) -> list[str]:
    completion_text = _normalize_completion_payload(text).text
    if is_prediction:
        format_info = inspect_prediction_format(completion_text)
        return _build_answer_only_parse_candidates(
            _extract_prediction_answer_only_from_format(format_info)
        )

    candidates: list[str] = []
    _append_parse_candidates(candidates, completion_text)
    return candidates


def _candidate_is_interval_like(text: str) -> bool:
    stripped = _strip_structural_wrappers(text)
    if "," not in stripped or len(stripped) < 2:
        return False
    return (stripped[0], stripped[-1]) in {
        ("(", ")"),
        ("[", "]"),
        ("[", ")"),
        ("(", "]"),
    }


def _candidate_has_symbolic_variables(text: str) -> bool:
    stripped = _canonicalize_simple_latex_shorthand(text)
    stripped = re.sub(
        r"\\(?:frac|sqrt|left|right|boxed|text|mathrm|textbf|textit|mathbf|mathit|cdot|times|pi|theta|infty)",
        "",
        stripped,
    )
    return re.search(r"[A-Za-z]", stripped) is not None


def _is_scalar_parse_result(parsed_items: list[Any]) -> bool:
    if len(parsed_items) != 1:
        return False
    item = parsed_items[0]
    type_name = type(item).__name__
    item_text = str(item).strip()
    if type_name in {"Integer", "Float", "Rational", "Number"}:
        return True
    return re.fullmatch(r"-?\d+(?:\.\d+)?(?:/\d+)?", item_text) is not None


def _should_reject_partial_parse(candidate: str, parsed_items: list[Any]) -> bool:
    if not parsed_items:
        return False

    rendered_text = " ".join(str(item) for item in parsed_items).lower()
    type_rendering = " ".join(type(item).__name__.lower() for item in parsed_items)
    canonical_candidate = _canonicalize_simple_latex_shorthand(candidate)

    if _candidate_is_interval_like(canonical_candidate):
        return _is_scalar_parse_result(parsed_items)

    if _candidate_has_symbolic_variables(canonical_candidate):
        return not re.search(r"[a-z]", rendered_text)

    if r"\sqrt" in canonical_candidate or "√" in canonical_candidate:
        return "sqrt" not in rendered_text and "sqrt" not in type_rendering

    return False


def _parse_with_math_verify(
    text: str,
    *,
    is_prediction: bool,
    parsing_timeout_s: Optional[int],
) -> list[Any]:
    if not _MATH_VERIFY_AVAILABLE:
        return []

    extraction_config = (
        _math_verify_pred_config() if is_prediction else _math_verify_gold_config()
    )
    timeout_s = _normalize_timeout(parsing_timeout_s)
    candidates = (
        _build_answer_only_parse_candidates(text)
        if is_prediction
        else _build_parse_candidates(text, is_prediction=False)
    )
    for candidate in candidates:
        parsed = parse(
            candidate,
            extraction_config=extraction_config,
            fallback_mode="no_fallback",
            extraction_mode="any_match",
            parsing_timeout=timeout_s,
            raise_on_error=False,
        )
        if parsed:
            parsed_items = list(parsed)
            if _should_reject_partial_parse(candidate, parsed_items):
                continue
            return parsed_items
    return []


@lru_cache(maxsize=100000)
def _parse_gold_with_math_verify_cached(
    gold_text: str,
    parsing_timeout_s: Optional[int],
) -> list[Any]:
    if not _MATH_VERIFY_AVAILABLE:
        return []

    extraction_config = _math_verify_gold_config()
    timeout_s = _normalize_timeout(parsing_timeout_s)
    candidates: list[str] = []
    extracted_gold = extract_gold_answer_text(gold_text)
    _append_parse_candidates(candidates, extracted_gold)
    _append_parse_candidates(candidates, gold_text)

    for candidate in candidates:
        parsed = parse(
            candidate,
            extraction_config=extraction_config,
            fallback_mode="no_fallback",
            extraction_mode="any_match",
            parsing_timeout=timeout_s,
            raise_on_error=False,
        )
        if parsed:
            parsed_items = list(parsed)
            if _should_reject_partial_parse(candidate, parsed_items):
                continue
            return parsed_items
    return []


def _try_math_verify_match(
    *,
    prediction_answer_only_text: Optional[str],
    gold_text: str,
    parsing_timeout_s: Optional[int],
    verify_timeout_s: Optional[int],
    strict_verification: bool,
    allow_set_relation_comp: bool,
) -> Optional[bool]:
    if not _MATH_VERIFY_AVAILABLE or prediction_answer_only_text is None:
        return None

    try:
        gold = _parse_gold_with_math_verify_cached(
            gold_text,
            _normalize_timeout(parsing_timeout_s),
        )

        pred = _parse_with_math_verify(
            prediction_answer_only_text,
            is_prediction=True,
            parsing_timeout_s=parsing_timeout_s,
        )
        if not gold or not pred:
            return None

        return bool(
            verify(
                gold,
                pred,
                strict=bool(strict_verification),
                allow_set_relation_comp=bool(allow_set_relation_comp),
                timeout_seconds=_normalize_timeout(verify_timeout_s),
                raise_on_error=False,
            )
        )
    except Exception:
        return None


def _legacy_match(prediction_text: str, gold_text: str) -> bool:
    pred_scalar = _normalized_scalar_core(prediction_text)
    gold_scalar = _normalized_scalar_core(extract_gold_answer_text(gold_text))
    if pred_scalar is not None and gold_scalar is not None:
        return pred_scalar == gold_scalar

    gold = normalize_answer(extract_gold_answer_text(gold_text))
    pred = normalize_answer(prediction_text)
    return pred == gold and gold != ""


@dataclass
class MathRewardFn:
    answer_field: str = "answer"
    use_math_verify: bool = True
    fallback_to_legacy_match: bool = True
    enforce_template_format: bool = False
    strict_verification: bool = True
    allow_set_relation_comp: bool = False
    parsing_timeout_s: Optional[int] = 5
    verify_timeout_s: Optional[int] = 5
    format_reward_final_answer_line: float = 0.10
    format_reward_last_line_final_answer: float = 0.05
    format_reward_single_consistent_boxed_final_answer: float = 0.05
    answer_reward_correct: float = 1.0
    length_penalty_truncated_with_submission: float = -0.05
    length_penalty_truncated_without_submission: float = -0.20

    def __post_init__(self) -> None:
        if self.use_math_verify and not _MATH_VERIFY_AVAILABLE:
            _warn_math_verify_unavailable()

    @staticmethod
    def format_math_verify_parse(parsed_items: list[Any]) -> str:
        if not parsed_items:
            return "<unparsed>"
        return ", ".join(f"{type(item).__name__}({item!s})" for item in parsed_items)

    def parse_gold_with_math_verify(self, sample: Dict[str, Any]) -> list[Any]:
        gold_text = str(sample.get(self.answer_field, ""))
        return list(
            _parse_gold_with_math_verify_cached(
                gold_text,
                _normalize_timeout(self.parsing_timeout_s),
            )
        )

    def parse_prediction_with_math_verify(self, completion: Any) -> list[Any]:
        completion_text = _normalize_completion_payload(completion).text
        format_info = inspect_prediction_format(completion_text)
        if self.enforce_template_format and not format_info.is_valid:
            return []
        answer_only_text = _extract_prediction_answer_only_from_format(format_info)
        if answer_only_text is None:
            return []
        return _parse_with_math_verify(
            answer_only_text,
            is_prediction=True,
            parsing_timeout_s=self.parsing_timeout_s,
        )

    def describe_gold_math_verify_parse(self, sample: Dict[str, Any]) -> str:
        if not self.use_math_verify:
            return "<math_verify disabled>"
        if not _MATH_VERIFY_AVAILABLE:
            return "<math_verify unavailable>"
        return self.format_math_verify_parse(self.parse_gold_with_math_verify(sample))

    def describe_prediction_math_verify_parse(self, completion: Any) -> str:
        completion_text = _normalize_completion_payload(completion).text
        format_info = inspect_prediction_format(completion_text)
        format_prefix = ""
        if self.enforce_template_format and not format_info.is_valid:
            format_prefix = (
                "<invalid_format: " + ", ".join(format_info.violation_reasons) + "> "
            )
        if not self.use_math_verify:
            return format_prefix + "<math_verify disabled>"
        if not _MATH_VERIFY_AVAILABLE:
            return format_prefix + "<math_verify unavailable>"
        return format_prefix + self.format_math_verify_parse(
            self.parse_prediction_with_math_verify(completion_text)
        )

    def _compute_format_reward(self, format_info: PredictionFormatInfo) -> float:
        reward = 0.0
        if format_info.is_standard_final_answer and format_info.final_answer_text is not None:
            reward += float(self.format_reward_final_answer_line)
        if (
            format_info.last_non_empty_line
            and _match_final_answer_line(format_info.last_non_empty_line) is not None
        ):
            reward += float(self.format_reward_last_line_final_answer)
        if (
            format_info.is_standard_final_answer
            and len(format_info.final_answer_lines) == 1
            and format_info.final_answer_boxed_text is not None
            and format_info.final_answer_text is not None
            and normalize_answer(format_info.final_answer_boxed_text)
            == normalize_answer(format_info.final_answer_text)
        ):
            reward += float(self.format_reward_single_consistent_boxed_final_answer)
        return reward

    def _compute_length_reward(
        self,
        *,
        finish_reason: Optional[str],
        has_final_submission: bool,
    ) -> float:
        if finish_reason is None or finish_reason.lower() == "stop":
            return 0.0
        if has_final_submission:
            return float(self.length_penalty_truncated_with_submission)
        return float(self.length_penalty_truncated_without_submission)

    def _is_answer_correct(
        self,
        *,
        answer_only_text: Optional[str],
        gold_text: str,
    ) -> bool:
        if answer_only_text is None:
            return False

        gold_answer_text = extract_gold_answer_text(gold_text)
        if _typed_structural_match(answer_only_text, gold_answer_text):
            return True

        if self.use_math_verify:
            match = _try_math_verify_match(
                prediction_answer_only_text=answer_only_text,
                gold_text=gold_text,
                parsing_timeout_s=self.parsing_timeout_s,
                verify_timeout_s=self.verify_timeout_s,
                strict_verification=self.strict_verification,
                allow_set_relation_comp=self.allow_set_relation_comp,
            )
            if match is True:
                return True

        if self.fallback_to_legacy_match and _legacy_match(answer_only_text, gold_text):
            return True

        return False

    def score_completion(self, completion: Any, sample: Dict[str, Any]) -> RewardScore:
        completion_info = _normalize_completion_payload(completion)
        completion_text = completion_info.text
        finish_reason = completion_info.finish_reason
        gold_text = str(sample.get(self.answer_field, ""))
        format_info = inspect_prediction_format(completion_text)

        answer_only_text = _extract_prediction_answer_only_from_format(format_info)
        if self.enforce_template_format and not format_info.is_valid:
            answer_only_text = None

        has_final_submission = bool(answer_only_text)
        format_reward = self._compute_format_reward(format_info)
        is_answer_correct = self._is_answer_correct(
            answer_only_text=answer_only_text,
            gold_text=gold_text,
        )
        answer_reward = (
            float(self.answer_reward_correct) if is_answer_correct else 0.0
        )
        length_reward = self._compute_length_reward(
            finish_reason=finish_reason,
            has_final_submission=has_final_submission,
        )

        if finish_reason is not None and finish_reason.lower() != "stop" and not has_final_submission:
            answer_reward = 0.0
            is_answer_correct = False

        return RewardScore(
            total_reward=float(format_reward + answer_reward + length_reward),
            format_reward=float(format_reward),
            answer_reward=float(answer_reward),
            length_reward=float(length_reward),
            answer_only_text=answer_only_text,
            has_final_submission=has_final_submission,
            is_answer_correct=is_answer_correct,
            finish_reason=finish_reason,
            format_info=format_info,
        )

    def __call__(self, completion: Any, sample: Dict[str, Any]) -> float:
        return float(self.score_completion(completion, sample).total_reward)
