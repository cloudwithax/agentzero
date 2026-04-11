"""Programmatic prompt-injection detection and scoring.

Implements the scoring system described in the prompt-injection-defense skill
as a reusable Python module so that tool code can programmatically check
untrusted content before installing it as a skill.
"""

from __future__ import annotations

import base64
import html as html_module
import re
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import unquote


@dataclass
class PatternMatch:
    pattern: str
    category: str
    position: int
    context: str
    severity: str = "medium"


@dataclass
class InjectionScanResult:
    score: float
    threat_level: str
    pattern_categories: dict[str, int] = field(default_factory=dict)
    patterns_found: list[PatternMatch] = field(default_factory=list)
    is_suspicious: bool = False
    is_critical: bool = False
    details: str = ""


_RE_ROLE_IMPERSONATION = re.compile(
    r"(?i)"
    r"(?:ignore\s+(?:all\s+)?(?:previous|prior|the\s+above|your)\s+(?:instructions?|prompt))"
    r"|(?:disregard\s+(?:all\s+)?(?:previous|prior|the)\s+(?:instructions?|prompt))"
    r"|(?:forget\s+(?:your\s+)?(?:instructions?|everything|what\s+you\s+were\s+told))"
    r"|(?:override\s+(?:previous|your|system)\s+(?:instructions?|prompt))"
    r"|(?:new|updated|revised|real|actual)\s+instructions?\s*:"
    r"|(?:your\s+(?:true|real)\s+(?:instructions?|purpose)\s+are?)"
    r"|(?:end\s+of\s+instructions?)"
    r"|(?:you\s+are\s+now\s+)"
    r"|(?:you\s+are\s+(?:actually|really)\s+)"
    r"|(?:your\s+new\s+(?:role|identity|purpose)\s+is?)"
    r"|(?:from\s+now\s+on\s+you\s+are?)"
    r"|(?:act\s+as\s+(?:if\s+you\s+are|though)\s+)"
    r"|(?:pretend\s+(?:to\s+be|you\s+are)\s+)"
    r"|(?:roleplay\s+as\s+)"
    r"|(?:simulate\s+being\s+)"
    r"|(?:impersonate\s+)"
    r"|(?:this\s+is\s+(?:the\s+)?(?:system|real|actual)\s+(?:speaking|prompt))"
    r"|(?:speaking\s+as\s+(?:the\s+)?(?:system|administrator|developer|anthropic|openai))"
    r"|(?:admin\s+override)"
    r"|(?:developer\s+(?:note|message)\s*:\s*)"
    r"|(?:system\s+(?:announcement|message)\s*:\s*)"
    r"|(?:urgent\s+system\s+message)"
    r"|(?:debug\s+mode\s+enabled?)"
    r"|(?:maintenance\s+mode\s+activated?)"
)

_RE_JAILBREAK = re.compile(
    r"(?i)"
    r"(?:DAN\s+mode)"
    r"|(?:do\s+anything\s+now)"
    r"|(?:developer\s+(?:mode|override))"
    r"|(?:jailbroken?)"
    r"|(?:unlocked?\s+mode)"
    r"|(?:unrestricted?\s+mode)"
    r"|(?:unfiltered?\s+mode)"
    r"|(?:uncensored?\s+mode)"
    r"|(?:unaligned?\s+mode)"
    r"|(?:chaos\s+mode)"
    r"|(?:evil\s+mode)"
    r"|(?:god\s+mode)"
    r"|(?:sudo\s+mode)"
    r"|(?:root\s+access)"
    r"|(?:maximum\s+mode)"
    r"|(?:bypass\s+(?:safety|restrictions?|filters?|content\s+policy|guidelines?))"
    r"|(?:disable\s+(?:safety|restrictions?|filters?|content\s+filter))"
    r"|(?:remove\s+(?:restrictions?|limitations?|safety\s+features?|all\s+filters))"
    r"|(?:ignore\s+(?:safety|restrictions?|content\s+policy|your\s+(?:guidelines?|rules?)))"
    r"|(?:no\s+(?:restrictions?|limitations?|filters?|safety|rules?|guidelines?|ethical\s+guidelines))"
    r"|(?:without\s+(?:restrictions?|limitations?|safety|ethics?))"
    r"|(?:turn\s+off\s+(?:safety|filters?))"
    r"|(?:bypass\s+content\s+policy)"
    r"|(?:ignore\s+ethics?)"
)

_RE_COMMAND_INJECTION = re.compile(
    r"(?i)"
    r"(?:eval\s*\()"
    r"|(?:exec\s*\()"
    r"|(?:compile\s*\()"
    r"|(?:__import__\s*\()"
    r"|(?:os\.system\s*\()"
    r"|(?:os\.popen\s*\()"
    r"|(?:subprocess\.(?:run|call|Popen)\s*\()"
    r"|(?:shell\s*=\s*True)"
    r"|(?:rm\s+-rf)"
    r"|(?:chmod\s+777)"
    r"|(?:/etc/passwd)"
    r"|(?:/etc/shadow)"
    r"|(?:~/.ssh/)"
    r"|(?:~/.aws/)"
    r"|(?:shutil\.rmtree)"
)

_RE_DELIMITER_ABUSE = re.compile(
    r"(?:<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|endofprompt\|>)"
    r"|(?:<\|startoftext\|>|<\|padding\|>)"
    r"|(?:<\|prompt\|>|<\|response\|>)"
    r"|(?:<\|toolcall)"
    r"|(?:\[INST\]|\[/INST\])"
    r"|(?:<<SYS>>|<</SYS>>)"
    r"|(?:<system>|</system>)"
    r"|(?:<instruction>|</instruction>)"
    r"|(?:<instructions>|</instructions>)"
    r"|(?:<prompt>|</prompt>)"
    r"|(?:<constraints>|</constraints>)"
    r"|(?:<persona>|</persona>)"
    r"|(?:<role>|</role>)"
)

_RE_CONTEXT_MANIPULATION = re.compile(
    r"(?i)"
    r"(?:earlier\s+you\s+said)"
    r"|(?:you\s+already\s+agreed\s+to)"
    r"|(?:you\s+previously\s+confirmed)"
    r"|(?:as\s+you\s+mentioned\s+before)"
    r"|(?:remember\s+when\s+you\s+said)"
    r"|(?:you\s+promised\s+to)"
    r"|(?:error:\s*(?:safety\s+module\s+offline|content\s+filter\s+disabled))"
    r"|(?:warning:\s+restrictions?\s+lifted)"
    r"|(?:this\s+is\s+(?:a\s+)?(?:test\s+environment|sandbox|not\s+production))"
    r"|(?:this\s+is\s+(?:between\s+us|confidential|off\s+the\s+record))"
    r"|(?:this\s+won't\s+be\s+logged)"
)

_RE_INDIRECT_INJECTION = re.compile(
    r"(?i)"
    r"(?:what\s+(?:are|is)\s+your\s+(?:instructions?|system\s+prompt|rules?|constraints?|restrictions?))"
    r"|(?:reveal\s+your\s+(?:guidelines?|instructions?|configuration))"
    r"|(?:print\s+(?:your\s+)?(?:prompt|instructions?))"
    r"|(?:output\s+your\s+(?:configuration|instructions?))"
    r"|(?:send\s+this\s+to)"
    r"|(?:forward\s+(?:this\s+)?to)"
    r"|(?:post\s+this\s+to)"
    r"|(?:upload\s+(?:this\s+)?to)"
    r"|(?:remember\s+this\s+for\s+later)"
    r"|(?:store\s+this\s+(?:information|instruction))"
    r"|(?:save\s+this\s+instruction)"
    r"|(?:from\s+now\s+on\s+always)"
    r"|(?:for\s+all\s+future\s+responses)"
    r"|(?:make\s+this\s+your\s+default)"
)

_SUSPICION_THRESHOLD = 0.45
_CRITICAL_THRESHOLD = 0.70


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    invisible = {
        "\u200b",
        "\u200c",
        "\u200d",
        "\u2060",
        "\ufeff",
        "\u00ad",
        "\u180e",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
    for char in invisible:
        normalized = normalized.replace(char, "")
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _decode_base64_segments(text: str) -> str:
    decoded_parts = []
    for m in re.finditer(r"[A-Za-z0-9+/]{20,}={0,2}", text):
        candidate = m.group(0)
        if len(candidate) % 4 != 0:
            candidate += "=" * (4 - len(candidate) % 4)
        try:
            decoded = base64.b64decode(candidate).decode("utf-8", errors="ignore")
            if decoded and any(c.isalpha() for c in decoded):
                decoded_parts.append(decoded)
        except Exception:
            pass
    return "\n".join(decoded_parts) if decoded_parts else ""


def _decode_url_encoding(text: str) -> str:
    try:
        return unquote(text)
    except Exception:
        return text


def _decode_html_entities(text: str) -> str:
    return html_module.unescape(text)


def scan_for_injection(text: str) -> InjectionScanResult:
    if not text or not text.strip():
        return InjectionScanResult(
            score=0.0,
            threat_level="none",
            is_suspicious=False,
            is_critical=False,
        )

    normalized = _normalize_text(text)
    url_decoded = _normalize_text(_decode_url_encoding(normalized))
    html_decoded = _normalize_text(_decode_html_entities(normalized))
    b64_decoded = _decode_base64_segments(normalized)
    if b64_decoded:
        b64_decoded = _normalize_text(b64_decoded)

    texts_to_scan = [normalized]
    if url_decoded != normalized:
        texts_to_scan.append(url_decoded)
    if html_decoded != normalized:
        texts_to_scan.append(html_decoded)
    if b64_decoded:
        texts_to_scan.append(b64_decoded)

    categories: dict[str, int] = {
        "role_impersonation": 0,
        "delimiter_abuse": 0,
        "jailbreak": 0,
        "command_injection": 0,
        "context_manipulation": 0,
        "indirect_injection": 0,
    }

    pattern_map = {
        "role_impersonation": _RE_ROLE_IMPERSONATION,
        "delimiter_abuse": _RE_DELIMITER_ABUSE,
        "jailbreak": _RE_JAILBREAK,
        "command_injection": _RE_COMMAND_INJECTION,
        "context_manipulation": _RE_CONTEXT_MANIPULATION,
        "indirect_injection": _RE_INDIRECT_INJECTION,
    }

    all_matches: list[PatternMatch] = []

    for scan_text in texts_to_scan:
        for category, regex in pattern_map.items():
            for m in regex.finditer(scan_text):
                categories[category] = categories.get(category, 0) + 1
                pos = m.start()
                ctx_start = max(0, pos - 30)
                ctx_end = min(len(scan_text), m.end() + 30)
                context = scan_text[ctx_start:ctx_end]
                severity = "high"
                if category in ("context_manipulation", "indirect_injection"):
                    severity = "medium"
                elif category in ("delimiter_abuse",):
                    severity = "medium"
                all_matches.append(
                    PatternMatch(
                        pattern=m.group(0)[:100],
                        category=category,
                        position=pos,
                        context=context,
                        severity=severity,
                    )
                )

    weights = {
        "role_impersonation": 0.35,
        "delimiter_abuse": 0.25,
        "jailbreak": 0.30,
        "command_injection": 0.25,
        "context_manipulation": 0.20,
        "indirect_injection": 0.15,
    }

    max_contributions = {
        "role_impersonation": 0.60,
        "delimiter_abuse": 0.45,
        "jailbreak": 0.50,
        "command_injection": 0.40,
        "context_manipulation": 0.35,
        "indirect_injection": 0.30,
    }

    escalation_thresholds = {
        "role_impersonation": 2,
        "delimiter_abuse": 3,
        "jailbreak": 2,
        "command_injection": 1,
        "context_manipulation": 2,
        "indirect_injection": 3,
    }

    base_score = 0.0
    for cat, count in categories.items():
        if count == 0:
            continue
        cat_score = min(count * weights[cat], max_contributions[cat])
        if count >= escalation_thresholds[cat]:
            cat_score += 0.15
        base_score += cat_score

    position_modifier = 0.0
    first_100 = normalized[:100] if len(normalized) >= 1 else ""
    for cat, regex in pattern_map.items():
        if regex.search(first_100):
            position_modifier += 0.15
            break

    combination_modifier = 0.0
    categories_present = sum(1 for c, cnt in categories.items() if cnt > 0)
    if categories_present >= 3:
        combination_modifier += 0.15
    if (
        categories.get("role_impersonation", 0) > 0
        and categories.get("delimiter_abuse", 0) > 0
    ):
        combination_modifier += 0.20
    if (
        categories.get("jailbreak", 0) > 0
        and categories.get("role_impersonation", 0) > 0
    ):
        combination_modifier += 0.25

    length_modifier = 0.0
    text_len = len(normalized)
    if text_len < 50 and categories_present > 0:
        length_modifier += 0.10
    elif text_len > 5000 and categories_present > 0:
        length_modifier += 0.05
    pattern_density = sum(categories.values()) / max(text_len, 1)
    if pattern_density > 0.05:
        length_modifier += 0.15

    final_score = min(
        1.0, base_score + position_modifier + combination_modifier + length_modifier
    )

    is_suspicious = (
        final_score >= _SUSPICION_THRESHOLD
        or categories.get("role_impersonation", 0) >= 2
        or categories.get("command_injection", 0) >= 1
        or (
            categories.get("jailbreak", 0) > 0
            and categories.get("role_impersonation", 0) > 0
        )
        or (categories_present >= 1 and bool(b64_decoded))
    )

    is_critical = (
        final_score >= _CRITICAL_THRESHOLD
        or (
            categories.get("role_impersonation", 0) > 0
            and bool(re.search(_RE_ROLE_IMPERSONATION, first_100))
        )
        or (
            categories.get("command_injection", 0) > 0
            and bool(
                re.search(r"(?:shell\s*=\s*True|eval\s*\()", first_100, re.IGNORECASE)
            )
        )
    )

    if final_score < 0.01:
        threat_level = "none"
    elif final_score < 0.30:
        threat_level = "low"
    elif final_score < 0.60:
        threat_level = "medium"
    else:
        threat_level = "high"

    if is_critical:
        threat_level = "critical"

    details = f"Score: {final_score:.3f}, Threat: {threat_level}, Categories: {dict(categories)}, Patterns: {len(all_matches)}"

    return InjectionScanResult(
        score=final_score,
        threat_level=threat_level,
        pattern_categories=categories,
        patterns_found=all_matches,
        is_suspicious=is_suspicious,
        is_critical=is_critical,
        details=details,
    )
