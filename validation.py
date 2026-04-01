"""Validation classes for output parsing and validation."""

import re
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class OutputFormat(Enum):
    """Supported output formats."""
    JSON = "json"
    STRUCTURED_MARKDOWN = "structured_markdown"
    SIMPLE_TEXT = "simple_text"


@dataclass
class StepResult:
    """Result of executing a step."""
    step_id: str
    success: bool
    output: Any
    raw_response: str
    attempts: int
    error: Optional[str] = None


class OutputValidator:
    """Validates and parses model outputs."""

    def __init__(self):
        self.retry_count = 0

    def validate_and_parse(
        self, raw_output: str, step: Any = None, output_schema: Optional[Dict] = None,
        attempt: int = 1, expected_format: OutputFormat = OutputFormat.JSON
    ) -> Tuple[bool, Any, Optional[str]]:
        """Validate and parse model output."""
        schema = output_schema or (step.output_schema if step and hasattr(step, 'output_schema') else {})

        if expected_format == OutputFormat.JSON or "json" in schema.get("format", "").lower():
            return self._validate_json(raw_output, schema)

        if schema:
            return self._validate_structure(raw_output, schema)

        return True, raw_output, None

    def _validate_json(self, raw_output: str, schema: Dict) -> Tuple[bool, Any, Optional[str]]:
        """Validate and parse JSON output."""
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_output, re.DOTALL)
        if json_match:
            raw_output = json_match.group(1)

        if not raw_output.strip().startswith("{"):
            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if match:
                raw_output = match.group(0)

        try:
            parsed = json.loads(raw_output)
            required_keys = schema.get("required", [])
            missing = [k for k in required_keys if k not in parsed]
            if missing:
                return False, None, f"Missing required keys: {missing}"
            return True, parsed, None
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {str(e)}"

    def _validate_structure(self, raw_output: str, schema: Dict) -> Tuple[bool, Any, Optional[str]]:
        """Validate structured output without strict JSON."""
        result = {}
        for key in schema.keys():
            if key == "format":
                continue
            patterns = [
                rf"{key}[\s]*[:=][\s]*([^\n]+)",
                rf"\*{key}\*[\s]*[:=]?[\s]*([^\n]+)",
                rf"\*\*{key}\*\*[\s]*[:=]?[\s]*([^\n]+)",
                rf"{key}[\s]+([^\n]+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, raw_output, re.IGNORECASE)
                if match:
                    result[key] = match.group(1).strip()
                    break
        if result:
            result["_raw"] = raw_output
            return True, result, None
        return True, {"_raw": raw_output}, None
