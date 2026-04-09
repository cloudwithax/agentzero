"""Agent Skills discovery, validation, and activation support.

Implements progressive-disclosure behavior aligned with agentskills.io:
- Discover skills from conventional directories
- Load minimal catalog metadata into prompt context
- Activate skill instructions on demand
- Keep activated skills available per session without duplicate injections
- Fetch and install skills from remote URLs with injection scanning
"""

from __future__ import annotations

import html
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:  # Optional dependency; graceful fallback parser exists.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - dependency may be absent in some envs
    yaml = None

SKILL_FRONTMATTER_DELIM = "---"
SKILL_FILE_NAME = "SKILL.md"
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

DEFAULT_SCAN_MAX_DEPTH = 6
DEFAULT_SCAN_MAX_DIRS = 2000
DEFAULT_CATALOG_MAX_SKILLS = 200
DEFAULT_MAX_RESOURCE_FILES = 200
DEFAULT_REFRESH_INTERVAL_SECONDS = 30

DEFAULT_SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}


@dataclass
class SkillDefinition:
    """A discovered skill that passed validation."""

    name: str
    description: str
    skill_dir: str
    skill_md_path: str
    body: str
    scope: str
    source_root: str
    license: str = ""
    compatibility: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    model_invocation_disabled: bool = False
    frontmatter: dict[str, Any] = field(default_factory=dict)


def _is_truthy(value: Any) -> bool:
    """Interpret frontmatter toggles robustly."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "on", "enabled"}


def _coerce_scalar(value: str) -> str:
    """Normalize simple YAML scalar strings."""
    raw = (value or "").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        return raw[1:-1]
    return raw


def _parse_frontmatter_fallback(frontmatter_text: str) -> dict[str, Any]:
    """Minimal parser for expected Skill frontmatter fields.

    Supports top-level `key: value` pairs, one-level `metadata:` mappings,
    and YAML block scalars (folded `>` and literal `|`).
    """
    parsed: dict[str, Any] = {}
    metadata: dict[str, str] = {}
    in_metadata = False
    # Track block scalar collection (YAML `>` or `|` indicators).
    block_key: Optional[str] = None
    block_lines: list[str] = []
    block_folded: bool = False  # True for `>`, False for `|`

    def _flush_block() -> None:
        """Commit any accumulated block scalar to `parsed`."""
        nonlocal block_key, block_lines, block_folded
        if block_key is not None and block_lines:
            if block_folded:
                # Folded: join lines with spaces (like YAML `>`)
                parsed[block_key] = " ".join(block_lines)
            else:
                # Literal: preserve newlines (like YAML `|`)
                parsed[block_key] = "\n".join(block_lines)
        block_key = None
        block_lines = []
        block_folded = False

    for raw_line in frontmatter_text.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        is_indented = line.startswith(" ") or line.startswith("\t")

        # Collect continuation lines for a block scalar.
        if block_key is not None:
            if is_indented and stripped:
                block_lines.append(stripped)
                continue
            elif not stripped:
                # Blank line inside a block scalar: preserve for literal,
                # treat as paragraph break for folded.
                if not block_folded:
                    block_lines.append("")
                continue
            else:
                # Non-indented, non-empty line → block is over.
                _flush_block()

        if not stripped or stripped.startswith("#"):
            continue

        if in_metadata and is_indented:
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            if not key:
                continue
            metadata[key] = _coerce_scalar(value)
            continue

        in_metadata = False
        if ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if key == "metadata" and value == "":
            in_metadata = True
            continue

        # Detect YAML block scalar indicators (`>` or `|`).
        if value in (">", "|", ">-", "|-"):
            block_key = key
            block_lines = []
            block_folded = value.startswith(">")
            continue

        parsed[key] = _coerce_scalar(value)

    # Flush any trailing block scalar at end of frontmatter.
    _flush_block()

    if metadata:
        parsed["metadata"] = metadata

    return parsed


def _split_skill_markdown(
    raw_text: str,
) -> tuple[dict[str, Any], str] | tuple[None, str]:
    """Split SKILL.md into frontmatter and markdown body."""
    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != SKILL_FRONTMATTER_DELIM:
        return None, "Missing opening YAML frontmatter delimiter ('---')."

    closing_index: Optional[int] = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == SKILL_FRONTMATTER_DELIM:
            closing_index = idx
            break

    if closing_index is None:
        return None, "Missing closing YAML frontmatter delimiter ('---')."

    frontmatter_text = "\n".join(lines[1:closing_index])
    body = "\n".join(lines[closing_index + 1 :]).strip()

    frontmatter: dict[str, Any]
    if yaml is not None:
        try:
            loaded = (
                yaml.safe_load(frontmatter_text) if frontmatter_text.strip() else {}
            )
            if loaded is None:
                loaded = {}
            if not isinstance(loaded, dict):
                return None, "YAML frontmatter must parse to an object/map."
            frontmatter = loaded
        except Exception as exc:
            return None, f"Invalid YAML frontmatter: {exc}"
    else:
        frontmatter = _parse_frontmatter_fallback(frontmatter_text)

    return frontmatter, body


def _validate_skill_frontmatter(
    frontmatter: dict[str, Any],
    parent_dir_name: str,
) -> tuple[bool, Optional[str], dict[str, Any]]:
    """Validate frontmatter against Agent Skills spec constraints."""
    normalized = dict(frontmatter)

    name = normalized.get("name")
    if not isinstance(name, str):
        return False, "Frontmatter 'name' is required and must be a string.", {}

    name = name.strip()
    if not (1 <= len(name) <= 64):
        return False, "Frontmatter 'name' must be 1-64 characters.", {}

    if not SKILL_NAME_PATTERN.fullmatch(name):
        return (
            False,
            "Frontmatter 'name' must use lowercase letters, numbers, and single hyphens.",
            {},
        )

    if name != parent_dir_name:
        return (
            False,
            f"Frontmatter 'name' ({name}) must match parent directory ({parent_dir_name}).",
            {},
        )

    description = normalized.get("description")
    if not isinstance(description, str):
        return False, "Frontmatter 'description' is required and must be a string.", {}

    description = description.strip()
    if not (1 <= len(description) <= 1024):
        return False, "Frontmatter 'description' must be 1-1024 characters.", {}

    license_value = normalized.get("license", "")
    if license_value is None:
        license_value = ""
    if not isinstance(license_value, str):
        return False, "Frontmatter 'license' must be a string when provided.", {}

    compatibility = normalized.get("compatibility", "")
    if compatibility is None:
        compatibility = ""
    if not isinstance(compatibility, str):
        return False, "Frontmatter 'compatibility' must be a string when provided.", {}
    compatibility = compatibility.strip()
    if compatibility and len(compatibility) > 500:
        return False, "Frontmatter 'compatibility' must be <= 500 characters.", {}

    metadata = normalized.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        return False, "Frontmatter 'metadata' must be a map when provided.", {}

    normalized_metadata: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            return False, "Frontmatter 'metadata' keys must be strings.", {}
        normalized_metadata[key] = "" if value is None else str(value)

    allowed_tools_raw = normalized.get("allowed-tools", "")
    if allowed_tools_raw is None:
        allowed_tools_raw = ""
    if not isinstance(allowed_tools_raw, str):
        return False, "Frontmatter 'allowed-tools' must be a string when provided.", {}

    allowed_tools = [token for token in allowed_tools_raw.split() if token]

    normalized["name"] = name
    normalized["description"] = description
    normalized["license"] = license_value.strip()
    normalized["compatibility"] = compatibility
    normalized["metadata"] = normalized_metadata
    normalized["allowed-tools"] = allowed_tools

    return True, None, normalized


class SkillRegistry:
    """Discovers, catalogs, and activates Agent Skills."""

    def __init__(
        self,
        project_root: Optional[str] = None,
        user_home: Optional[str] = None,
        client_name: str = "agentzero",
        scan_max_depth: int = DEFAULT_SCAN_MAX_DEPTH,
        scan_max_dirs: int = DEFAULT_SCAN_MAX_DIRS,
        catalog_max_skills: int = DEFAULT_CATALOG_MAX_SKILLS,
        max_resource_files: int = DEFAULT_MAX_RESOURCE_FILES,
        refresh_interval_seconds: int = DEFAULT_REFRESH_INTERVAL_SECONDS,
    ):
        self.client_name = (
            str(client_name or "agentzero").strip().lower() or "agentzero"
        )
        self.project_root = os.path.abspath(project_root or os.getcwd())
        self.user_home = os.path.abspath(os.path.expanduser(user_home or "~"))

        self.scan_max_depth = max(1, int(scan_max_depth))
        self.scan_max_dirs = max(50, int(scan_max_dirs))
        self.catalog_max_skills = max(1, int(catalog_max_skills))
        self.max_resource_files = max(1, int(max_resource_files))
        self.refresh_interval_seconds = max(5, int(refresh_interval_seconds))

        self.enabled = os.environ.get("AGENTZERO_SKILLS_ENABLED", "1").strip() != "0"
        self.trust_project_skills = (
            os.environ.get("AGENTZERO_TRUST_PROJECT_SKILLS", "1").strip() != "0"
        )
        self.enable_user_skills = (
            os.environ.get("AGENTZERO_ENABLE_USER_SKILLS", "1").strip() != "0"
        )
        self.enable_builtin_skills = (
            os.environ.get("AGENTZERO_ENABLE_BUILTIN_SKILLS", "1").strip() != "0"
        )
        self.scan_claude_compat = (
            os.environ.get("AGENTZERO_SCAN_CLAUDE_SKILLS", "1").strip() != "0"
        )
        default_builtin_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".agentzero",
            "skills",
        )
        self.builtin_skills_root = os.path.abspath(
            os.environ.get(
                "AGENTZERO_BUILTIN_SKILLS_DIR",
                default_builtin_root,
            )
        )

        self.disabled_skills = {
            token.strip()
            for token in os.environ.get("AGENTZERO_DISABLED_SKILLS", "").split(",")
            if token.strip()
        }

        self._lock = threading.RLock()
        self._skills: dict[str, SkillDefinition] = {}
        self._active_by_session: dict[str, dict[str, str]] = {}
        self._last_scan_ts: float = 0.0

        self.discover_skills(force=True)

    def _scan_roots(self) -> list[tuple[str, str]]:
        """Return ordered discovery roots. Earlier roots win on name collisions."""
        roots: list[tuple[str, str]] = []

        project_native = os.path.join(
            self.project_root, f".{self.client_name}", "skills"
        )
        project_shared = os.path.join(self.project_root, ".agents", "skills")
        project_claude = os.path.join(self.project_root, ".claude", "skills")

        user_native = os.path.join(self.user_home, f".{self.client_name}", "skills")
        user_shared = os.path.join(self.user_home, ".agents", "skills")
        user_claude = os.path.join(self.user_home, ".claude", "skills")

        if self.trust_project_skills:
            roots.append(("project", project_native))
            roots.append(("project", project_shared))
            if self.scan_claude_compat:
                roots.append(("project", project_claude))

        if self.enable_user_skills:
            roots.append(("user", user_native))
            roots.append(("user", user_shared))
            if self.scan_claude_compat:
                roots.append(("user", user_claude))

        # Builtin skills are scanned last so project/user skills can override them.
        if self.enable_builtin_skills and self.builtin_skills_root:
            roots.append(("builtin", self.builtin_skills_root))

        deduped: list[tuple[str, str]] = []
        seen_paths: set[str] = set()
        for scope, path in roots:
            normalized_path = os.path.abspath(path)
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            deduped.append((scope, normalized_path))

        return deduped

    def _should_skip_dir(self, dirname: str) -> bool:
        return dirname in DEFAULT_SKIP_DIR_NAMES

    def _walk_skill_directories(self, root_path: str) -> list[str]:
        """Find directories under root that contain SKILL.md within safe bounds."""
        skill_dirs: list[str] = []
        visited = 0

        for current_root, dirnames, filenames in os.walk(root_path, topdown=True):
            visited += 1
            if visited > self.scan_max_dirs:
                logger.warning(
                    "Skill scan capped at %s directories under %s",
                    self.scan_max_dirs,
                    root_path,
                )
                break

            rel_path = os.path.relpath(current_root, root_path)
            depth = 0 if rel_path in {".", ""} else rel_path.count(os.sep) + 1
            if depth > self.scan_max_depth:
                dirnames[:] = []
                continue

            dirnames[:] = sorted([d for d in dirnames if not self._should_skip_dir(d)])

            if SKILL_FILE_NAME in filenames:
                skill_dirs.append(current_root)

        return skill_dirs

    def _build_skill_definition(
        self,
        skill_dir: str,
        scope: str,
        source_root: str,
    ) -> tuple[Optional[SkillDefinition], Optional[str]]:
        skill_md_path = os.path.join(skill_dir, SKILL_FILE_NAME)

        try:
            with open(skill_md_path, "r", encoding="utf-8") as handle:
                raw = handle.read()
        except Exception as exc:
            return None, f"Failed reading {skill_md_path}: {exc}"

        parsed_frontmatter, body_or_error = _split_skill_markdown(raw)
        if parsed_frontmatter is None:
            return None, f"{skill_md_path}: {body_or_error}"

        frontmatter = parsed_frontmatter
        body = body_or_error
        parent_dir_name = os.path.basename(skill_dir.rstrip(os.sep))

        valid, validation_error, normalized = _validate_skill_frontmatter(
            frontmatter=frontmatter,
            parent_dir_name=parent_dir_name,
        )
        if not valid:
            return None, f"{skill_md_path}: {validation_error}"

        skill_name = str(normalized["name"])
        if skill_name in self.disabled_skills:
            return None, f"{skill_md_path}: skill '{skill_name}' is disabled by config"

        metadata = normalized.get("metadata", {})
        model_invocation_disabled = _is_truthy(
            normalized.get("disable-model-invocation")
        ) or _is_truthy(metadata.get("disable-model-invocation"))

        return (
            SkillDefinition(
                name=skill_name,
                description=str(normalized["description"]),
                skill_dir=os.path.abspath(skill_dir),
                skill_md_path=os.path.abspath(skill_md_path),
                body=body,
                scope=scope,
                source_root=os.path.abspath(source_root),
                license=str(normalized.get("license", "") or ""),
                compatibility=str(normalized.get("compatibility", "") or ""),
                metadata=metadata if isinstance(metadata, dict) else {},
                allowed_tools=list(normalized.get("allowed-tools", []) or []),
                model_invocation_disabled=model_invocation_disabled,
                frontmatter=normalized,
            ),
            None,
        )

    def discover_skills(self, force: bool = False) -> dict[str, SkillDefinition]:
        """Discover and cache available skills."""
        if not self.enabled:
            with self._lock:
                self._skills = {}
                self._last_scan_ts = time.time()
            return {}

        now = time.time()
        with self._lock:
            if not force and (now - self._last_scan_ts) < self.refresh_interval_seconds:
                return dict(self._skills)

        discovered: dict[str, SkillDefinition] = {}
        roots = self._scan_roots()

        for scope, root in roots:
            if not os.path.isdir(root):
                continue

            for skill_dir in self._walk_skill_directories(root):
                definition, error = self._build_skill_definition(
                    skill_dir=skill_dir,
                    scope=scope,
                    source_root=root,
                )
                if error:
                    logger.warning("Skipping skill: %s", error)
                    continue
                if definition is None:
                    continue

                if definition.name in discovered:
                    previous = discovered[definition.name]
                    logger.warning(
                        "Skill name collision for '%s'. Keeping %s, shadowing %s",
                        definition.name,
                        previous.skill_md_path,
                        definition.skill_md_path,
                    )
                    continue

                discovered[definition.name] = definition

        with self._lock:
            self._skills = discovered
            self._last_scan_ts = time.time()

        return dict(discovered)

    def refresh_if_due(self) -> None:
        """Refresh discovery cache if scan interval has elapsed."""
        self.discover_skills(force=False)

    def get_skill(self, name: str) -> Optional[SkillDefinition]:
        """Return one discovered skill by exact name."""
        normalized = str(name or "").strip()
        if not normalized:
            return None

        self.refresh_if_due()
        with self._lock:
            return self._skills.get(normalized)

    def list_skills(self, include_model_hidden: bool = True) -> list[SkillDefinition]:
        """Return discovered skills sorted by name."""
        self.refresh_if_due()
        with self._lock:
            skills = list(self._skills.values())

        if not include_model_hidden:
            skills = [s for s in skills if not s.model_invocation_disabled]

        skills.sort(key=lambda s: s.name)
        return skills

    def _escape(self, value: str) -> str:
        return html.escape(value or "", quote=False)

    def build_available_skills_catalog(self) -> str:
        """Build XML-ish catalog block for system prompt tier-1 disclosure."""
        skills = self.list_skills(include_model_hidden=False)
        if not skills:
            return ""

        lines = [
            "The following skills provide specialized instructions for specific tasks.",
            "When a task matches a skill's description, call activate_skill with the skill name before proceeding.",
            "After activating a skill, you will receive its full instructions in the tool result. You MUST read and follow those instructions step-by-step to complete the user's request.",
            "When a skill references relative paths, resolve them against the skill directory and use absolute paths in tool calls.",
            "",
            "<available_skills>",
        ]

        for skill in skills[: self.catalog_max_skills]:
            lines.extend(
                [
                    "  <skill>",
                    f"    <name>{self._escape(skill.name)}</name>",
                    f"    <description>{self._escape(skill.description)}</description>",
                    f"    <location>{self._escape(skill.skill_md_path)}</location>",
                    "  </skill>",
                ]
            )

        lines.append("</available_skills>")
        return "\n".join(lines).strip()

    def build_activation_tool_schema(self) -> Optional[dict[str, Any]]:
        """Return activate_skill tool schema with enum-constrained names."""
        skills = self.list_skills(include_model_hidden=False)
        skill_names = [skill.name for skill in skills]
        if not skill_names:
            return None

        return {
            "type": "function",
            "function": {
                "name": "activate_skill",
                "description": "Activate a discovered skill by name and load its full instructions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": skill_names,
                            "description": "Exact skill name to activate",
                        }
                    },
                    "required": ["name"],
                },
            },
        }

    def _list_skill_resources(self, skill: SkillDefinition) -> tuple[list[str], bool]:
        """List bundled skill resources without reading file contents."""
        resources: list[str] = []
        truncated = False

        for top_level in ("scripts", "references", "assets"):
            base = os.path.join(skill.skill_dir, top_level)
            if not os.path.isdir(base):
                continue

            for root, dirnames, filenames in os.walk(base, topdown=True):
                dirnames[:] = sorted(
                    [d for d in dirnames if not self._should_skip_dir(d)]
                )
                for filename in sorted(filenames):
                    abs_path = os.path.join(root, filename)
                    rel = os.path.relpath(abs_path, skill.skill_dir)
                    resources.append(rel)
                    if len(resources) >= self.max_resource_files:
                        truncated = True
                        return resources, truncated

        return resources, truncated

    def _build_wrapped_skill_content(self, skill: SkillDefinition) -> str:
        """Wrap activated skill body with structural tags for context management."""
        resources, truncated = self._list_skill_resources(skill)

        lines = [f'<skill_content name="{self._escape(skill.name)}">']

        if skill.body:
            lines.append(skill.body)
        else:
            lines.append("(No body content provided in SKILL.md)")

        lines.append("")
        lines.append(f"Skill directory: {skill.skill_dir}")
        lines.append(
            "Relative paths in this skill are relative to the skill directory."
        )

        if skill.compatibility:
            lines.append(f"Compatibility: {skill.compatibility}")

        lines.append("<skill_resources>")
        for rel_path in resources:
            lines.append(f"  <file>{self._escape(rel_path)}</file>")
        if truncated:
            lines.append("  <note>Resource listing truncated.</note>")
        lines.append("</skill_resources>")
        lines.append("</skill_content>")

        return "\n".join(lines).strip()

    def activate_skill(
        self,
        name: str,
        session_id: Optional[str] = None,
        source: str = "model",
    ) -> dict[str, Any]:
        """Activate one skill and optionally cache it for a session."""
        normalized_name = str(name or "").strip()
        if not normalized_name:
            return {"success": False, "error": "Skill name is required"}

        skill = self.get_skill(normalized_name)
        if skill is None:
            return {"success": False, "error": f"Unknown skill: {normalized_name}"}

        if source == "model" and skill.model_invocation_disabled:
            return {
                "success": False,
                "error": f"Skill is not available for model-driven activation: {skill.name}",
            }

        normalized_session = str(session_id or "").strip()

        with self._lock:
            if normalized_session:
                active = self._active_by_session.setdefault(normalized_session, {})
                if skill.name in active:
                    return {
                        "success": True,
                        "name": skill.name,
                        "already_active": True,
                        "message": (
                            f"Skill '{skill.name}' is already active for this session; "
                            "skipping duplicate injection."
                        ),
                        "skill_dir": skill.skill_dir,
                    }

            wrapped = self._build_wrapped_skill_content(skill)

            if normalized_session:
                self._active_by_session.setdefault(normalized_session, {})[
                    skill.name
                ] = wrapped

        return {
            "success": True,
            "name": skill.name,
            "description": skill.description,
            "skill_dir": skill.skill_dir,
            "already_active": False,
            "content": wrapped,
        }

    def get_active_skill_names(self, session_id: Optional[str]) -> list[str]:
        """Return active skill names for a session in activation order."""
        normalized_session = str(session_id or "").strip()
        if not normalized_session:
            return []

        with self._lock:
            active = self._active_by_session.get(normalized_session, {})
            return list(active.keys())

    def build_active_skills_context(self, session_id: Optional[str]) -> str:
        """Build context block containing all activated skills for a session."""
        normalized_session = str(session_id or "").strip()
        if not normalized_session:
            return ""

        with self._lock:
            active = self._active_by_session.get(normalized_session, {})
            if not active:
                return ""
            snippets = list(active.values())

        lines = ["[Activated Skills - Persist for this session]:"]
        lines.extend(snippets)
        return "\n\n".join(lines).strip()

    def clear_session_active_skills(self, session_id: Optional[str]) -> None:
        """Clear activated-skill cache for a session."""
        normalized_session = str(session_id or "").strip()
        if not normalized_session:
            return

        with self._lock:
            self._active_by_session.pop(normalized_session, None)

    async def add_skill_from_url(
        self,
        url: str,
        session_id: Optional[str] = None,
        auto_activate: bool = True,
    ) -> dict[str, Any]:
        """Fetch a SKILL.md from a remote URL, scan for injection, and install it.

        The skill is written to the user-level native skills directory
        (``~/.agentzero/skills/<name>/SKILL.md``) so it persists across
        restarts and is picked up by future discovery scans.
        """
        from injection_scanner import scan_for_injection

        url = (url or "").strip()
        if not url:
            return {"success": False, "error": "URL is required"}

        if not re.match(r"^https?://", url, re.IGNORECASE):
            return {
                "success": False,
                "error": "URL must start with http:// or https://",
            }

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=30),
                    headers={"User-Agent": "AgentZero-SkillFetcher/1.0"},
                ) as resp:
                    if resp.status != 200:
                        return {
                            "success": False,
                            "error": f"Failed to fetch URL (HTTP {resp.status})",
                        }
                    raw_text = await resp.text()
        except Exception as exc:
            return {"success": False, "error": f"Failed to fetch URL: {exc}"}

        if not raw_text or not raw_text.strip():
            return {"success": False, "error": "Fetched content is empty"}

        scan_result = scan_for_injection(raw_text)
        if scan_result.is_critical:
            return {
                "success": False,
                "error": (
                    f"Content failed prompt-injection scan (score={scan_result.score:.3f}, "
                    f"threat={scan_result.threat_level}). "
                    f"Detected patterns: {scan_result.details}"
                ),
                "scan_score": scan_result.score,
                "threat_level": scan_result.threat_level,
            }

        if scan_result.is_suspicious:
            return {
                "success": False,
                "error": (
                    f"Content flagged as suspicious by prompt-injection scan "
                    f"(score={scan_result.score:.3f}, threat={scan_result.threat_level}). "
                    f"Review the skill manually before installing. "
                    f"Details: {scan_result.details}"
                ),
                "scan_score": scan_result.score,
                "threat_level": scan_result.threat_level,
            }

        parsed_frontmatter, body_or_error = _split_skill_markdown(raw_text)
        if parsed_frontmatter is None:
            return {
                "success": False,
                "error": f"Content is not valid SKILL.md format: {body_or_error}",
            }

        frontmatter = parsed_frontmatter
        body = body_or_error

        tentative_name = str(frontmatter.get("name", "")).strip()
        if not tentative_name:
            return {
                "success": False,
                "error": "SKILL.md frontmatter must include a 'name' field",
            }

        parent_dir_name = re.sub(
            r"[^a-z0-9-]", "", tentative_name.lower().replace(" ", "-")
        )
        if not parent_dir_name:
            return {
                "success": False,
                "error": f"Could not derive a valid directory name from skill name '{tentative_name}'",
            }

        frontmatter["name"] = parent_dir_name

        valid, validation_error, normalized = _validate_skill_frontmatter(
            frontmatter=frontmatter,
            parent_dir_name=parent_dir_name,
        )
        if not valid:
            return {
                "success": False,
                "error": f"SKILL.md validation failed: {validation_error}",
            }

        skill_name = str(normalized["name"])

        if skill_name in self.disabled_skills:
            return {
                "success": False,
                "error": f"Skill '{skill_name}' is disabled by configuration",
            }

        user_native = os.path.join(self.user_home, f".{self.client_name}", "skills")
        skill_dir = os.path.join(user_native, skill_name)
        skill_md_path = os.path.join(skill_dir, SKILL_FILE_NAME)

        try:
            os.makedirs(skill_dir, exist_ok=True)
            with open(skill_md_path, "w", encoding="utf-8") as handle:
                handle.write(raw_text)
        except Exception as exc:
            return {
                "success": False,
                "error": f"Failed to write skill to disk: {exc}",
            }

        definition, build_error = self._build_skill_definition(
            skill_dir=skill_dir,
            scope="user",
            source_root=user_native,
        )
        if build_error or definition is None:
            try:
                os.remove(skill_md_path)
                os.rmdir(skill_dir)
            except Exception:
                pass
            return {
                "success": False,
                "error": f"Skill was written but failed to load: {build_error}",
            }

        with self._lock:
            self._skills[skill_name] = definition

        result: dict[str, Any] = {
            "success": True,
            "name": skill_name,
            "description": definition.description,
            "skill_dir": skill_dir,
            "scan_score": scan_result.score,
            "threat_level": scan_result.threat_level,
        }

        if auto_activate and session_id:
            activation = self.activate_skill(
                name=skill_name,
                session_id=session_id,
                source="user",
            )
            result["activated"] = activation.get("success", False)
            result["activation_message"] = activation.get(
                "message", activation.get("error", "")
            )
            if activation.get("content"):
                result["content"] = activation["content"]

        return result
