#!/usr/bin/env python3
"""Tests for Agent Skills discovery and activation."""

import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integrations import _parse_skill_invocation
from skills import SkillRegistry


@contextmanager
def _temp_env(overrides: dict[str, str]):
    """Temporarily set environment variables for a test."""
    original: dict[str, str | None] = {}
    for key, value in overrides.items():
        original[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, previous in original.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _write_skill(skill_root: str, name: str, description: str, body: str = "") -> None:
    os.makedirs(skill_root, exist_ok=True)
    with open(os.path.join(skill_root, "SKILL.md"), "w", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "---",
                    f"name: {name}",
                    f"description: {description}",
                    "---",
                    body or "# Instructions\\nUse this skill when appropriate.",
                ]
            )
        )


def _write_model_hidden_skill(skill_root: str, name: str) -> None:
    os.makedirs(skill_root, exist_ok=True)
    with open(os.path.join(skill_root, "SKILL.md"), "w", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "---",
                    f"name: {name}",
                    "description: Hidden from model-driven activation.",
                    "metadata:",
                    "  disable-model-invocation: true",
                    "---",
                    "# Hidden Skill",
                    "",
                    "Only user-explicit activation should succeed.",
                ]
            )
        )


def test_skill_discovery_and_precedence() -> None:
    print("Test 1: skill discovery + precedence")
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = os.path.join(tmpdir, "project")
        user_home = os.path.join(tmpdir, "home")
        os.makedirs(project_root, exist_ok=True)
        os.makedirs(user_home, exist_ok=True)

        # Project-level native skill (should win collisions)
        _write_skill(
            os.path.join(project_root, ".agentzero", "skills", "code-review"),
            "code-review",
            "Review code for regressions.",
        )
        # User-level shared skill with same name (should be shadowed)
        _write_skill(
            os.path.join(user_home, ".agents", "skills", "code-review"),
            "code-review",
            "User-level fallback skill.",
        )
        # User-level additional skill
        _write_skill(
            os.path.join(user_home, ".agentzero", "skills", "release-notes"),
            "release-notes",
            "Generate concise release notes.",
        )
        # Invalid skill (name mismatch with directory) should be skipped
        _write_skill(
            os.path.join(project_root, ".agents", "skills", "bad-dir-name"),
            "bad-name",
            "Invalid naming should be rejected.",
        )

        with _temp_env(
            {
                "AGENTZERO_SKILLS_ENABLED": "1",
                "AGENTZERO_TRUST_PROJECT_SKILLS": "1",
                "AGENTZERO_ENABLE_USER_SKILLS": "1",
                "AGENTZERO_SCAN_CLAUDE_SKILLS": "0",
                "AGENTZERO_DISABLED_SKILLS": "",
            }
        ):
            registry = SkillRegistry(project_root=project_root, user_home=user_home)
            skills = registry.list_skills(include_model_hidden=True)
            names = [skill.name for skill in skills]

            assert "code-review" in names
            assert "release-notes" in names
            assert "bad-name" not in names

            code_review = registry.get_skill("code-review")
            assert code_review is not None
            assert code_review.scope == "project"
            assert "/.agentzero/skills/code-review" in code_review.skill_dir

    print("  ✓ Passed")


def test_skill_activation_and_deduplication() -> None:
    print("Test 2: skill activation + deduplication")
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = os.path.join(tmpdir, "project")
        user_home = os.path.join(tmpdir, "home")
        skill_dir = os.path.join(project_root, ".agents", "skills", "pdf-processing")
        _write_skill(
            skill_dir,
            "pdf-processing",
            "Extract and transform PDF content.",
            body="# PDF Processing\\nRun scripts/extract.py when needed.",
        )
        os.makedirs(os.path.join(skill_dir, "scripts"), exist_ok=True)
        with open(
            os.path.join(skill_dir, "scripts", "extract.py"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write("print('extract')\n")

        with _temp_env(
            {
                "AGENTZERO_SKILLS_ENABLED": "1",
                "AGENTZERO_TRUST_PROJECT_SKILLS": "1",
                "AGENTZERO_ENABLE_USER_SKILLS": "0",
                "AGENTZERO_SCAN_CLAUDE_SKILLS": "0",
            }
        ):
            registry = SkillRegistry(project_root=project_root, user_home=user_home)

            catalog = registry.build_available_skills_catalog()
            assert "<available_skills>" in catalog
            assert "pdf-processing" in catalog

            schema = registry.build_activation_tool_schema()
            assert schema is not None
            enum_values = schema["function"]["parameters"]["properties"]["name"]["enum"]
            assert "pdf-processing" in enum_values

            first = registry.activate_skill(
                name="pdf-processing",
                session_id="session-1",
                source="model",
            )
            assert first.get("success") is True
            assert first.get("already_active") is False
            assert "<skill_content" in str(first.get("content", ""))
            assert "scripts/extract.py" in str(first.get("content", ""))

            second = registry.activate_skill(
                name="pdf-processing",
                session_id="session-1",
                source="model",
            )
            assert second.get("success") is True
            assert second.get("already_active") is True

            active_context = registry.build_active_skills_context("session-1")
            assert "[Activated Skills - Persist for this session]" in active_context
            assert "pdf-processing" in active_context

    print("  ✓ Passed")


def test_model_hidden_and_invocation_parsing() -> None:
    print("Test 3: model-hidden skills + explicit invocation parsing")
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = os.path.join(tmpdir, "project")
        user_home = os.path.join(tmpdir, "home")
        hidden_dir = os.path.join(project_root, ".agentzero", "skills", "hidden-skill")
        _write_model_hidden_skill(hidden_dir, "hidden-skill")

        with _temp_env(
            {
                "AGENTZERO_SKILLS_ENABLED": "1",
                "AGENTZERO_TRUST_PROJECT_SKILLS": "1",
                "AGENTZERO_ENABLE_USER_SKILLS": "0",
                "AGENTZERO_SCAN_CLAUDE_SKILLS": "0",
            }
        ):
            registry = SkillRegistry(project_root=project_root, user_home=user_home)

            visible_names = [
                skill.name
                for skill in registry.list_skills(include_model_hidden=False)
            ]
            assert "hidden-skill" not in visible_names

            model_activation = registry.activate_skill(
                name="hidden-skill",
                session_id="s",
                source="model",
            )
            assert model_activation.get("success") is False

            user_activation = registry.activate_skill(
                name="hidden-skill",
                session_id="s",
                source="user",
            )
            assert user_activation.get("success") is True

    # Explicit invocation parsing
    assert _parse_skill_invocation("/code-review") == ("code-review", "")
    assert _parse_skill_invocation("$release-notes draft from changelog") == (
        "release-notes",
        "draft from changelog",
    )
    assert _parse_skill_invocation("/start") == ("", "")
    assert _parse_skill_invocation("not a command") == ("", "")

    print("  ✓ Passed")


def test_builtin_default_skill_creator_and_scaffold_script() -> None:
    print("Test 4: builtin default skill + scaffold script")
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = os.path.join(tmpdir, "project")
        user_home = os.path.join(tmpdir, "home")
        builtin_root = os.path.join(tmpdir, "builtin", "skills")
        skill_creator_root = os.path.join(builtin_root, "skill-creator")
        script_path = os.path.join(skill_creator_root, "scripts", "create_skill.py")

        os.makedirs(os.path.join(skill_creator_root, "scripts"), exist_ok=True)
        os.makedirs(project_root, exist_ok=True)
        os.makedirs(user_home, exist_ok=True)

        # Minimal builtin skill-creator definition.
        _write_skill(
            skill_creator_root,
            "skill-creator",
            "Create skills and save them to disk.",
            body="# Skill Creator\\nUse the scaffold script.",
        )

        # Copy real scaffolder script from repository fixture.
        repo_script = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                ".agentzero",
                "skills",
                "skill-creator",
                "scripts",
                "create_skill.py",
            )
        )
        with open(repo_script, "r", encoding="utf-8") as src:
            script_content = src.read()
        with open(script_path, "w", encoding="utf-8") as dst:
            dst.write(script_content)

        with _temp_env(
            {
                "AGENTZERO_SKILLS_ENABLED": "1",
                "AGENTZERO_TRUST_PROJECT_SKILLS": "0",
                "AGENTZERO_ENABLE_USER_SKILLS": "0",
                "AGENTZERO_ENABLE_BUILTIN_SKILLS": "1",
                "AGENTZERO_BUILTIN_SKILLS_DIR": builtin_root,
                "AGENTZERO_SCAN_CLAUDE_SKILLS": "0",
            }
        ):
            registry = SkillRegistry(project_root=project_root, user_home=user_home)
            skill = registry.get_skill("skill-creator")
            assert skill is not None
            assert skill.scope == "builtin"

            out_root = os.path.join(tmpdir, "generated-skills")
            result = subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--name",
                    "Incident Response Playbook",
                    "--description",
                    "Create incident response procedures. Use when handling outages.",
                    "--root",
                    out_root,
                    "--with-references",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            assert "Created skill: incident-response-playbook" in result.stdout

            generated_skill_dir = os.path.join(out_root, "incident-response-playbook")
            generated_skill_md = os.path.join(generated_skill_dir, "SKILL.md")
            assert os.path.exists(generated_skill_md)

            with open(generated_skill_md, "r", encoding="utf-8") as handle:
                generated = handle.read()
            assert "name: incident-response-playbook" in generated

    print("  ✓ Passed")


if __name__ == "__main__":
    test_skill_discovery_and_precedence()
    test_skill_activation_and_deduplication()
    test_model_hidden_and_invocation_parsing()
    test_builtin_default_skill_creator_and_scaffold_script()
    print("All skill tests passed!")
