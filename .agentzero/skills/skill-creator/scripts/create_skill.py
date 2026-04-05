#!/usr/bin/env python3
"""Scaffold Agent Skills on disk."""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def normalize_skill_name(raw: str) -> str:
    """Normalize free-form input into a valid skill name."""
    value = str(raw or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")

    if len(value) > 64:
        value = value[:64].rstrip("-")

    if not value or not SKILL_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            "Skill name is invalid after normalization. "
            "Use lowercase letters, numbers, and hyphens."
        )

    return value


def title_from_name(skill_name: str) -> str:
    return " ".join(part.capitalize() for part in skill_name.split("-"))


def build_skill_markdown(skill_name: str, description: str) -> str:
    title = title_from_name(skill_name)
    body = textwrap.dedent(
        f"""\
        ---
        name: {skill_name}
        description: {description}
        ---

        # {title}

        ## When To Use This Skill
        Define when this skill should be activated.

        ## Steps
        1. Add task-specific instructions.
        2. Add clear success criteria.
        3. Keep complex reference material in `references/`.

        ## Notes
        Add edge cases, constraints, and examples.
        """
    )
    return body.rstrip() + "\n"


def write_file(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a new Agent Skill scaffold")
    parser.add_argument("--name", required=True, help="Skill name or topic")
    parser.add_argument(
        "--description",
        required=True,
        help="What the skill does and when to use it",
    )
    parser.add_argument(
        "--root",
        default=".agentzero/skills",
        help="Root directory where the skill directory will be created",
    )
    parser.add_argument(
        "--with-scripts",
        action="store_true",
        help="Create scripts/ with a starter script",
    )
    parser.add_argument(
        "--with-references",
        action="store_true",
        help="Create references/REFERENCE.md",
    )
    parser.add_argument(
        "--with-assets",
        action="store_true",
        help="Create assets/.gitkeep",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated files if they already exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    normalized_name = normalize_skill_name(args.name)
    description = str(args.description or "").strip()
    if not description:
        raise ValueError("description is required")

    root = Path(args.root).expanduser().resolve()
    skill_dir = root / normalized_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md_path = skill_dir / "SKILL.md"
    write_file(
        skill_md_path,
        build_skill_markdown(normalized_name, description),
        force=args.force,
    )

    created_paths = [str(skill_md_path)]

    if args.with_scripts:
        starter_script = skill_dir / "scripts" / "run.py"
        write_file(
            starter_script,
            "#!/usr/bin/env python3\nprint('replace with skill logic')\n",
            force=args.force,
        )
        created_paths.append(str(starter_script))

    if args.with_references:
        reference_path = skill_dir / "references" / "REFERENCE.md"
        write_file(
            reference_path,
            "# Reference\n\nAdd deep details for this skill here.\n",
            force=args.force,
        )
        created_paths.append(str(reference_path))

    if args.with_assets:
        asset_path = skill_dir / "assets" / ".gitkeep"
        write_file(asset_path, "", force=args.force)
        created_paths.append(str(asset_path))

    print(f"Created skill: {normalized_name}")
    if normalized_name != args.name:
        print(f"Normalized from: {args.name}")
    print(f"Skill directory: {skill_dir}")
    print("Files:")
    for path in created_paths:
        print(f"- {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
