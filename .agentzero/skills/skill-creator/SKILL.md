---
name: skill-creator
description: Create new Agent Skills for any domain, scaffold the required SKILL.md structure, and save the skill to disk. Use when the user asks to create or update a reusable skill.
compatibility: Requires Python 3 and local filesystem write access.
metadata:
  author: agentzero
  purpose: default-builtin-skill
---

# Skill Creator

## When To Use This Skill
Use this skill when the user asks to create a new skill, bootstrap multiple skills, or update an existing skill definition so it can be reused in future conversations.

## Core Behavior
1. Collect or infer the skill intent.
2. Choose a valid skill name (lowercase letters, numbers, and hyphens).
3. Generate a specific `description` that states what the skill does and when to use it.
4. Scaffold the skill on disk using the script below.
5. Verify that `SKILL.md` exists and the frontmatter `name` matches the directory name.
6. Report the absolute path that was written.

## Command
Run the scaffold script from the repository root:

`python .agentzero/skills/skill-creator/scripts/create_skill.py --name "<skill-name-or-topic>" --description "<what it does and when to use it>"`

Optional flags:
- `--root <path>` write to a different skill root (default: `.agentzero/skills`)
- `--with-scripts` create `scripts/` with a starter script
- `--with-references` create `references/REFERENCE.md`
- `--with-assets` create `assets/.gitkeep`
- `--force` overwrite existing files in that skill directory

## Output Requirements
- Always show the final absolute path of the created skill.
- If a name is invalid, normalize it and show both input and normalized name.
- Keep `SKILL.md` concise; move long details into `references/` files.

## Files
- Main scaffolder: `scripts/create_skill.py`
- Notes: `references/REFERENCE.md`
