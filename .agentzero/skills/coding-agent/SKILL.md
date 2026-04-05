---
name: coding-agent
description: "Structured workflow for multi-file coding projects. Activated automatically when a task involves creating or modifying multiple files, architectural decisions, library integration, or feature implementation in a codebase. Not for simple scripts or one-off snippets."
metadata:
  author: agentzero
  purpose: default-builtin-skill
---

# coding agent

structured workflow for tackling coding projects. follow these phases in order. do not skip phases.

## when this applies

this workflow is for coding PROJECTS, not scripts. use it when:

- creating or modifying multiple files
- adding a feature to an existing codebase
- integrating a library or framework
- refactoring across module boundaries
- fixing bugs that span multiple components
- setting up project scaffolding

do NOT use this for:
- writing a single script
- answering code questions
- one-file snippets or examples
- simple config changes

## phase 0: clarify (only if needed)

if the request is ambiguous about scope, tech stack, or expected behavior, ask ONE focused question. do not ask multiple questions or over-clarify obvious requirements.

skip this phase if the task is clear enough to proceed.

## phase 1: reconnaissance

before writing ANY code, understand the terrain:

1. **project structure** - use `glob` to map the file tree
   - identify entrypoints, config files, test directories
   - note the package manager, build tool, and framework
2. **existing patterns** - use `grep` + `read` to study conventions
   - naming conventions (camelCase, snake_case, etc)
   - import style (relative vs aliases)
   - error handling patterns
   - existing abstractions and utilities
3. **dependencies** - check package.json/cargo.toml/go.mod/etc
   - what libraries are already available
   - do NOT introduce new dependencies unless necessary
4. **tests** - find existing test files and patterns
   - test framework in use
   - test file naming convention
   - assertion style

output a brief summary of findings. this prevents wasted work.

## phase 2: plan

create a concrete implementation plan:

1. **files to create** - list each new file with its purpose
2. **files to modify** - list each existing file and what changes
3. **order of operations** - which files to touch first
4. **interfaces** - define contracts between modules before implementing
5. **verification** - how to confirm it works (test commands, manual checks)

keep the plan short. 5-15 lines max. this is a checklist, not a design document.

## phase 3: implement

execute the plan incrementally:

1. **smallest coherent changes** - one logical change at a time
2. **types/interfaces first** - define shapes before implementations
3. **core logic next** - implement the main functionality
4. **integration last** - wire everything together
5. **match existing style** - your code should look like it belongs

rules:
- use `edit` for modifying existing files (never rewrite entire files unless necessary)
- use `write` for creating new files
- use `read` to verify your changes after writing
- do not leave TODOs, placeholders, or stub implementations
- do not add comments explaining obvious code
- all comments lowercase, no punctuation

## phase 4: verify

confirm the implementation works:

1. **read back** - re-read modified files to catch errors
2. **run tests** - if test commands are available, run them
3. **run typecheck/lint** - if applicable (bun run check, tsc, eslint, etc)
4. **check imports** - verify all imports resolve
5. **check integration** - make sure new code is properly wired in

if verification fails, fix the issue and re-verify. do not move on with known broken code.

## phase 5: summarize

brief summary of what was done:

- files created/modified (as a list)
- key decisions made and why
- how to test or verify
- any follow-up work needed

keep it to 3-5 lines. the user can read the code for details.

## principles

- **read before write** - always understand existing code before changing it
- **match the codebase** - adopt existing patterns, dont impose new ones
- **incremental progress** - small changes that build on each other
- **no dead code** - every line you write should be reachable and necessary
- **production quality** - no placeholders, no "example" code, no shortcuts
- **verify everything** - assume your code has bugs until proven otherwise
