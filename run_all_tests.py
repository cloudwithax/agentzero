import asyncio
import os
import sys
import time


tests = [
    "tests/test_simple.py",
    "tests/test_tools.py",
    "tests/test_openai_compat_server.py",
    "tests/test_agentic_loop.py",
    "tests/test_process_response.py",
    "tests/test_acp.py",
    "tests/test_acp_core_tool_flow.py",
    "tests/test_acp_remote_http.py",
    "tests/test_tool_calling_flow.py",
    "tests/test_openai_compat_tool_calls.py",
    "tests/test_setprompt_command.py",
    "tests/test_telegram_message_flow.py",
    "tests/test_telegram_voice_memo.py",
    "tests/test_multimodal_integrations.py",
    "tests/test_sendblue_debounce.py",
    "tests/test_sendblue_voice_memo.py",
    "tests/test_memory_maintenance.py",
    "tests/test_orchestrated_pipeline.py",
    "tests/test_consortium_mode.py",
    "tests/test_skills.py",
    "tests/test_injection_scanner.py",
    "tests/test_self_heal.py",
    "tests/test_reminder_tasks.py",
    "tests/test_credentials.py",
]

CONCURRENCY = int(os.environ.get("AGENTZERO_TEST_CONCURRENCY", "6"))
STAGGER_SECONDS = float(os.environ.get("AGENTZERO_TEST_STAGGER", "0.3"))


async def _run_one(test_path: str, env: dict[str, str]) -> tuple[str, int, str, str]:
    """Run a single test and return (path, returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        test_path,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    out_str = stdout.decode(errors="replace")
    err_str = stderr.decode(errors="replace")
    return (test_path, proc.returncode or 0, out_str, err_str)


async def main_async() -> int:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "." if not existing_pythonpath else f".:{existing_pythonpath}"
    if "AGENTZERO_LIVE_TESTS" not in env:
        env["AGENTZERO_LIVE_TESTS"] = "1"

    sem = asyncio.Semaphore(CONCURRENCY)

    async def _bounded_run(test_path: str, index: int) -> tuple[str, int, str, str]:
        async with sem:
            # Stagger start to avoid thundering-herd on live API tests.
            await asyncio.sleep(index * STAGGER_SECONDS)
            print(f"[{index+1}/{len(tests)}] Running {test_path} ...", flush=True)
            t0 = time.monotonic()
            result = await _run_one(test_path, env)
            elapsed = time.monotonic() - t0
            status = "PASS" if result[1] == 0 else "FAIL"
            print(f"      ({test_path}) → {status}  ({elapsed:.1f}s)", flush=True)
            return result

    tasks = [_bounded_run(p, i) for i, p in enumerate(tests)]
    results = await asyncio.gather(*tasks)

    failures: list[tuple[str, str]] = []
    for path, rc, stdout, stderr in results:
        if rc != 0:
            failures.append((path, stderr.strip() or stdout.strip()))

    print()
    if failures:
        print(f"FAILED: {len(failures)}/{len(tests)}")
        for path, err in failures:
            print(f"\n── {path} ──")
            # Show last 40 lines of error output.
            lines = err.splitlines()
            for line in lines[-40:]:
                print(f"    {line}")
        return 1

    print(f"All {len(tests)} tests passed.", flush=True)
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
