import os
import subprocess
import sys


tests = [
    "tests/test_simple.py",
    "tests/test_tools.py",
    "tests/test_openai_compat_server.py",
    "tests/test_agentic_loop.py",
    "tests/test_process_response.py",
    "tests/test_acp.py",
    "tests/test_acp_core_tool_flow.py",
    "tests/test_acp_remote_http.py",
    # "tests/test_tool_calling_flow.py", # mock is currently broken out of scope
    "tests/test_setprompt_command.py",
    "tests/test_telegram_message_flow.py",
    "tests/test_multimodal_integrations.py",
    "tests/test_sendblue_debounce.py",
    "tests/test_sendblue_voice_memo.py",
    "tests/test_memory_maintenance.py",
    "tests/test_consortium_mode.py",
    "tests/test_skills.py",
    "tests/test_injection_scanner.py",
]


def main() -> int:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "." if not existing_pythonpath else f".:{existing_pythonpath}"

    failures: list[str] = []

    for test_path in tests:
        result = subprocess.run([sys.executable, test_path], env=env)
        if result.returncode != 0:
            failures.append(test_path)

    if failures:
        print("\nFailed tests:")
        for test_path in failures:
            print(f" - {test_path}")
        return 1

    print("\nAll tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
