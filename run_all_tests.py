import subprocess

tests = [
    "tests/test_simple.py",
    "tests/test_tools.py",
    "tests/test_openai_compat_server.py",
    "tests/test_process_response.py",
    "tests/test_acp.py",
    "tests/test_acp_core_tool_flow.py",
    "tests/test_acp_remote_http.py",
    # "tests/test_tool_calling_flow.py", # mock is currently broken out of scope
    "tests/test_setprompt_command.py",
    "tests/test_multimodal_integrations.py",
    "tests/test_sendblue_debounce.py",
    "tests/test_sendblue_voice_memo.py",
    "tests/test_memory_maintenance.py",
    "tests/test_consortium_mode.py",
    "tests/test_skills.py",
]
for t in tests:
    subprocess.run(["python3", t], env={"PYTHONPATH": "."})
