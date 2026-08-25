import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYZER = PROJECT_ROOT / "scripts" / "07_analyze.py"

spec = importlib.util.spec_from_file_location(
    "analyzer07",
    ANALYZER,
)
assert spec is not None
assert spec.loader is not None

analyzer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analyzer
spec.loader.exec_module(analyzer)


def trajectory(*steps):
    return {"steps": list(steps)}


def agent_step(**kwargs):
    return {
        "source": "agent",
        "message": "",
        **kwargs,
    }


def test_structured_read_action():
    data = trajectory(
        agent_step(
            tool_calls=[{
                "tool_call_id": "r1",
                "function_name": "Read",
                "arguments": {
                    "file_path": "/workspace/tests/test_a.py",
                },
            }],
            observation={
                "results": [{
                    "source_call_id": "r1",
                    "content": "file contents",
                }]
            },
        )
    )

    actions, metrics = analyzer.extract_behavior_actions(data)

    assert len(actions) == 1
    assert actions[0]["category"] == "read"
    assert actions[0]["path"] == "/workspace/tests/test_a.py"
    assert actions[0]["action_origin"] == "structured_tool_call"
    assert metrics["behavioral_action_calls"] == 1


def test_codex_empty_write_stdin_is_polling():
    data = trajectory(
        agent_step(
            tool_calls=[{
                "tool_call_id": "w1",
                "function_name": "write_stdin",
                "arguments": {
                    "session_id": 5255,
                    "chars": "",
                    "yield_time_ms": 1000,
                },
            }],
            observation={
                "results": [{
                    "source_call_id": "w1",
                    "content": "Process running",
                }]
            },
        )
    )

    actions, metrics = analyzer.extract_behavior_actions(data)

    assert actions == []
    assert metrics["behavioral_action_calls"] == 0
    assert metrics["behavior_excluded_empty_write_stdin"] == 1


def test_codex_nonempty_write_stdin_is_shell_action():
    command = "rg -n 'LoadMainModule' internal/js\n"

    data = trajectory(
        agent_step(
            tool_calls=[{
                "tool_call_id": "w2",
                "function_name": "write_stdin",
                "arguments": {
                    "session_id": 92901,
                    "chars": command,
                },
            }],
            observation={
                "results": [{
                    "source_call_id": "w2",
                    "content": "internal/js/bundle.go:123",
                }]
            },
        )
    )

    actions, metrics = analyzer.extract_behavior_actions(data)

    assert len(actions) == 1
    assert actions[0]["category"] == "bash"
    assert actions[0]["command"] == command.strip()
    assert actions[0]["action_origin"] == "codex_write_stdin"
    assert metrics["behavior_codex_write_stdin_actions"] == 1


def test_update_plan_is_not_behavioral_action():
    data = trajectory(
        agent_step(
            tool_calls=[{
                "tool_call_id": "p1",
                "function_name": "update_plan",
                "arguments": {"plan": []},
            }]
        )
    )

    actions, metrics = analyzer.extract_behavior_actions(data)

    assert actions == []
    assert metrics["behavior_excluded_planning_events"] == 1


def test_miniswe_command_is_recovered_once():
    command = "cd /workspace && go build ./... && go test ./..."

    fenced = (
        "```mswea_bash_command\n"
        + command
        + "\n```"
    )

    data = trajectory(
        agent_step(
            message=fenced,
            reasoning_content=fenced,
            observation={
                "results": [{
                    "content": (
                        '{"returncode": 0, '
                        '"output": "ok"}'
                    ),
                }]
            },
        )
    )

    actions, metrics = analyzer.extract_behavior_actions(data)

    assert len(actions) == 1
    assert actions[0]["command"] == command
    assert actions[0]["category"] == "bash"
    assert actions[0]["failed"] == 0
    assert actions[0]["action_origin"] == "miniswe_message_command"
    assert metrics["behavior_miniswe_actions"] == 1


def test_miniswe_nested_failed_returncode():
    command = "cd /workspace && go test ./..."

    fenced = (
        "```mswea_bash_command\n"
        + command
        + "\n```"
    )

    data = trajectory(
        agent_step(
            message=fenced,
            reasoning_content=fenced,
            observation={
                "results": [{
                    "content": (
                        '{"returncode": 1, '
                        '"output": "compile failed"}'
                    ),
                }]
            },
        )
    )

    actions, _ = analyzer.extract_behavior_actions(data)

    assert len(actions) == 1
    assert actions[0]["failed"] == 1


def test_miniswe_completion_sentinel_is_excluded():
    fenced = (
        "```mswea_bash_command\n"
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"
        "```"
    )

    data = trajectory(
        agent_step(
            message=fenced,
            reasoning_content=fenced,
        )
    )

    actions, metrics = analyzer.extract_behavior_actions(data)

    assert actions == []
    assert metrics["behavior_excluded_completion_sentinel"] == 1


def test_miniswe_disagreeing_duplicate_fences_fail_closed():
    data = trajectory(
        agent_step(
            message=(
                "```mswea_bash_command\n"
                "pytest -q\n"
                "```"
            ),
            reasoning_content=(
                "```mswea_bash_command\n"
                "git status\n"
                "```"
            ),
        )
    )

    actions, metrics = analyzer.extract_behavior_actions(data)

    assert actions == []
    assert metrics["behavior_miniswe_unrecoverable_action_steps"] == 1


def test_plain_agent_text_is_not_unrecoverable_action():
    data = trajectory(
        agent_step(
            message="I finished the requested change.",
            reasoning_content="",
        )
    )

    actions, metrics = analyzer.extract_behavior_actions(data)

    assert actions == []
    assert metrics["behavior_non_action_agent_turns"] == 1
    assert metrics.get(
        "behavior_miniswe_unrecoverable_action_steps", 0
    ) == 0


def test_codex_web_search_call_is_behavioral_web_search():
    data = trajectory(
        agent_step(
            tool_calls=[{
                "tool_call_id": "ws1",
                "function_name": "web_search_call",
                "arguments": {
                    "query": "Grafana module resolver implementation",
                },
            }],
            observation={
                "results": [{
                    "source_call_id": "ws1",
                    "content": "search results",
                }]
            },
        )
    )

    actions, _ = analyzer.extract_behavior_actions(data)

    assert len(actions) == 1
    assert actions[0]["category"] == "web_search"


def test_agent_delegation():
    data = trajectory(
        agent_step(
            tool_calls=[{
                "tool_call_id": "a1",
                "function_name": "Agent",
                "arguments": {
                    "description": "Search repository",
                    "subagent_type": "Explore",
                    "prompt": "Find all references to Foo.",
                },
            }],
            observation={
                "results": [{
                    "source_call_id": "a1",
                    "content": "Async agent launched successfully.",
                }]
            },
        )
    )

    actions, metrics = analyzer.extract_behavior_actions(data)

    assert len(actions) == 1
    assert actions[0]["category"] == "delegate"
    assert actions[0]["action_origin"] == "subagent_delegation"
    assert metrics["behavior_subagent_delegation_actions"] == 1


def test_task_output_excluded():
    data = trajectory(
        agent_step(
            tool_calls=[{
                "tool_call_id": "to1",
                "function_name": "TaskOutput",
                "arguments": {
                    "task_id": "abc",
                    "block": True,
                },
            }]
        )
    )

    actions, metrics = analyzer.extract_behavior_actions(data)

    assert actions == []
    assert metrics[
        "behavior_excluded_task_orchestration_events"
    ] == 1


def test_task_stop_excluded():
    data = trajectory(
        agent_step(
            tool_calls=[{
                "tool_call_id": "ts1",
                "function_name": "TaskStop",
                "arguments": {
                    "task_id": "abc",
                },
            }]
        )
    )

    actions, metrics = analyzer.extract_behavior_actions(data)

    assert actions == []
    assert metrics[
        "behavior_excluded_task_orchestration_events"
    ] == 1
