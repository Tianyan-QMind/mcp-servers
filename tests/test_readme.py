import re
from pathlib import Path

import pytest

EXPECTED_TOOL_NAMES = (
    "list_available_quantum_backends",
    "get_quantum_backend",
    "get_quantum_experiment_result",
    "submit_quantum_experiment",
    "submit_quantum_experiment_batch",
    "estimate_quantum_expectation",
    "batch_estimate_quantum_expectation",
    "get_quantum_expectation_result",
)


def _readme_section(readme: str, heading: str) -> str:
    match = re.search(
        rf"^{re.escape(heading)}\n(?P<section>.*?)(?=^## |\Z)",
        readme,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    return match.group("section")


def _mvp_tool_names(readme: str) -> list[str]:
    section = _readme_section(readme, "当前 MVP 提供八个 Tool：")
    return re.findall(r"^- `([^`]+)`：", section, re.MULTILINE)


def _assert_mvp_tool_contract(readme: str) -> None:
    assert _mvp_tool_names(readme) == list(EXPECTED_TOOL_NAMES)


def _assert_result_query_contract(result_section: str) -> None:
    assert re.search(r"`status=1` 返回 `processing`，表示实验仍在处理中", result_section)
    assert re.search(r"`status=0` 返回\s*`completed`，并包含", result_section)
    assert re.search(
        r"默认会裁剪完整的 `result_status`.*?"
        r"`include_result_status` 设为 `true` 时才返回它",
        result_section,
        re.DOTALL,
    )


def test_readme_describes_controlled_submission_prerequisites() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    expected_sdk_prerequisite = (
        "实验结果查询和实验提交都还需要天衍云 SDK API 的网络访问能力，"
        "以及当前身份模式可用的 Tianyan SDK 凭证。"
    )

    assert "只读查询并受控提交实验的 Streamable HTTP MCP Server。" in readme
    assert expected_sdk_prerequisite in readme
    assert "缺少 SDK 凭证不会阻止 HTTP 或 stdio Server 启动" in readme
    assert (
        "调用 `get_quantum_experiment_result`、`submit_quantum_experiment` 或 "
        "`submit_quantum_experiment_batch` 时都会安全失败" in readme
    )
    assert (
        "提交实验还需要天衍云 SDK API 的网络访问能力，以及由 Server 端配置的 SDK 凭证。"
        not in readme
    )


def test_readme_documents_exactly_eight_tools_and_result_query_contract() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Server 公开八个 Tool" in readme
    assert "当前 MVP 提供八个 Tool" in readme
    _assert_mvp_tool_contract(readme)
    assert "get_quantum_experiment_result" in readme
    assert "query_id" in readme
    assert "include_result_status" in readme
    assert "readOnlyHint=true" in readme
    assert "destructiveHint=false" in readme
    assert "idempotentHint=true" in readme
    assert "openWorldHint=true" in readme

    assert "Server 公开三个 Tool" not in readme
    assert "当前 MVP 提供三个 Tool" not in readme
    assert "查询实验结果 Tool 尚未实现" not in readme


def test_readme_explains_safe_result_query_usage_and_boundaries() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    result_section = _readme_section(readme, "## 实验结果查询（只读）")

    assert '"query_id": "replace-with-an-existing-query-id"' in readme
    assert '"include_result_status": false' in readme
    assert "MCP Tool 输入" in readme
    assert "不是 shell 命令" in readme
    assert "完整测量结果可能很大" in readme
    assert "不轮询" in readme
    assert "不自动重试" in readme
    _assert_result_query_contract(result_section)
    assert "direct_sdk" in readme
    assert "复用" in readme and "SDK Open ID" in readme
    assert "https://qc.zdxlz.com" in readme
    assert "TIANYAN_LIVE_TEST=1" in readme
    assert "只读" in readme


def test_readme_contract_helpers_reject_wrong_documentation_samples() -> None:
    wrong_tools = """当前 MVP 提供八个 Tool：
- `list_available_quantum_backends`：...
- `get_quantum_backend`：...
- `submit_quantum_experiment`：...
## 前置条件
"""
    with pytest.raises(AssertionError):
        _assert_mvp_tool_contract(wrong_tools)

    wrong_result = """## 实验结果查询（只读）
`status=1` 返回 `completed`。
`status=0` 返回 `processing`。
## 受控实验提交
"""
    with pytest.raises(AssertionError):
        _assert_result_query_contract(_readme_section(wrong_result, "## 实验结果查询（只读）"))


def test_readme_documents_identity_mode_operations_without_write_quick_start() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    for required in (
        "TIANYAN_IDENTITY_MODE",
        "multi_tenant",
        "tianyan-mcp-admin",
        "tianyan-mcp-client",
        "SQLite",
        "Nginx",
        "TIANYAN_LIVE_TEST=1",
        "TIANYAN_LIVE_WRITE_TEST=1",
        "只读",
    ):
        assert required in readme
    assert "uv run tianyan-mcp-client configure --sdk-open-id" not in readme
    assert "$env:TIANYAN_SDK_OPEN_ID" + " =" not in readme
    assert "Nginx 将上游 Host 固定为 `127.0.0.1`" in readme
    assert "仅 `single_user` HTTP 模式" in readme
    assert "`multi_tenant`：每位用户使用自己的 MCP API Key" in readme
    assert "`direct_sdk`：Host 发送 SDK Open ID" in readme
    assert "拒绝 `TIANYAN_MCP_API_KEY`" in readme
    assert "[direct_sdk Host 配置](#direct_sdk-受控网络部署)" in readme


def test_readme_documents_direct_sdk_restricted_network_contract() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    for required in (
        "TIANYAN_IDENTITY_MODE=direct_sdk",
        "TIANYAN_DIRECT_SDK_FINGERPRINT_KEY",
        '"Authorization": "Bearer <TIANYAN_SDK_OPEN_ID>"',
        "不写入 SQLite",
        "受控网络测试",
        "不是公网生产安全基线",
        "multi_tenant",
        "回滚",
        "TIANYAN_LIVE_TEST=1 继续只读",
        "TIANYAN_LIVE_WRITE_TEST=1",
        "已存在的实验可使用只读结果查询 Tool 查看状态",
    ):
        assert required in readme

    assert "无需创建用户或绑定账号" in readme
    assert "仅监听 loopback" in readme
    assert "管理员、数据库或 account 路由" in readme
    assert "当前上游登录与实验提交地址固定为天衍云生产环境" in readme
    assert "测试环境密钥不兼容" in readme
    assert "无需修改服务器环境变量或重新部署" in readme
    assert "只有真实硬件提交" in readme
    assert (
        "模拟器提交保持 `TIANYAN_ALLOW_HARDWARE_SUBMISSION=false` 和 "
        "`allow_hardware=false`" in readme
    )
    assert re.search(r"TIANYAN_SDK_OPEN_ID\s*=\s*(?!<)[^\s`'\"]+", readme) is None
    assert "查询实验结果 Tool 尚未实现" not in readme

    command_prefixes = ("uv run", "curl", "Invoke-", "powershell", '{"jsonrpc"')
    for line in readme.splitlines():
        assert not (
            "submit_quantum_experiment" in line
            and any(command in line for command in command_prefixes)
        )


def test_readme_scopes_http_environment_variables_by_identity_mode() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "| `TIANYAN_IDENTITY_MODE` | 无（HTTP 必填）" in readme
    assert "| `TIANYAN_MCP_API_KEY` | 无 | 仅 `single_user` HTTP 模式必填" in readme
    assert "| `TIANYAN_DIRECT_SDK_FINGERPRINT_KEY` | 无 | 仅 `direct_sdk`" in readme
    assert "以下认证边界仅适用于 `single_user`" in readme
    assert "模型和 Tool 参数仍不会接收 SDK Open ID" in readme


def test_readme_single_user_http_quick_start_sets_required_identity_mode() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    quick_start = re.search(
        r"以下启动示例仅适用于 `single_user`；.*?```powershell\n(.*?)```",
        readme,
        re.DOTALL,
    )
    assert quick_start is not None
    assert "$env:TIANYAN_IDENTITY_MODE = 'single_user'" in quick_start.group(1)
    assert quick_start.group(1).index("TIANYAN_IDENTITY_MODE") < quick_start.group(1).index(
        "TIANYAN_MCP_API_KEY"
    )
    assert "| `TIANYAN_IDENTITY_MODE` | 无（HTTP 必填） |" in readme
    assert "| `TIANYAN_IDENTITY_MODE` | `single_user` |" not in readme


def test_readme_documents_optional_lab_name_without_write_quick_start() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    submission = _readme_section(readme, "## 受控实验提交")

    assert "四个必填输入和两个可选输入" in submission
    assert "`lab_name`：可选" in submission
    assert "完全不发送 `lab_name` 字段" in submission
    assert "空字符串或仅包含空白" in submission
    assert "`allow_hardware`：可选" in submission
    assert "submit_quantum_experiment_batch" in submission
    assert "2–20 条 QCIS" in submission

    command_prefixes = ("uv run", "curl", "Invoke-", "powershell", '{"jsonrpc"')
    for line in submission.splitlines():
        assert not (
            "submit_quantum_experiment" in line
            and any(command in line for command in command_prefixes)
        )
