import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "deployment" / "ctyunos-arm64-test-deployment.txt"
DOCKERFILE = ROOT / "Dockerfile"


def _runbook() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def _phase(number: str) -> str:
    match = re.search(
        rf"^第{number}阶段：.*?(?=^第[^\n]+阶段：|\Z)",
        _runbook(),
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    return match.group(0)


def _dockerfile_python_image() -> str:
    match = re.search(
        r"^ARG PYTHON_IMAGE=(\S+)$",
        DOCKERFILE.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert match is not None
    return match.group(1)


def _assert_native_command_guarded(phase: str, command_prefix: str, failure_message: str) -> None:
    lines = [line.strip() for line in phase.splitlines() if line.strip()]
    command_index = next(
        index for index, line in enumerate(lines) if line.startswith(command_prefix)
    )
    assert lines[command_index + 1] == (f"if ($LASTEXITCODE -ne 0) {{ throw '{failure_message}' }}")


def test_runbook_covers_verified_server_and_release_inputs() -> None:
    text = _runbook()

    required = (
        "CTyunOS 2.0.1",
        "aarch64",
        "/mnt/sdc/MCPserver",
        "docker-compose version",
        "docker inspect nginx-test",
        "127.0.0.1:8000 is free",
        "git archive",
        "Get-FileHash -Algorithm SHA256",
        'sha256sum -- "$ARCHIVE"',
    )
    for value in required:
        assert value in text


def test_release_transfer_matches_vm_and_browser_bastion_path() -> None:
    publication = _phase("三")

    for required in (
        "公网开发机",
        "VPN 虚拟机",
        "网页堡垒机",
        "/mnt/sdc/MCPserver/deploy",
        "$ArchiveName",
        "$HashFile",
        "两个文件",
    ):
        assert required in publication
    assert "scp $Archive" not in publication


def test_reconnectable_phases_define_mcp_root_before_use() -> None:
    for number in ("四", "十四"):
        phase = _phase(number)
        assignment = "MCP_ROOT=/mnt/sdc/MCPserver"

        assert assignment in phase
        assert phase.index("set -euo pipefail") < phase.index(assignment)
        first_expansion = phase.index("$MCP_ROOT")
        assert phase.index(assignment) < first_expansion


def test_runbook_initializes_secrets_without_printing_or_overwriting() -> None:
    text = _runbook()

    assert "runtime.env 已存在或是符号链接：停止" in text
    assert "TIANYAN_IDENTITY_MODE=multi_tenant" in text
    assert "TIANYAN_DATABASE_PATH=/data/tenants.sqlite3" in text
    assert "TIANYAN_ALLOW_HARDWARE_SUBMISSION=false" in text
    assert "sed 's/=.*$/=[REDACTED]/'" in text
    assert "cat /mnt/sdc/MCPserver/secrets/runtime.env" not in text
    assert "TIANYAN_LIVE_WRITE_TEST=1" in text


def test_runtime_env_creation_rejects_links_and_cannot_clobber() -> None:
    text = _runbook()

    assert 'test -e "$RUNTIME_ENV" || test -L "$RUNTIME_ENV"' in text
    assert "set -o noclobber" in text
    assert "set -e" in text
    assert '> "$RUNTIME_ENV"' in text


def test_runbook_uses_existing_nginx_with_three_exact_routes() -> None:
    text = _runbook()

    assert "listen 30002" in text
    for route in ("/mcp", "/account/sdk-credential", "/healthz"):
        assert f"location = {route}" in text
    assert "proxy_pass http://127.0.0.1:8000" in text
    assert "proxy_set_header Host 127.0.0.1" in text
    assert "proxy_buffering off" in text
    assert "proxy_request_buffering off" in text
    assert text.index("nginx -t") < text.index("nginx -s reload")


def test_runbook_keeps_http_test_credentials_inside_ssh_tunnel() -> None:
    text = _runbook()

    assert "ssh -N -L 18000:127.0.0.1:30002" in text
    assert "http://127.0.0.1:18000" in text
    client_lines = [line for line in text.splitlines() if "tianyan-mcp-client" in line]
    assert client_lines
    assert all("http://127.0.0.1:18000" in line for line in client_lines)
    assert "30002 只允许可信内网访问" in text


def test_runbook_has_health_auth_persistence_backup_and_stop_conditions() -> None:
    text = _runbook()

    required = (
        "docker-compose config",
        "docker-compose build",
        "docker-compose up -d",
        "docker-compose ps",
        "docker-compose logs --tail=100",
        "tianyan-mcp-admin create-user",
        "tianyan-mcp-admin backup",
        "HTTP 200",
        "HTTP 401",
        "查询实验结果 Tool 尚未实现",
        "不要直接重复提交",
    )
    for value in required:
        assert value in text


def test_runbook_preflights_aliyun_uv_before_build() -> None:
    text = _runbook()

    assert "https://mirrors.aliyun.com/pypi/simple/" in text
    assert "python -m pip index versions uv" in text
    assert "0.11.32" in text
    assert 'grep -E "(^|[ ,])0\\.11\\.32($|[ ,])"' in text
    assert text.index("python -m pip index versions uv") < text.index("docker-compose build")


def test_failed_first_build_releases_without_overwriting_app() -> None:
    text = _runbook()

    assert 'RELEASES_ROOT="$MCP_ROOT/releases"' in text
    assert 'RELEASE_DIR="$RELEASES_ROOT/$RELEASE_COMMIT"' in text
    assert 'ARCHIVE="tianyan-mcp-server-$RELEASE_COMMIT.tar.gz"' in text
    assert 'test ! -e "$RELEASE_DIR" && test ! -L "$RELEASE_DIR"' in text
    assert 'tar --keep-old-files -xzf "$ARCHIVE_PATH" -C "$RELEASE_DIR"' in text
    assert 'cd "$RELEASE_DIR"' in text
    assert "rm -" not in text
    assert "mv " not in text


def test_compose_config_validation_is_quiet_only() -> None:
    command_lines = [
        line.strip()
        for line in _runbook().splitlines()
        if line.strip().startswith("docker-compose config")
    ]

    assert command_lines == ["docker-compose config --quiet"]


def test_archive_manifest_is_compared_to_current_archive_before_safe_extract() -> None:
    text = _runbook()

    required_in_order = (
        'tr -d "\\\\r" < "$HASH_PATH" | wc -l',
        'EXPECTED_HASH_LINE=$(tr -d "\\\\r" < "$HASH_PATH")',
        'ACTUAL_HASH_LINE=$(cd "$MCP_ROOT/deploy" && sha256sum -- "$ARCHIVE")',
        'test "$EXPECTED_HASH_LINE" = "$ACTUAL_HASH_LINE"',
        'tar --keep-old-files -xzf "$ARCHIVE_PATH"',
    )
    positions = [text.index(value) for value in required_in_order]
    assert positions == sorted(positions)
    assert "sha256sum --check" not in text


def test_runbook_contains_no_destructive_cleanup_or_real_secret_examples() -> None:
    text = _runbook()

    forbidden = (
        "rm -",
        "docker system prune",
        "docker volume prune",
        "TIANYAN_SDK_OPEN_ID=",
        "TIANYAN_MCP_API_KEY=",
    )
    for value in forbidden:
        assert value not in text
    assert re.search(r"tymcp_[A-Za-z0-9_-]{20,}", text) is None


def test_runbook_documents_direct_sdk_switch_and_rollback_contract() -> None:
    text = _runbook()
    direct_phase = _phase("十五")
    assert "回滚步骤：" in direct_phase
    switch, rollback = direct_phase.split("回滚步骤：", maxsplit=1)

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
        "查询实验结果 Tool 尚未实现",
    ):
        assert required in text

    assert "不创建用户或绑定账号" in direct_phase
    assert "仅重建 MCP 容器" in direct_phase
    assert "认证后的 account 路由返回 404" in direct_phase
    assert "保留 SQLite、API Pepper、加密密钥、备份和 Nginx 容器" in direct_phase
    assert "secrets.token_bytes(32)" in switch
    assert "TIANYAN_MCP_API_KEY=" not in switch
    assert "TIANYAN_SDK_OPEN_ID=" not in switch

    release_dir = 'RELEASE_DIR="$MCP_ROOT/releases/$RELEASE_COMMIT"'
    compose = "docker-compose up -d --no-deps --force-recreate tianyan-mcp-server"
    healthcheck = "docker inspect --format '{{.State.Health.Status}}' tianyan-mcp-server"
    account_404 = "raise SystemExit(0 if e.code == 404 else 1)"
    for procedure in (switch, rollback):
        assert "read -r -p '输入当前运行版本的40位Git提交SHA: ' RELEASE_COMMIT" in procedure
        assert 'test "${#RELEASE_COMMIT}" -eq 40' in procedure
        assert release_dir in procedure
        assert 'test -d "$RELEASE_DIR" && test ! -L "$RELEASE_DIR"' in procedure
        assert procedure.index(release_dir) < procedure.index('cd "$RELEASE_DIR"')
        assert compose in procedure
        assert (
            "docker-compose up -d --no-deps --force-recreate\ntianyan-mcp-server" not in procedure
        )
        assert healthcheck in procedure
        assert "http://127.0.0.1:8000/healthz" in procedure

    assert switch.index(compose) < switch.index(healthcheck) < switch.index(account_404)
    assert switch.count('test "$(stat -c \'%a:%U:%G\' "$RUNTIME_ENV")"') == 2
    assert (
        switch.index("unset DIRECT_FINGERPRINT_LINE")
        < switch.rindex('test "$(stat -c \'%a:%U:%G\' "$RUNTIME_ENV")"')
        < switch.index(compose)
    )

    host_config = re.search(r"受控网络 MCP Host.*?\n\n(\{.*?\n\})", switch, re.DOTALL)
    assert host_config is not None
    assert '"url": "http://<restricted-network-host>:<port>/mcp"' in host_config.group(1)
    assert '"Authorization": "Bearer <TIANYAN_SDK_OPEN_ID>"' in host_config.group(1)
    assert "127.0.0.1" not in host_config.group(1)
    assert account_404 in switch

    command_prefixes = ("uv run", "curl", "Invoke-", "powershell", '{"jsonrpc"')
    for line in text.splitlines():
        assert not (
            "submit_quantum_experiment" in line
            and any(command in line for command in command_prefixes)
        )


def test_helper_containers_use_the_pinned_dockerfile_image() -> None:
    text = _runbook()
    preflight = _phase("一")
    secrets = _phase("四")
    build = _phase("六")
    direct_sdk = _phase("十五")
    python_image = _dockerfile_python_image()

    assert text.count(python_image) == 4
    assert text.count('"$PYTHON_HELPER_IMAGE"') == 4
    assert "--network container:nginx-test" in preflight
    assert "--network container:nginx-test" in build
    assert "--network none" in secrets
    for phase in (preflight, secrets, build, direct_sdk):
        assert "--read-only" in phase
        assert "--cap-drop ALL" in phase
        assert "--security-opt no-new-privileges" in phase
        assert "--user 65534:65534" in phase
    assert re.search(r"docker run[^\n]*(?:\\\n[^\n]*)*python:3\.11-slim(?!@)", text) is None


def test_server_bash_command_phases_are_fail_fast() -> None:
    first_commands = {
        "一": "PYTHON_HELPER_IMAGE=",
        "二": "MCP_ROOT=",
        "四": "MCP_ROOT=",
        "七": "docker exec tianyan-mcp-server",
        "八": "NGINX_CONF=",
        "九": "docker exec tianyan-mcp-server",
        "十": "docker exec -it tianyan-mcp-server",
        "十三": "MCP_ROOT=",
        "十四": "MCP_ROOT=",
    }
    for number, first_command in first_commands.items():
        phase = _phase(number)
        assert phase.count("\nset -euo pipefail\n") == 1
        assert phase.index("\nset -euo pipefail\n") < phase.index(first_command)


def test_release_and_build_fail_fast_without_exiting_parent_shell() -> None:
    for number in ("五", "六"):
        phase = _phase(number)

        assert 'if RELEASE_COMMIT="$RELEASE_COMMIT"' in phase
        assert "bash -euo pipefail -c '\n" in phase
        assert f'echo "第{number}阶段失败：保留现场并停止后续操作"' in phase


def test_external_bash_fail_fast_stops_child_but_keeps_parent_alive() -> None:
    if os.name == "nt":
        candidates = (
            Path(r"C:\Program Files\Git\bin\bash.exe"),
            Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
        )
    else:
        located = shutil.which("bash")
        candidates = (Path(located),) if located is not None else ()

    bash: Path | None = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        version = subprocess.run(  # noqa: S603
            [str(candidate), "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        if version.returncode == 0 and "GNU bash" in version.stdout:
            bash = candidate
            break
    if bash is None:
        pytest.skip("validated GNU Bash unavailable; CTyunOS Bash verification is target-deferred")

    script = """\
set -e
if "$BASH" -euo pipefail -c 'printf "before\\n"; false; printf "after\\n"'; then
  printf "unexpected-success\\n"
else
  printf "handler\\n"
fi
printf "parent-alive\\n"
"""
    completed = subprocess.run(  # noqa: S603
        [str(bash), "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout.splitlines() == ["before", "handler", "parent-alive"]


def test_windows_native_commands_are_checked_immediately() -> None:
    publication = _phase("三")
    client = _phase("十一")

    assert "$ErrorActionPreference = 'Stop'" in publication
    for command, message in (
        ("$Commit = git", "git rev-parse failed"),
        ("$Status = @(git", "git status failed"),
        ("git -C $Repo archive", "git archive failed"),
    ):
        _assert_native_command_guarded(publication, command, message)

    assert "$ErrorActionPreference = 'Stop'" in client
    for command, message in (
        ("ssh -N", "ssh tunnel failed"),
        (
            "& $Uv --directory $Repo run tianyan-mcp-client configure",
            "credential configure failed",
        ),
        (
            "& $Uv --directory $Repo run tianyan-mcp-client status",
            "credential status failed",
        ),
    ):
        _assert_native_command_guarded(client, command, message)


def test_nginx_auth_probes_require_exactly_401() -> None:
    phase = _phase("九")

    for route in ("/mcp", "/account/sdk-credential"):
        route_start = phase.index(f'30002{route}"')
        strict_exit = phase.index("raise SystemExit(0 if e.code == 401 else 1)", route_start)
        assert route_start < strict_exit
    assert phase.count("raise SystemExit(0 if e.code == 401 else 1)") == 2


def test_restart_waits_for_nginx_and_keeps_host_reboot_drill_deferred() -> None:
    phase = _phase("十三")

    assert "nginx-test must be running" in phase
    assert "docker inspect" in phase
    assert "docker-compose up -d" in phase
    assert "真实宿主机重启演练保持 target-deferred" in phase


def test_readme_links_runbook_and_labels_it_non_production() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "docs/deployment/ctyunos-arm64-test-deployment.txt" in readme
    assert "CTyunOS ARM64 内网测试部署" in readme
    assert "不用于正式生产开放" in readme
    assert "/mnt/sdc/MCPserver/releases/$RELEASE_COMMIT" in readme
    assert "不覆盖已有 app" in readme


def test_direct_sdk_runbook_documents_production_credentials_and_write_boundaries() -> None:
    phase = _phase("十五")

    assert "当前上游登录与实验提交地址固定为天衍云生产环境" in phase
    assert "测试环境密钥不兼容" in phase
    assert "无需修改服务器环境变量或重新部署" in phase
    assert "所有真实提交" in phase
    assert "只有真实硬件提交" in phase
    assert "模拟器提交保持 TIANYAN_ALLOW_HARDWARE_SUBMISSION=false 和 allow_hardware=false" in phase
