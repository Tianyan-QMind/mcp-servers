import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALIYUN_INDEX = "https://mirrors.aliyun.com/pypi/simple/"


def test_cqlib_is_pinned_to_verified_python_binding_source() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "cqlib==1.4.0b1" in project["project"]["dependencies"]
    assert project["tool"]["uv"]["sources"]["cqlib"] == {
        "git": "https://github.com/cq-lib/cqlib",
        "rev": "1d0a2c49ac32712d995f46147dfc5e3c4f4ac8e6",
        "subdirectory": "crates/binding-python",
    }


def test_native_sdk_build_tools_are_confined_to_builder() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    builder, runtime = dockerfile.split("FROM ${PYTHON_IMAGE} AS runtime")
    assert "rust:1.93-bookworm@sha256:" in builder
    assert "COPY --from=rust /usr/local/cargo /usr/local/cargo" in builder
    assert "COPY --from=rust /usr/local/rustup /usr/local/rustup" in builder
    assert "build-essential" in builder and " git" in builder
    assert "cargo" not in runtime and "rustup" not in runtime


def _runtime_stage_instructions(dockerfile: str) -> list[str]:
    runtime = dockerfile.split("FROM ${PYTHON_IMAGE} AS runtime", maxsplit=1)[1]
    instructions: list[str] = []
    instruction_parts: list[str] = []

    for line in runtime.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        has_continuation = stripped.endswith("\\")
        instruction_parts.append(stripped[:-1].rstrip() if has_continuation else stripped)
        if has_continuation:
            continue

        instructions.append(" ".join(instruction_parts))
        instruction_parts.clear()

    assert not instruction_parts, "runtime stage has an unterminated instruction"
    return instructions


def test_dockerfile_builds_locked_non_root_runtime() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert (
        "ARG PYTHON_IMAGE=python:3.11-slim@sha256:"
        "a630a63cdb314e2d138a2fca3e375e319e8568346ffafac5b980f888630ac4f1"
    ) in dockerfile
    assert "FROM ${PYTHON_IMAGE} AS builder" in dockerfile
    assert "python -m pip install" in dockerfile
    assert "--no-cache-dir" in dockerfile
    assert "--index-url" in dockerfile
    assert "uv==0.11.32" in dockerfile
    assert "uv sync --frozen --no-dev --no-editable" in dockerfile
    assert "FROM ${PYTHON_IMAGE} AS runtime" in dockerfile
    assert "COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert 'ENTRYPOINT ["/app/.venv/bin/tianyan-mcp-http"]' in dockerfile
    assert "pytest" not in dockerfile.lower()
    assert "TIANYAN_SDK_OPEN_ID" not in dockerfile
    assert "TIANYAN_MCP_API_KEY" not in dockerfile


def test_python_packages_use_one_aliyun_https_index() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["tool"]["uv"]["index"] == [
        {
            "name": "aliyun",
            "url": ALIYUN_INDEX,
            "default": True,
        }
    ]


def test_dockerfile_bootstraps_uv_from_project_index() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert f"--index-url {ALIYUN_INDEX}" in dockerfile
    assert "uv==0.11.32" in dockerfile
    assert "uv sync --frozen --no-dev --no-editable" in dockerfile
    assert "--trusted-host" not in dockerfile
    assert "http://mirrors.aliyun.com" not in dockerfile


def test_lockfile_contains_only_aliyun_registry_urls() -> None:
    lockfile = (ROOT / "uv.lock").read_text(encoding="utf-8")

    assert ALIYUN_INDEX.rstrip("/") in lockfile
    assert "https://pypi.org" not in lockfile
    assert "https://files.pythonhosted.org" not in lockfile
    assert "http://" not in lockfile


def test_dockerfile_runtime_avoids_writable_source_install() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    runtime_instructions = _runtime_stage_instructions(dockerfile)
    runtime_instruction_names = [
        instruction.split(maxsplit=1)[0].upper() for instruction in runtime_instructions
    ]
    copy_instructions = [
        instruction
        for instruction in runtime_instructions
        if instruction.split(maxsplit=1)[0].upper() == "COPY"
    ]

    assert "COPY . ." not in dockerfile
    assert runtime_instruction_names == ["ENV", "WORKDIR", "COPY", "USER", "ENTRYPOINT"]
    assert copy_instructions == ["COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv"]
    assert "PYTHONDONTWRITEBYTECODE=1" in runtime_instructions[0]
    assert "PYTHONUNBUFFERED=1" in runtime_instructions[0]


def test_dockerignore_excludes_secrets_and_local_state() -> None:
    ignored = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    required = {
        ".git",
        ".env",
        ".env.*",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".superpowers",
        "info.txt",
        "*.sqlite",
        "*.sqlite3",
        "*.db",
        "__pycache__",
        "tests",
    }
    assert required <= ignored
    required_recursive = {
        "**/.git",
        "**/.env",
        "**/.env.*",
        "**/.venv",
        "**/.pytest_cache",
        "**/.ruff_cache",
        "**/.mypy_cache",
        "**/.pyright",
        "**/__pycache__",
        "**/*.py[cod]",
        "**/*.sqlite",
        "**/*.sqlite3",
        "**/*.db",
        "**/*.log",
        "**/info.txt",
    }
    assert required_recursive <= ignored
    assert not [line for line in ignored if line.startswith("!")], (
        "negated Docker ignore rules can reinclude protected build-context paths"
    )
