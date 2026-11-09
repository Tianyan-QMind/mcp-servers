import pytest

from tianyan_mcp.errors import rejection_message, upstream_business_message


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (None, ""),
        ("not-an-object", ""),
        ([1, 2], ""),
        ({}, ""),
        ({"message": None}, ""),
        ({"message": "  "}, ""),
        ({"message": "拒绝原因"}, "拒绝原因"),
        ({"msg": "拒绝原因"}, "拒绝原因"),
        ({"message": "line1\nline2\t\tagain"}, "line1 line2 again"),
        ({"message": "a\x00b"}, "a b"),
        (
            {"msg": '{"CN":"中文原因","EN":"english reason"}'},
            "中文原因 english reason",
        ),
        ({"msg": '"scalar reason"'}, "scalar reason"),
        ({"message": "重复", "msg": "重复"}, "重复"),
    ],
)
def test_upstream_business_message_extracts_structured_text(payload: object, expected: str) -> None:
    assert upstream_business_message(payload) == expected


def test_upstream_business_message_truncates_to_limit() -> None:
    message = "原因" * 200
    result = upstream_business_message({"message": message}, limit=20)

    assert len(result) == 20
    assert result.endswith("…")


def test_rejection_message_appends_detail_or_keeps_fixed_text() -> None:
    assert rejection_message("固定文案", {"message": "上游原文"}) == "固定文案：上游原文"
    assert rejection_message("固定文案", None) == "固定文案"
    assert rejection_message("固定文案", {}) == "固定文案"


@pytest.mark.parametrize(
    "raw",
    [
        "java.io.FileNotFoundException: /opt/app/config.yml does not exist",
        "cannot open C:\\Users\\svc\\secret.pem for reading",
        "failed to load ~/.config/tianyan/credentials.json",
        "see https://qc.zdxlz.com/internal/debug?id=1 for details",
        (
            "java.lang.NullPointerException\n"
            "\tat com.zdxlz.qccp.service.SubmitService.submit(SubmitService.java:88)\n"
            "\tat com.zdxlz.qccp.Controller.handle(Controller.java:21)"
        ),
        "InternalError: bad state; read .env.production by mistake, key=id_rsa copied",
        "keystore /etc/app/server.jks and cert server.pem are unavailable",
    ],
    ids=[
        "unix-path",
        "windows-path",
        "home-path",
        "url",
        "java-stack",
        "exception-and-secret-files",
        "jks-and-pem",
    ],
)
def test_upstream_business_message_redacts_internal_details(raw: str) -> None:
    result = upstream_business_message({"message": raw})

    assert "[redacted]" in result
    for secret_fragment in (
        "/opt/app",
        "config.yml",
        "C:\\Users",
        "secret.pem",
        "~/.config",
        "credentials.json",
        "https://",
        "http://",
        "FileNotFoundException",
        "NullPointerException",
        "InternalError",
        "com.zdxlz",
        "SubmitService.java:88",
        ".env.production",
        "id_rsa",
        "server.jks",
        "server.pem",
    ):
        assert secret_fragment not in result


@pytest.mark.parametrize(
    "raw",
    [
        "qcCode 不存在",
        "门校验未通过",
        "shots 超出上限",
        "该账号无真实硬件提交权限",
        "gate verification failed",
        "Unknown exception",
        "Internal error while submitting",
    ],
)
def test_upstream_business_message_keeps_normal_business_text(raw: str) -> None:
    assert upstream_business_message({"message": raw}) == raw


def test_redaction_runs_before_truncation() -> None:
    raw = "/opt/secret/config.yml " + "拒绝 " * 40
    result = upstream_business_message({"message": raw}, limit=40)

    assert "/opt/secret/config.yml" not in result
    assert result.startswith("[redacted]")
    assert len(result) <= 40
    assert result.endswith("…")
