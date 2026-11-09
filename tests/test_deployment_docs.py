import re
from pathlib import Path


def test_nginx_example_has_safe_multi_tenant_boundaries() -> None:
    text = Path("docs/deployment/nginx-multi-tenant.conf.example").read_text(encoding="utf-8")

    for required in (
        "listen 443 ssl",
        "server_name <operator-managed-domain>",
        "ssl_certificate <operator-managed-certificate-path>",
        "ssl_certificate_key <operator-managed-private-key-path>",
        "location = /mcp",
        "location = /account/sdk-credential",
        "location = /healthz",
        "proxy_pass http://127.0.0.1:8000",
        "proxy_buffering off",
        "client_max_body_size 64k",
        "access_log off",
        "deny all;",
        "proxy_set_header Host 127.0.0.1;",
    ):
        assert required in text

    locations = re.findall(r"^\s*location\s+(?:=\s+)?(\S+)", text, flags=re.MULTILINE)
    assert locations == ["/mcp", "/account/sdk-credential", "/healthz"]
    assert text.count("allow all;") == 3
    assert "$host" not in text
    assert "$http_authorization" not in text
    assert "$request_body" not in text
