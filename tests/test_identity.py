import pytest

from tianyan_mcp.identity import TenantPrincipal, current_tenant, tenant_scope


def test_tenant_scope_is_request_local() -> None:
    principal = TenantPrincipal("tenant-a", "key-a", 1)

    with tenant_scope(principal):
        assert current_tenant() == principal

    with pytest.raises(RuntimeError, match="tenant context"):
        current_tenant()
