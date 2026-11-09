from pydantic import SecretStr

from tianyan_mcp.identity import DirectSdkPrincipal, current_direct_sdk, direct_sdk_scope


def principal(value: str = "fake-open-id") -> DirectSdkPrincipal:
    return DirectSdkPrincipal(SecretStr(value), b"f" * 32, "request-1")


def test_direct_sdk_scope_sets_and_resets_principal() -> None:
    with direct_sdk_scope(principal()):
        assert current_direct_sdk().sdk_fingerprint == b"f" * 32

    try:
        current_direct_sdk()
    except RuntimeError as error:
        assert str(error) == "direct SDK context is unavailable"
    else:
        raise AssertionError("direct context leaked after scope exit")


def test_tenant_and_direct_contexts_are_independent() -> None:
    with direct_sdk_scope(principal("alice")):
        assert current_direct_sdk().sdk_open_id.get_secret_value() == "alice"
