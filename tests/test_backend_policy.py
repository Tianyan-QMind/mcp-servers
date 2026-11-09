import pytest

from tianyan_mcp.backend_policy import resolve_submission_backend
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.models import BackendType


@pytest.mark.parametrize(
    ("provided", "canonical", "lookup", "resource_type"),
    [
        ("tianyan-176", "tianyan176", "tianyan176", BackendType.QUANTUM_COMPUTER),
        ("tianyan176", "tianyan176", "tianyan176", BackendType.QUANTUM_COMPUTER),
        ("tianyan176-2", "tianyan176-2", "tianyan176-2", BackendType.QUANTUM_COMPUTER),
        ("tianyan24", "tianyan24", "tianyan24", BackendType.QUANTUM_COMPUTER),
        ("tianyan-504", "tianyan504", "tianyan504", BackendType.QUANTUM_COMPUTER),
        ("tianyan-287", "tianyan-287", "tianyan-287", BackendType.QUANTUM_COMPUTER),
        ("tianyan287", "tianyan-287", "tianyan-287", BackendType.QUANTUM_COMPUTER),
        ("tianyan-294", "tianyan-294", "tianyan-294", BackendType.QUANTUM_COMPUTER),
        ("tianyan294", "tianyan-294", "tianyan-294", BackendType.QUANTUM_COMPUTER),
        ("tianyan_sw", "tianyan_sw", "tianyan_sw", BackendType.SIMULATOR),
        (" tianyan_s ", "tianyan_s", "tianyan_s", BackendType.SIMULATOR),
    ],
)
def test_resolves_only_explicit_backend_names(
    provided: str,
    canonical: str,
    lookup: str,
    resource_type: BackendType,
) -> None:
    target = resolve_submission_backend(provided)
    assert target.canonical_code == canonical
    assert target.lookup_code == lookup
    assert target.resource_type is resource_type


@pytest.mark.parametrize(
    "name", ["tianyan_tnn", "tianyan285", "tianyan-285", "TIANYAN_SW", "unknown"]
)
def test_rejects_backends_outside_the_submission_whitelist(name: str) -> None:
    with pytest.raises(ServiceError) as captured:
        resolve_submission_backend(name)
    assert captured.value.code == "UNSUPPORTED_BACKEND"
    assert captured.value.retryable is False
