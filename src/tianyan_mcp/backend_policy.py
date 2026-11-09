from __future__ import annotations

from dataclasses import dataclass

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.models import BackendType

QUANTUM_BACKENDS = frozenset(
    {
        "tianyan176",
        "tianyan176-2",
        "tianyan24",
        "tianyan504",
        "tianyan-287",
        "tianyan-294",
    }
)
SIMULATOR_BACKENDS = frozenset(
    {"tianyan_sw", "tianyan_s", "tianyan_tn", "tianyan_sa", "tianyan_swn"}
)
BACKEND_ALIASES = {
    "tianyan-176": "tianyan176",
    "tianyan-176-2": "tianyan176-2",
    "tianyan-24": "tianyan24",
    "tianyan-504": "tianyan504",
    "tianyan287": "tianyan-287",
    "tianyan294": "tianyan-294",
}
# Upstream computerCode (canonical_code) is not always identical to the code
# exposed by the backend listing API (lookup_code). All eleven codes below were
# verified against the live detail endpoint on 2026-10-19: the listing API
# accepts the same spellings as submission (tianyan24 / tianyan176-2 /
# tianyan504 without extra hyphens; tianyan-287/294 are natively hyphenated).
# Hyphenated guesses like tianyan-24 return RESOURCE_NOT_FOUND.
LOOKUP_CODES = {
    "tianyan176": "tianyan176",
    "tianyan176-2": "tianyan176-2",
    "tianyan24": "tianyan24",
    "tianyan504": "tianyan504",
    "tianyan-287": "tianyan-287",
    "tianyan-294": "tianyan-294",
    "tianyan_sw": "tianyan_sw",
    "tianyan_s": "tianyan_s",
    "tianyan_tn": "tianyan_tn",
    "tianyan_sa": "tianyan_sa",
    "tianyan_swn": "tianyan_swn",
}


@dataclass(frozen=True, slots=True)
class SubmissionBackend:
    canonical_code: str
    lookup_code: str
    resource_type: BackendType

    @property
    def is_hardware(self) -> bool:
        return self.resource_type is BackendType.QUANTUM_COMPUTER


def resolve_submission_backend(name: str) -> SubmissionBackend:
    supplied = name.strip()
    canonical = BACKEND_ALIASES.get(supplied, supplied)
    if canonical in QUANTUM_BACKENDS:
        resource_type = BackendType.QUANTUM_COMPUTER
    elif canonical in SIMULATOR_BACKENDS:
        resource_type = BackendType.SIMULATOR
    else:
        raise ServiceError(
            "UNSUPPORTED_BACKEND",
            f"不支持向后端 {supplied} 提交实验",
            False,
        )
    return SubmissionBackend(canonical, LOOKUP_CODES[canonical], resource_type)
