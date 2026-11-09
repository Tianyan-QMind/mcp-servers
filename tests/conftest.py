import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def upstream_payload() -> dict[str, Any]:
    path = Path(__file__).parent / "fixtures" / "quantum-backends.json"
    return json.loads(path.read_text(encoding="utf-8"))
