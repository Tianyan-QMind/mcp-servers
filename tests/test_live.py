import os

import pytest

from tianyan_mcp.client import TianyanApiClient
from tianyan_mcp.models import BackendType
from tianyan_mcp.service import BackendService

pytestmark = pytest.mark.skipif(
    os.environ.get("TIANYAN_LIVE_TEST") != "1",
    reason="set TIANYAN_LIVE_TEST=1 to run the public read-only smoke test",
)


@pytest.mark.asyncio
async def test_public_tianyan_read_only_flow() -> None:
    service = BackendService(TianyanApiClient(timeout_seconds=10))
    result = await service.list_available(BackendType.ALL)

    assert result.total >= 1
    detail = await service.get_backend(result.backends[0].id)
    assert detail.id == result.backends[0].id
