from __future__ import annotations

import logging
from dataclasses import dataclass

import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp

from tianyan_mcp.client import TianyanApiClient
from tianyan_mcp.config import Settings
from tianyan_mcp.credential_crypto import ApiKeyCodec, CredentialCipher
from tianyan_mcp.credential_routes import credential_routes
from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager
from tianyan_mcp.direct_sdk_experiment_client import DirectSdkExperimentClient
from tianyan_mcp.direct_sdk_security import DirectSdkSecurityMiddleware
from tianyan_mcp.direct_sdk_waveform_client import DirectSdkWaveformClient
from tianyan_mcp.experiment_client import TianyanExperimentClient
from tianyan_mcp.experiment_result_service import ExperimentResultService
from tianyan_mcp.experiment_service import ExperimentService
from tianyan_mcp.identity import IdentityMode
from tianyan_mcp.sdk_auth import TianyanSdkAuthProvider, TianyanSdkLoginClient
from tianyan_mcp.security import SecurityMiddleware
from tianyan_mcp.server import create_mcp_server
from tianyan_mcp.service import BackendService
from tianyan_mcp.tenant_experiment_client import TenantExperimentClient
from tianyan_mcp.tenant_sdk_auth import TenantSdkAuthManager
from tianyan_mcp.tenant_security import TenantSecurityMiddleware
from tianyan_mcp.tenant_store import TenantStore
from tianyan_mcp.tenant_waveform_client import TenantWaveformClient
from tianyan_mcp.waveform_client import TianyanWaveformClient
from tianyan_mcp.waveform_service import WaveformService

_LOGGER = logging.getLogger(__name__)


async def healthz(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


@dataclass(frozen=True, slots=True)
class TenantDependencies:
    store: TenantStore
    cipher: CredentialCipher
    auth_manager: TenantSdkAuthManager
    backend_service: BackendService
    experiment_service: ExperimentService
    experiment_result_service: ExperimentResultService
    waveform_service: WaveformService


@dataclass(frozen=True, slots=True)
class DirectSdkDependencies:
    auth_manager: DirectSdkAuthManager
    backend_service: BackendService
    experiment_service: ExperimentService
    experiment_result_service: ExperimentResultService
    waveform_service: WaveformService


def create_app(
    settings: Settings | None = None,
    *,
    tenant_dependencies: TenantDependencies | None = None,
    direct_sdk_dependencies: DirectSdkDependencies | None = None,
) -> ASGIApp:
    resolved = settings or Settings.from_env(default_identity_mode=None)
    if resolved.identity_mode is IdentityMode.SINGLE_USER:
        return create_single_user_app(resolved)
    if resolved.identity_mode is IdentityMode.DIRECT_SDK:
        direct = direct_sdk_dependencies or build_direct_sdk_dependencies(resolved)
        inner = create_http_inner_app(
            resolved,
            direct.backend_service,
            direct.experiment_service,
            direct.experiment_result_service,
            direct.waveform_service,
            [],
        )
        return DirectSdkSecurityMiddleware(
            inner,
            resolved,
            direct.auth_manager,
            verify_open_id=resolved.direct_sdk_verify_openid,
            enforcement=resolved.direct_sdk_auth_enforcement,
        )
    dependencies = tenant_dependencies or build_tenant_dependencies(resolved)
    inner = create_http_inner_app(
        resolved,
        dependencies.backend_service,
        dependencies.experiment_service,
        dependencies.experiment_result_service,
        dependencies.waveform_service,
        credential_routes(
            dependencies.store,
            dependencies.cipher,
            dependencies.auth_manager.invalidate,
        ),
    )
    return TenantSecurityMiddleware(inner, resolved, dependencies.store)


def build_direct_sdk_dependencies(settings: Settings) -> DirectSdkDependencies:
    if not settings.direct_sdk_verify_openid:
        _LOGGER.warning(
            "event=direct_sdk_openid_verification_disabled "
            "TIANYAN_DIRECT_SDK_VERIFY_OPENID is false; forged SDK credentials will be accepted"
        )
    if settings.direct_sdk_auth_enforcement == "shadow":
        _LOGGER.warning(
            "event=direct_sdk_openid_verification_shadow "
            "Open ID verification failures are logged but not enforced"
        )
    if not settings.direct_sdk_rate_limit_enabled:
        _LOGGER.warning(
            "event=direct_sdk_rate_limit_disabled "
            "TIANYAN_DIRECT_SDK_RATE_LIMIT_ENABLED is false; per-client rate limiting is off"
        )
    manager = DirectSdkAuthManager(
        TianyanSdkLoginClient(settings.http_timeout_seconds),
        reject_ttl_seconds=settings.direct_sdk_verify_reject_ttl_seconds,
        max_verify_attempts=settings.direct_sdk_max_verify_attempts,
    )
    backend_service = BackendService(TianyanApiClient(settings.http_timeout_seconds))
    experiment_client = DirectSdkExperimentClient(manager, settings.http_timeout_seconds)
    experiment_service = ExperimentService(
        backend_service,
        experiment_client,
        allow_hardware_submission=settings.allow_hardware_submission,
    )
    experiment_result_service = ExperimentResultService(experiment_client)
    waveform_client = DirectSdkWaveformClient(manager, settings.http_timeout_seconds)
    waveform_service = WaveformService(waveform_client, waveform_client)
    return DirectSdkDependencies(
        manager,
        backend_service,
        experiment_service,
        experiment_result_service,
        waveform_service,
    )


def create_single_user_app(resolved: Settings) -> ASGIApp:
    backend_client = TianyanApiClient(resolved.http_timeout_seconds)
    backend_service = BackendService(backend_client)
    auth_provider = TianyanSdkAuthProvider(
        resolved.sdk_open_id,
        resolved.http_timeout_seconds,
    )
    experiment_client = TianyanExperimentClient(
        auth_provider,
        resolved.http_timeout_seconds,
    )
    experiment_service = ExperimentService(
        backend_service,
        experiment_client,
        allow_hardware_submission=resolved.allow_hardware_submission,
    )
    experiment_result_service = ExperimentResultService(experiment_client)
    waveform_client = TianyanWaveformClient(
        auth_provider,
        resolved.http_timeout_seconds,
    )
    waveform_service = WaveformService(waveform_client, waveform_client)
    return SecurityMiddleware(
        create_http_inner_app(
            resolved,
            backend_service,
            experiment_service,
            experiment_result_service,
            waveform_service,
            [],
        ),
        resolved,
    )


def build_tenant_dependencies(settings: Settings) -> TenantDependencies:
    if (
        settings.database_path is None
        or settings.api_key_pepper is None
        or settings.credential_key_version is None
    ):
        raise RuntimeError("multi-tenant settings are incomplete")
    store = TenantStore(
        settings.database_path,
        ApiKeyCodec(settings.api_key_pepper.get_secret_value()),
    )
    store.initialize()
    cipher = CredentialCipher(
        active_version=settings.credential_key_version,
        keys={version: key.get_secret_value() for version, key in settings.credential_keys.items()},
    )
    manager = TenantSdkAuthManager(
        store,
        cipher,
        TianyanSdkLoginClient(settings.http_timeout_seconds),
    )
    backend_service = BackendService(TianyanApiClient(settings.http_timeout_seconds))
    experiment_client = TenantExperimentClient(manager, settings.http_timeout_seconds)
    experiment_service = ExperimentService(
        backend_service,
        experiment_client,
        allow_hardware_submission=False,
    )
    experiment_result_service = ExperimentResultService(experiment_client)
    waveform_client = TenantWaveformClient(manager, settings.http_timeout_seconds)
    waveform_service = WaveformService(waveform_client, waveform_client)
    return TenantDependencies(
        store,
        cipher,
        manager,
        backend_service,
        experiment_service,
        experiment_result_service,
        waveform_service,
    )


def create_http_inner_app(
    settings: Settings,
    backend_service: BackendService,
    experiment_service: ExperimentService,
    experiment_result_service: ExperimentResultService,
    waveform_service: WaveformService,
    account_routes: list[Route],
) -> ASGIApp:
    mcp = create_mcp_server(
        backend_service,
        experiment_service,
        experiment_result_service,
        waveform_service,
    )
    mcp_app = mcp.streamable_http_app()
    mcp_app.router.routes.insert(0, Route("/healthz", healthz, methods=["GET"]))
    mcp_app.router.routes.extend(account_routes)
    return mcp_app


def main() -> None:
    settings = Settings.from_env(default_identity_mode=None)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level="info",
    )
