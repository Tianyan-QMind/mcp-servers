from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolRequest, ToolAnnotations
from pydantic import Field, StringConstraints

from tianyan_mcp.circuit_tools import register_circuit_tools
from tianyan_mcp.compile_tools import register_compile_tools
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.expectation_service import ExpectationService
from tianyan_mcp.expectation_tools import register_expectation_tools
from tianyan_mcp.experiment_models import (
    ExperimentBatchSubmissionResult,
    ExperimentResultQueryResult,
    ExperimentSubmissionResult,
)
from tianyan_mcp.experiment_result_service import ExperimentResultService
from tianyan_mcp.experiment_service import ExperimentService
from tianyan_mcp.models import BackendDetail, BackendListResult, BackendType
from tianyan_mcp.quantum_info_tools import register_quantum_info_tools
from tianyan_mcp.service import BackendService
from tianyan_mcp.waveform_models import WaveformDiagramResult, WaveformGenerationResult
from tianyan_mcp.waveform_service import WaveformService

LOGGER = logging.getLogger(__name__)
Identifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
Circuit = Annotated[str, Field(min_length=1, max_length=20_000)]
Circuits = Annotated[list[Circuit], Field(min_length=2, max_length=20)]
ExperimentName = Annotated[str, Field(min_length=1, max_length=100)]
QcCode = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"
    ),
]
ShotCount = Annotated[int, Field(strict=True, ge=1, le=1000)]
AllowHardware = Annotated[bool, Field(strict=True)]
IncludeResultStatus = Annotated[bool, Field(strict=True)]
IsVerify = Annotated[bool, Field(strict=True)]


def _logging_call_tool_handler(
    original_handler: Callable[[CallToolRequest], Awaitable[Any]],
) -> Callable[[CallToolRequest], Awaitable[Any]]:
    async def logging_handler(req: CallToolRequest) -> Any:
        tool_name = req.params.name
        LOGGER.info("event=tool_call_start tool=%s", tool_name)
        try:
            result = await original_handler(req)
        except Exception:
            LOGGER.warning(
                "event=tool_call_exception tool=%s",
                tool_name,
                exc_info=True,
            )
            raise
        payload = result.root if hasattr(result, "root") else result
        if getattr(payload, "isError", False):
            LOGGER.warning("event=tool_call_failed tool=%s", tool_name)
        else:
            LOGGER.info("event=tool_call_done tool=%s", tool_name)
        return result

    return logging_handler


def install_tool_call_logging(mcp: FastMCP) -> None:
    """Log the concrete tool name for every CallToolRequest.

    The low-level server only emits "Processing request of type CallToolRequest",
    which does not reveal which registered tool was invoked. Wrap the low-level
    handler once so start/completion/failure log lines all carry ``tool=<name>``.
    """
    # FastMCP 1.x 未公开底层 Server 访问器，只能经 _mcp_server 取请求分发表；
    # 这是接入统一调用日志的唯一集中入口，定点豁免私有成员访问检查。
    low_level_server = mcp._mcp_server  # pyright: ignore[reportPrivateUsage]
    original_handler = low_level_server.request_handlers[CallToolRequest]
    low_level_server.request_handlers[CallToolRequest] = _logging_call_tool_handler(
        original_handler
    )


def create_mcp_server(
    service: BackendService,
    experiment_service: ExperimentService,
    experiment_result_service: ExperimentResultService,
    waveform_service: WaveformService,
) -> FastMCP:
    mcp = FastMCP(
        name="Tianyan Quantum Resources",
        instructions=(
            "Query Tianyan quantum resources and experiment results, submit single or batch "
            "QCIS experiments, or estimate observables using measurement circuits. "
            "Eighteen circuit, compilation and simulation tools run locally and never contact "
            "the cloud; the remaining tools call the Tianyan cloud. Every tool call over HTTP "
            "requires a Bearer credential, including the local tools that do not use it. "
            "Expectation submission defaults to a dry-run preview. Actual submission requires "
            "host confirmation and may consume cloud resources. Waveform tools create and "
            "query SDK pulse rendering tasks; generation is a remote write that does not "
            "execute quantum hardware, and the returned URL is valid for about ten minutes."
        ),
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    async def list_available_quantum_backends(
        backend_type: BackendType = BackendType.ALL,
    ) -> BackendListResult:
        """查询当前运行中的天衍云量子计算机和量子仿真机。

        backend_type 可为 all、quantum_computer 或 simulator。
        每次调用都会实时读取天衍云公开列表接口，不使用缓存。
        """
        try:
            return await service.list_available(backend_type)
        except ServiceError as exc:
            LOGGER.warning(
                "event=tool_error tool=list_available_quantum_backends code=%s",
                exc.code,
            )
            raise RuntimeError(exc.as_json()) from None
        except Exception:
            LOGGER.error(
                "event=tool_error tool=list_available_quantum_backends code=INTERNAL_ERROR"
            )
            error = ServiceError("INTERNAL_ERROR", "服务内部错误", False)
            raise RuntimeError(error.as_json()) from None

    async def get_quantum_backend(identifier: Identifier) -> BackendDetail:
        """按资源 id 或 code 精确查询天衍云量子资源详情。

        先匹配 id，再忽略 code 大小写进行完全匹配；不进行名称或模糊搜索。
        """
        try:
            return await service.get_backend(identifier)
        except ServiceError as exc:
            LOGGER.warning(
                "event=tool_error tool=get_quantum_backend code=%s",
                exc.code,
            )
            raise RuntimeError(exc.as_json()) from None
        except Exception:
            LOGGER.error("event=tool_error tool=get_quantum_backend code=INTERNAL_ERROR")
            error = ServiceError("INTERNAL_ERROR", "服务内部错误", False)
            raise RuntimeError(error.as_json()) from None

    async def submit_quantum_experiment(
        computer_code: Identifier,
        circuit: Circuit,
        name: ExperimentName,
        shots: ShotCount,
        lab_name: ExperimentName | None = None,
        allow_hardware: AllowHardware = False,
    ) -> ExperimentSubmissionResult:
        """Submit and run one QCIS experiment circuit.

        This creates an experiment and consumes Tianyan cloud resources. Repeated calls
        may create duplicate tasks. The host must show the backend, circuit name, optional
        lab name, and shots to the user and obtain confirmation before submitting.
        """
        try:
            return await experiment_service.submit(
                computer_code=computer_code,
                circuit=circuit,
                name=name,
                lab_name=lab_name,
                shots=shots,
                allow_hardware=allow_hardware,
            )
        except ServiceError as exc:
            LOGGER.warning(
                "event=tool_error tool=submit_quantum_experiment code=%s",
                exc.code,
            )
            raise RuntimeError(exc.as_json()) from None
        except Exception:
            LOGGER.error("event=tool_error tool=submit_quantum_experiment code=INTERNAL_ERROR")
            error = ServiceError("INTERNAL_ERROR", "服务内部错误", False)
            raise RuntimeError(error.as_json()) from None

    async def submit_quantum_experiment_batch(
        computer_code: Identifier,
        circuits: Circuits,
        name: ExperimentName,
        shots: ShotCount,
        lab_name: ExperimentName | None = None,
        allow_hardware: AllowHardware = False,
    ) -> ExperimentBatchSubmissionResult:
        """Submit and run 2 to 20 QCIS experiment circuits in one request.

        This creates multiple remote experiments and may consume Tianyan cloud resources.
        The host must show the backend, circuit count, circuit contents, optional lab name,
        and shots to the user and obtain confirmation before submitting.
        """
        try:
            return await experiment_service.submit_batch(
                computer_code=computer_code,
                circuits=circuits,
                name=name,
                lab_name=lab_name,
                shots=shots,
                allow_hardware=allow_hardware,
            )
        except ServiceError as exc:
            LOGGER.warning(
                "event=tool_error tool=submit_quantum_experiment_batch code=%s",
                exc.code,
            )
            raise RuntimeError(exc.as_json()) from None
        except Exception:
            LOGGER.error(
                "event=tool_error tool=submit_quantum_experiment_batch code=INTERNAL_ERROR"
            )
            error = ServiceError("INTERNAL_ERROR", "服务内部错误", False)
            raise RuntimeError(error.as_json()) from None

    async def get_quantum_experiment_result(
        query_id: Identifier,
        include_result_status: IncludeResultStatus = False,
    ) -> ExperimentResultQueryResult:
        """查询单个天衍云实验的处理状态或已完成结果。

        query_id 必须是实验提交返回的数字字符串。默认不返回可能很大的完整逐次
        测量结果；仅在明确设置 include_result_status=true 时返回该字段。
        """
        try:
            return await experiment_result_service.query(
                query_id,
                include_result_status,
            )
        except ServiceError as exc:
            LOGGER.warning(
                "event=tool_error tool=get_quantum_experiment_result code=%s",
                exc.code,
            )
            raise RuntimeError(exc.as_json()) from None
        except Exception:
            LOGGER.error("event=tool_error tool=get_quantum_experiment_result code=INTERNAL_ERROR")
            error = ServiceError("INTERNAL_ERROR", "服务内部错误", False)
            raise RuntimeError(error.as_json()) from None

    mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )(list_available_quantum_backends)
    mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )(get_quantum_backend)
    mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        )
    )(submit_quantum_experiment)
    mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        )
    )(submit_quantum_experiment_batch)
    mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )(get_quantum_experiment_result)

    async def generate_waveform_diagram(
        qc_code: QcCode,
        circuit: Circuit,
        circuit_name: ExperimentName | None = None,
        is_verify: IsVerify = True,
    ) -> WaveformGenerationResult:
        """根据 QCIS 线路创建天衍云 SDK 脉冲波形图异步渲染任务。

        该操作只在云端生成波形文件，不会在量子硬件上执行实验、不消耗 shots。
        重复调用会创建重复的渲染任务。Host 必须先向用户展示 qc_code、线路内容和
        线路名称并取得确认后再调用；成功后使用返回的 id 轮询
        get_waveform_diagram，提交失败不会自动重试。circuit_name 留空时由天衍云
        自动生成；is_verify=false 仅跳过服务端门校验（波形线路允许没有测量门，
        PZ 等非公开脉冲门也需此开关才能建单），机侧渲染编译仍可能拒绝此类门，
        终态查询会返回 WAVEFORM_GENERATION_FAILED，并非建单成功就一定能出图。
        """
        try:
            return await waveform_service.generate_waveform(
                qc_code=qc_code,
                circuit=circuit,
                circuit_name=circuit_name,
                is_verify=is_verify,
            )
        except ServiceError as exc:
            LOGGER.warning(
                "event=tool_error tool=generate_waveform_diagram code=%s",
                exc.code,
            )
            raise RuntimeError(exc.as_json()) from None
        except Exception:
            LOGGER.error("event=tool_error tool=generate_waveform_diagram code=INTERNAL_ERROR")
            error = ServiceError("INTERNAL_ERROR", "服务内部错误", False)
            raise RuntimeError(error.as_json()) from None

    async def get_waveform_diagram(task_id: Identifier) -> WaveformDiagramResult:
        """按任务 id 查询脉冲波形图下载地址。

        status 为 generating 时 visible_url 为空，表示仍在排队/渲染（机侧队列可能
        排队数分钟到数小时），可稍后再次查询；status 为 completed 时 visible_url
        是天衍云临时下载地址，指向 .pkl 波形数据文件（非图片），约 10 分钟有效，
        本 Tool 只透传地址、不下载文件。本 Tool 不轮询、不自动重试；机侧渲染失败
        （如不支持的脉冲门）、任务不存在或非本人任务会返回稳定错误码。
        """
        try:
            return await waveform_service.get_waveform(task_id)
        except ServiceError as exc:
            LOGGER.warning(
                "event=tool_error tool=get_waveform_diagram code=%s",
                exc.code,
            )
            raise RuntimeError(exc.as_json()) from None
        except Exception:
            LOGGER.error("event=tool_error tool=get_waveform_diagram code=INTERNAL_ERROR")
            error = ServiceError("INTERNAL_ERROR", "服务内部错误", False)
            raise RuntimeError(error.as_json()) from None

    mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True,
        )
    )(generate_waveform_diagram)
    mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )(get_waveform_diagram)
    register_circuit_tools(mcp)
    register_compile_tools(mcp)
    register_quantum_info_tools(mcp)
    register_expectation_tools(
        mcp, ExpectationService(experiment_service, experiment_result_service)
    )
    install_tool_call_logging(mcp)
    return mcp
