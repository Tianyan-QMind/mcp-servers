# Tianyan MCP Server

把**天衍云**（Tianyan Cloud）量子计算平台封装成 [MCP](https://modelcontextprotocol.io) 工具：
查询量子后端、提交 QCIS 实验、估计观测量期望值，并用 cqlib 完成电路转换、编译、
仿真与可视化。

Python ≥ 3.11 · MCP SDK ≥ 1.27 · 28 个 Tool

## 安装

```powershell
uv sync
```

## 快速开始

```powershell
$env:TIANYAN_IDENTITY_MODE = 'direct_sdk';
$env:TIANYAN_DIRECT_SDK_FINGERPRINT_KEY = '<server-only-random-value-at-least-32-bytes>';
$env:TIANYAN_MCP_HOST = '127.0.0.1';
$env:TIANYAN_MCP_PORT = '8000';
uv run tianyan-mcp-http
```

指纹密钥至少 32 字节，只保存在服务端。

启动后：

```text
Health: http://127.0.0.1:8000/healthz
MCP:    http://127.0.0.1:8000/mcp
```

`/healthz` 不要求鉴权、不访问天衍云，只证明进程存活，不代表凭证有效或后端在线。

## 接入 MCP Host

```json
{
  "mcpServers": {
    "tianyan-cloud": {
      "type": "streamable-http",
      "url": "http://127.0.0.1:8000/mcp",
      "headers": { "Authorization": "Bearer <你的天衍云 SDK Open ID>" }
    }
  }
}
```

SDK Open ID 由天衍云平台签发。

## 工具

| 分组 | 数量 | 说明 |
|---|---:|---|
| 云端资源与实验 | 7 | 查询后端、提交/批量提交实验、查询结果、脉冲波形图 |
| 期望值 | 3 | 观测量期望值估计，默认 `dry_run` 只做本地预览 |
| 电路操作 | 7 | 格式互转、深度分析、矩阵、绘图、等价性判断 |
| 编译 | 5 | 目标设备编译、门分解、优化、门对易 |
| 仿真与量子信息 | 6 | 状态仿真、Bloch 图、Pauli 串与哈密顿量计算 |


工具职责与接口约定见 [docs/tools.md](docs/tools.md)；完整参数以每个 Tool 的 JSON Schema
和 docstring 为准。

## 配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TIANYAN_IDENTITY_MODE` | 无（必填） | 推荐 `direct_sdk` |
| `TIANYAN_DIRECT_SDK_FINGERPRINT_KEY` | 无 | 必填，≥32 字节 |
| `TIANYAN_ALLOW_HARDWARE_SUBMISSION` | `false` | 真机提交总开关，需配合 Tool 参数与用户确认 |
| `TIANYAN_MCP_HOST` / `TIANYAN_MCP_PORT` | `127.0.0.1` / `8000` | `direct_sdk` 只允许 loopback |

完整变量见 [docs/operations.md](docs/operations.md)。


## 安全须知

- 凭证只从环境变量读取，不落库、不写日志。
- 提交类接口会消耗云端额度，且**从不自动重试**。。
- 上游接口可能随平台升级而变化。


## 许可证

[Apache-2.0](LICENSE)。可自由使用、修改与分发，需保留版权与许可声明；同时包含专利授权
与免责条款。
