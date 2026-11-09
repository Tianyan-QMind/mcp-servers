# 部署与运维

## 身份模式

| 模式 | 入站 Header | 定位 |
|---|---|---|
| `direct_sdk` | `Authorization: Bearer <SDK_OPEN_ID>` | **推荐**，逐请求身份隔离，无需建库 |
| `single_user` | `Authorization: Bearer <API_KEY>` | 历史方案，不推荐 |
| `multi_tenant` | 每租户独立 API Key | 历史方案，不推荐 |

凭证校验覆盖范围：`direct_sdk` 拦截除 `/healthz` 外的全部路径


## 配置项

`direct_sdk` 模式下的完整环境变量：

| 名称 | 默认值 | 说明 |
|---|---|---|
| `TIANYAN_IDENTITY_MODE` | 无（HTTP 必填） | 本部署应为 `direct_sdk` |
| `TIANYAN_DIRECT_SDK_FINGERPRINT_KEY` | 无 | 必填，≥32 字节，不得交给 Host |
| `TIANYAN_DIRECT_SDK_VERIFY_OPENID` | `true` | 首次使用 Open ID 时向上游登录校验 |
| `TIANYAN_DIRECT_SDK_VERIFY_REJECT_TTL_SECONDS` | `60` | 失败计数窗口，范围 `(0,300]` 秒 |
| `TIANYAN_DIRECT_SDK_MAX_VERIFY_ATTEMPTS` | `5` | 窗口内允许的上游登录失败次数，`[1,20]` |
| `TIANYAN_DIRECT_SDK_AUTH_ENFORCEMENT` | `enforce` | `enforce` 失败即拦截；`shadow` 只告警 |
| `TIANYAN_DIRECT_SDK_RATE_LIMIT_ENABLED` | `true` | 按凭据指纹限流，与来源 IP 无关 |
| `TIANYAN_DIRECT_SDK_RATE_LIMIT_WINDOW_SECONDS` | `60` | 窗口长度，范围 `(0,3600]` 秒 |
| `TIANYAN_DIRECT_SDK_RATE_LIMIT_MAX_REQUESTS` | `120` | 每窗口请求数，`[1,100000]` |
| `TIANYAN_ALLOW_HARDWARE_SUBMISSION` | `false` | 严格小写 `true`/`false` |
| `TIANYAN_MCP_ALLOWED_ORIGINS` | 空 | 逗号分隔的精确 Origin，禁止 `*` |
| `TIANYAN_MCP_ALLOWED_HOSTS` | `127.0.0.1,localhost,::1` | 逗号分隔，不含端口 |
| `TIANYAN_HTTP_TIMEOUT_SECONDS` | `10` | 上游超时，范围 `(0,60]` 秒 |
| `TIANYAN_MCP_HOST` | `127.0.0.1` | `direct_sdk` 只接受 loopback |
| `TIANYAN_MCP_PORT` | `8000` | 监听端口 |
| `TIANYAN_LIVE_TEST` | `0` | 为 `1` 时启用真实只读联网测试 |

布尔值一律严格小写，`True`/`1` 会被拒绝。启动时若关闭鉴权校验或限流，日志会输出 WARNING。

## 身份链路

```text
MCP Host
  │ Authorization: Bearer <SDK_OPEN_ID>
  ▼
反向代理（可选，负责 TLS 与访问来源控制）
  ▼
Tianyan MCP Server（默认 127.0.0.1，direct_sdk）
  ├─ 校验 Host、Origin 与 Bearer 格式
  ├─ 计算 SDK Open ID 的不可逆 HMAC 指纹
  ├─ 按指纹隔离并短期缓存上游 Access Token
  └─ 调用固定的天衍云接口
  ▼
https://qc.zdxlz.com
```

- SDK Open ID 与 Access Token **不落库、不写日志**；日志只记录 request id 与截断的不可逆指纹；
- 不把调用方的 `Authorization`、Cookie 或其他 Header 转发给天衍云；
- 调用方无法控制上游 URL、Host、Path 或任意 Header；
- Access Token 仅进程内短期缓存，鉴权失败即失效。


## 安全说明

- 本项目按「受控环境」设计，**不是开箱即用的公网生产安全基线**。
- 凭证只从环境变量读取，仓库中不含任何真实密钥；`.env` 已被 `.gitignore` 与
  `.dockerignore` 排除。
- 日志脱敏：上游返回的错误文本在透传前会把服务器绝对路径、URL、Java 堆栈帧、异常类名，
  以及 `.env`、`id_rsa`、`*.pem`、`*.jks` 等密钥文件名替换为 `[redacted]`，再截断到 300 字。
- 提交类接口可能消耗云端额度；使用真实硬件前请确认账号权限与计费方式。

## 错误码与排障

| 现象 | 检查与处理 |
|---|---|
| 启动报配置错误 | `direct_sdk` 需要 32+ 字节指纹密钥、只绑定 loopback，且不得设置单用户凭证 |
| `/healthz` 正常但 `/mcp` 401 | Host 必须只发送一个格式正确的 `Bearer`，值为当前 SDK Open ID |
| Host/Origin 被拒绝 | 核对精确白名单；不要使用 `*` |
| `SDK_AUTH_FAILED` | 上游固定指向天衍云生产环境，测试环境凭证不兼容 |
| `SDK_AUTH_UNAVAILABLE` | 检查到 SDK 登录接口的网络与超时 |
| `BACKEND_UNAVAILABLE` | 先查资源列表，等待恢复或换后端 |
| `SUBMISSION_OUTCOME_UNKNOWN` | **可能已受理**，先去平台核查，禁止直接重提 |
| `SUBMISSION_REJECTED` | 业务拒绝，消息中附带上游结构化错误原文 |
| `WAVEFORM_SUBMISSION_REJECTED` | 波形任务建单失败，通常是门校验或权限问题 |
| 结果一直 `processing` | 服务端不轮询，请稍后重新调用只读查询 |
| 期望值 `incomplete_submission` | 查看 manifest 中各任务状态，分别核查已有 query id |
| 期望值 `invalid_result` | 核对概率、位宽与 `resultStatus` 位映射，不要绕过校验 |

## 已知限制

- 列表查询对网络错误、超时与 502/503/504 最多重试一次；**提交 POST 永不自动重试**；
- 结果查询不轮询，也不提供标准差、置信区间或扣费金额；
- 不实现 OAuth、响应缓存、API 网关与公网证书管理；
- 期望值不按 Pauli 项分组优化，`tianyan_sa` 不在期望值后端白名单内；
- 上游为未公开的 HTTP 接口，平台升级可能导致解析失败；
- 真机结果的比特位映射建议用单独批准的最小实验持续验证。
