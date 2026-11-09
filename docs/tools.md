# 工具详解

本项目共注册 **28 个 Tool**，按依赖关系分五组：7 个访问天衍云、3 个期望值工具、18 个计算工具。

每个 Tool 的完整参数约束（取值范围、枚举、默认值）以 MCP 的 JSON Schema 与函数 docstring
为准，这里只说明职责与关键行为。

## 分组总览

| 分组 | 数量 | 
|---|---:|
| 云端资源与实验 | 7 | 
| 期望值 | 3 |
| 电路操作 | 7 | 
| 编译 | 5 | 
| 仿真与量子信息 | 6 | 



## 云端工具（7）

| Tool | 类型 | 主要输入 | 作用 |
|---|---|---|---|
| `list_available_quantum_backends` | 只读 | `backend_type` | 列出当前运行中的量子计算机与仿真机 |
| `get_quantum_backend` | 只读 | `identifier` | 按 `id` 或 `code` 精确查询资源详情 |
| `get_quantum_experiment_result` | 只读 | `query_id`、`include_result_status` | 查询普通实验状态或结果 |
| `submit_quantum_experiment` | 写 | `computer_code`、`circuit`、`name`、`shots` | 提交一条 QCIS 线路 |
| `submit_quantum_experiment_batch` | 写 | 同上，`circuits` 为 2–20 条 | 一次提交一批线路 |
| `generate_waveform_diagram` | 写（远端渲染） | `qc_code`、`circuit`、`circuit_name`、`is_verify` | 创建脉冲波形图渲染任务 |
| `get_waveform_diagram` | 只读 | `task_id` | 查询波形任务状态与下载地址 |

### 资源查询

`backend_type` 可取 `all`、`quantum_computer`、`simulator`。每次实时读取上游，不使用缓存，
只返回 `available=true` 的资源。

`get_quantum_backend` 先按 `id` 精确匹配，再忽略大小写按 `code` 匹配，不做模糊搜索；
详情查询可以返回已知但当前不可用的资源。

### 实验提交

单条 Tool 的必填输入为 `computer_code`、`circuit`（一条 QCIS，最长 20 000 字符，禁止
`U+0000`）、`name`（1–100 字符）、`shots`（1–1000 的整数）；可选 `lab_name` 与
`allow_hardware`。省略 `lab_name` 时上游请求完全不携带该字段。

批量 Tool 把 `circuit` 换成 `circuits`（2–20 条），一次 POST 提交整批，返回的 `query_ids`
与输入顺序一一对应。

支持的后端 code：

| 类型 | code |
|---|---|
| 量子计算机 | `tianyan176`、`tianyan176-2`、`tianyan24`、`tianyan504`、`tianyan-287`、`tianyan-294` |
| 模拟器 | `tianyan_sw`、`tianyan_s`、`tianyan_tn`、`tianyan_sa`、`tianyan_swn` |

硬件另接受 `tianyan-176`、`tianyan-176-2`、`tianyan-24`、`tianyan-504`、`tianyan287`、
`tianyan294` 别名。上游固定 `language=qcis`、`priority=0`、`is_verify=true`、`lab_id=""`，
`tianyan_sa` 额外固定 `quantumState=0`，调用方不能覆盖。

**硬件提交的三重约束**：服务端 `TIANYAN_ALLOW_HARDWARE_SUBMISSION=true`、Tool 显式传入
`allow_hardware=true`、Host 已向用户展示后端/线路/名称/shots 并取得确认。三者缺一不可。
模拟器提交保持开关为 `false`。

### 结果查询

查询已有实验，不会提交、修改或重跑。上游 `status=1` 返回 `processing`，`status=0` 返回
`completed`（含 `probability` 与 `experiment_task_id`）。完整 `resultStatus` 可能很大，
默认裁剪，只有 `include_result_status=true` 时才返回。

单次调用只查询一次，**不轮询、不重试**；处理中请稍后再次调用。

### 脉冲波形图

`generate_waveform_diagram` 只在云端生成波形渲染任务，不在硬件上执行实验、不消耗 shots。
重复调用会创建重复任务。`is_verify=false` 仅跳过服务端门校验（波形线路允许没有测量门，
`PZ` 等非公开脉冲门需要该开关才能建单），机侧编译仍可能拒绝。

`get_waveform_diagram` 返回的 `visible_url` 指向 `.pkl` 波形数据（不是图片），有效期约
10 分钟；本 Tool 只透传地址，不下载文件。

## 期望值工具（3）

| Tool | 类型 | 主要输入 | 作用 |
|---|---|---|---|
| `estimate_quantum_expectation` | 写（默认 `dry_run=true`） | `computer_code`、`circuit`、`observable`、`name`、`shots` | 单条制备线路的观测量期望值 |
| `batch_estimate_quantum_expectation` | 写（默认 `dry_run=true`） | 同上，`circuits` 为 1–8 条 | 多条线路共用同一 observable |
| `get_quantum_expectation_result` | 只读 | `manifest` | 凭完整任务清单查询结果 |

`observable` 是 Pauli 项的线性组合，`X1`、`Y1`、`Z1` 表示物理 Q1，`Z1 Z9` 表示两比特乘积，
恒等项写 `I`：

```json
[
  {"pauli": "Z1 Z9", "coefficient": 0.5},
  {"pauli": "X1", "coefficient": -0.25},
  {"pauli": "I", "coefficient": 1.0}
]
```

系数必须为有限实数，范围 `[-1000000, 1000000]`；重复项会合并，零项删除。

单条预览：

```json
{
  "computer_code": "tianyan_sw",
  "circuit": "H Q1",
  "observable": [{"pauli": "X1", "coefficient": 1.0}],
  "name": "expectation-demo",
  "shots": 100,
  "dry_run": true
}
```

限制：最多 16 个输入 Pauli 项、32 个观测量物理比特、16 个展开测量任务；输入只接受
`I/H/X/Y/Z/X2P/X2M/Y2P/Y2M Qn`、`CZ Qn Qm` 与 `RX/RY/RZ Qn <数值弧度>`，
不接受测量、重置、注释、参数表达式、动态控制或未知门。

`dry_run=true` 是默认值，只在服务端生成并校验测量计划：不登录天衍云、不检查后端在线、
不提交任务、不消耗额度，返回 `measurements`、`job_count`、`total_shots` 与完整 `manifest`。
**预览成功不代表后端在线或硬件已授权。**

展开规则：每个非恒等 Pauli 项独立展开（Z 直接测量，X 前置 `H`，Y 前置 `RX(π/2)`，
最后按物理编号升序添加 `M`）。实际提交按顺序执行，首次失败即停止。manifest 状态：

| 状态 | 含义 |
|---|---|
| `submitted` | 已获得唯一 `query_id` |
| `rejected` | 明确拒绝或明确失败 |
| `unknown` | 无法确认上游是否受理 |
| `not_submitted` | 因前序失败未继续提交 |

结果查询必须传入提交阶段原样返回的完整 manifest。manifest 无凭证、未签名，不代表所有权。
计算链路为：QCIS 制备线路 → 云端测量概率 → 位序与概率校验 →
`Hamiltonian.expectation_probs()`。概率键必须是二进制字符串，概率和容差 `1e-6`，
不会静默归一化。

固定版本 cqlib 的 `Statevector.expectation(Y)` 存在已知符号差异，本功能不调用它。

## 计算工具（18）

### 电路操作（7）

| Tool | 作用 |
|---|---|
| `convert_circuit_format` | OpenQASM 2.0 / OpenQASM 3.0 / QCIS 互转 |
| `analyze_circuit` | 返回深度、门数、分层与操作列表 |
| `circuit_to_matrix` | 酉线路转 `2^n × 2^n` 复数矩阵 |
| `circuit_to_symbolic_matrix` | 转符号矩阵（保留参数表达式） |
| `draw_circuit` | 渲染 ASCII / SVG / PNG（PNG 为 base64） |
| `plot_measurement_result` | 测量计数直方图或概率分布图 |
| `check_circuit_equivalence` | 判断两条酉线路是否仅相差全局相位 |

### 编译（5）

| Tool | 作用 |
|---|---|
| `check_gate_commutation` | 判断同比特上两个单比特门是否对易 |
| `compile_for_device` | 按给定比特数编译到线性拓扑设备 |
| `compile_with_target` | 按显式 `CompileTarget` 编译 |
| `decompose_circuit` | 把门分解为更基础的基门 |
| `optimize_circuit` | 执行 `canonicalize` / 交换消去 / 单比特合并 / native 优化 |

### 仿真与量子信息（6）

| Tool | 作用 |
|---|---|
| `simulate_circuit` | statevector / density_matrix / 含噪 / stabilizer 仿真，可附带可视化 |
| `plot_bloch_vector` | 绘制单个 Bloch 矢量（SVG 或 base64 PNG） |
| `plot_state` | 用 cqlib 的状态可视化函数绘图 |
| `analyze_pauli_string` | Pauli 串的信息、矩阵、乘法、对易性 |
| `analyze_hamiltonian` | 由 Pauli 项构造哈密顿量，求矩阵、期望、方差或 Trotter 线路 |
| `calculate_state_metrics` | 状态距离、熵与纠缠度量 |


