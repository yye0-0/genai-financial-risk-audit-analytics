# 工作台快速上手与三分钟演示

## 直接体验

在仓库根目录执行（Python 3.10–3.12，CI 使用 3.11）：

```bash
python -m pip install -r requirements-lock.txt
python -m streamlit run app.py
```

不需要 API key、PaySim 下载或重新训练即可体验 Frozen-case walkthrough。它使用仓库中的 60 个冻结案例；顶部为保存的全量实验结果，案例队列为分层抽样，二者不能混为同一数据分布。默认 Mock 不是大模型，只验证产品流程。

## 完成一次任务

1. 在 Case investigation 查看队列，选择一个案例。
2. 检查交易事实、规则证据与 Applicable policy evidence。
3. 点击 Generate analyst brief，默认使用 Offline contract mock。
4. 核对 Evidence / Uncertainties。即使通过 schema 与控制校验，也要检查自然语言是否受证据支持。
5. 在 Reviewed summary 修改草稿，填写 Analyst notes、Reviewer name 和 Draft feedback。
6. 选择 Case status 并点击 Save disposition。升级与结案必须填写理由。
7. 查看审计事件，下载 case report；报告包含原稿、人工修改与反馈的历史记录。

刷新页面后用同一案例查看已保存修改稿。Reload saved review 会丢弃当前未保存编辑，并载入最新记录；遇到并发修改错误时先备份自己的文字再重载。

Evaluation mode 才显示离线标签与 False negatives 队列。这些信息用于离线评估，不用于日常审核建议。

## 真实模型

现有适配器支持 OpenAI-compatible Chat Completions 端点。先安装可选依赖，然后在启动进程的环境中设置 OPENAI_API_KEY、LLM_MODEL，按需设置 OPENAI_BASE_URL。`.env.example` 是配置示例；当前程序不会自动读取 `.env`，需要 shell、IDE 或进程管理器加载。不要将真实密钥写入仓库。

```bash
python -m pip install -e ".[llm]"
python -m src.evaluation --provider openai --limit 5 --prompt-version grounded_structured --output-dir artifacts/evaluations/live-smoke
```

先检查 manifest、输出与失败情况，再运行完整比较：

```bash
python -m src.evaluation --provider openai --output-dir artifacts/evaluations/live-full
```

默认 60 案例 × 4 提示词 = 240 次请求。可用 `--prompt-version` 和 `--limit` 明确缩小范围。`--input-price-per-million` 与 `--output-price-per-million` 接收你实际端点每百万 token 的美元单价；本项目不猜测当前价格。用量或价格缺失时成本未知，不当作 0。

调用有 60 秒超时及一次 SDK 自动重试；记录的是最终返回用量，供应商侧失败/重试可能另有计费，估算不替代账单。费用仅在用户配置价格并返回用量时可计算。

界面中选择 OpenAI-compatible API 才会发起真实请求。未配置密钥时明确失败，仍可使用 Mock。新 CLI 评测默认保存至 `artifacts/evaluations/<provider>`；治理页继续显示仓库发布的基准，查看新结果请打开对应运行目录。

## 三分钟演示顺序

- 0:00–0:30：说明用户任务、审核容量和模拟数据边界。
- 0:30–1:15：选择案例，展示交易、规则与政策证据。
- 1:15–2:00：生成草稿，解释 Mock/真实模型区别与信息缺口。
- 2:00–2:40：人工修改、填写理由和反馈并保存。
- 2:40–3:00：展示历史记录、报告及 PRD 中的下一步验证计划。

## 常见问题

- **为什么模型敏感性不可用？** 仓库不包含模型包；冻结演示仍可运行。完整敏感性分析需要按 README 重建实验产物。
- **为什么摘要被拒绝？** schema 或确定性控制不一致；先看原始输出和错误，再决定是否重新生成。拒绝的草稿不自动填入修改框。
- **为什么“结案”被拒绝？** 需要人工理由；分数和 AI 草稿不能代替理由。
- **数据保存在哪？** 演示默认 `artifacts/workspace_demo.sqlite3`；完整队列默认 `artifacts/case_audit.sqlite3`，均不提交。测试或隔离体验可设置 WORKSPACE_AUDIT_DB。
- **能证明提高审核效率了吗？** 还不能；需要按用户试用方案收集真实任务记录。
