# 无人值守任务清单（paperpass 7B→3B 蒸馏）
# 编排脚本: unattended_pipeline.ps1
# 状态文件: D:\hf_models\tools\dailv\unattended_state.json
# 结果报告: D:\hf_models\tools\dailv\unattended_eval_report.json

## 清单
1. [自动] TUN 按需开启（sesame-tun.ps1 on）
2. [自动] 等待蒸馏数据 Job 完成（6aa7936121047bf1b0386550 → dataset paperpass-rewrite-distill-3b）
3. [自动] 提交并等待 3B QLoRA 训练 Job（→ qwen25-3b-paperpass-rewrite-lora）
4. [自动] 提交并等待 3B LoRA 合并 Job（→ qwen25-3b-paperpass-rewrite-merged）
5. [自动] 本地用 3B+LoRA 改写样本，并用 D:\1code\cursor\ai率检测（zh_v3）测 AI 率；均值下降 ≥5 个百分点即达标
6. [自动] 取消账号下所有仍在跑的 HF Jobs，停止继续扣费
7. [自动] 写入结果报告后，电脑进入睡眠模式

## 说明
- 全程不弹窗询问；失败也会写 state 并尽量取消 Jobs，然后仍会睡眠（避免空转扣费）
- HF Write token 从环境变量 HF_TOKEN 读取
