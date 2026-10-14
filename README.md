# Dailv

降低中文文本 AI 率的本地改写工具。将中文学术段落改写成更接近人工撰写的表述，保留术语、数字与原意。

目前只支持中文。

## 模型

改写使用 Dailv 1.7 权重（GGUF F16，约 3.4 GB）：

https://huggingface.co/k86574/Dailv1.7

文件名为 `Dailv1.7-f16.gguf`。下载：

```bash
hf auth login
hf download k86574/Dailv1.7 Dailv1.7-f16.gguf --local-dir D:\hf_models\paperpass\gguf
```

程序默认读取 `D:\hf_models\paperpass\gguf\Dailv1.7-f16.gguf`。文件放在其他位置时，将 `DAILV17_MODEL` 设为完整路径。

## 环境

| 项目 | 要求 |
| --- | --- |
| 系统 | Windows 10 及以上。Linux 使用同一启动命令 |
| Python | 3.10 及以上 |
| 推理 | [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 `llama-server`，位于 `PATH`，或由 `DAILV_LLAMA_SERVER` 指定 |
| 显卡 | 建议使用 Vulkan 或 CUDA，权重尽量载入显存 |
| 磁盘 | 模型约 3.4 GB，另需检测器缓存 |
| 检测器 | 选优使用 `yuchuantian/AIGC_detector_zhv3`，首次运行时自动下载 |

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 目录

```
app/server.py      页面服务
app/index.html     页面
app/dailv.py       改写实现
requirements.txt   检测器依赖
```

## 启动

```bash
python app/server.py
```

在浏览器打开 http://127.0.0.1:8780 。页头可调节 AI 率降幅（默认 20%）和原文相似度（默认 70%）。

## 配置

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `DAILV17_MODEL` | `D:\hf_models\paperpass\gguf\Dailv1.7-f16.gguf` | 权重路径 |
| `DAILV_LLAMA_SERVER` | 在 `PATH` 中查找 `llama-server` | 推理程序 |
| `DAILV_SERVER` | `http://127.0.0.1:8765` | 推理服务地址。设为 `0` 时由页面拉起 |
| `DAILV_WEB_PORT` | `8780` | 页面端口 |
