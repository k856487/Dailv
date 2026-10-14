# Dailv

降低中文文本 AI 率的本地改写工具。把中文学术段落改写成更接近人工撰写的表述，保留术语、数字和原意。

目前只支持中文。

本仓库是私有项目，不公开权重，也不公开训练数据。

## 模型

现网权重是 **Dailv 1.7**（GGUF F16，约 3.4 GB），放在私有模型库：

https://huggingface.co/k86574/Dailv1.7

文件名：`Dailv1.7-f16.gguf`

没有该模型库读权限的账号下载会得到 404。用自己的 Hugging Face 账号登录，且令牌能读取 `k86574/Dailv1.7` 之后：

```bash
hf auth login
hf download k86574/Dailv1.7 Dailv1.7-f16.gguf --local-dir D:\hf_models\paperpass\gguf
```

默认就读这个路径。放在别处时设置 `DAILV17_MODEL` 为文件的完整路径。

## 环境

| 项目 | 要求 |
| --- | --- |
| 系统 | Windows 10 或更新版本。Linux 可用，启动脚本是 `.cmd` |
| Python | 3.10 及以上 |
| 推理 | [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 `llama-server`，在 `PATH` 里，或用 `DAILV_LLAMA_SERVER` 指向可执行文件 |
| 显卡 | 建议有 Vulkan 或 CUDA。权重会尽量放进显存（`DAILV17_NGL=99`） |
| 磁盘 | 模型约 3.4 GB，另加检测器缓存 |
| 检测器 | 选优时使用公开模型 `yuchuantian/AIGC_detector_zhv3`，首次运行自动下载 |

安装 Python 依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 使用

命令行（仓库里的 `tools\bin` 加入 `PATH` 后）：

```bat
Dailv 待改写的中文段落
```

或：

```bash
python tools/dailv/dailv.py --profile 1.7b "待改写的中文段落"
```

本地页面：

```bash
python web/server.py
```

浏览器打开 http://127.0.0.1:8780 。页面和命令行走同一条改写链路。页头可以调 AI 率降幅（默认 20%）和原文相似度（默认 70%）。

## 常用配置

在启动前设置。不设置就用括号里的默认值。

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `DAILV17_MODEL` | `D:\hf_models\paperpass\gguf\Dailv1.7-f16.gguf` | 权重路径 |
| `DAILV_LLAMA_SERVER` | 从 `PATH` 找 `llama-server` | llama.cpp 服务程序 |
| `DAILV_BON_N` | 命令行 `4`，页面 `8` | 每段生成几条候选再挑选 |
| `DAILV_AGGRESSIVE_DROP_FLOOR` | `0.20` | AI 率降幅低于此值会再改一版 |
| `DAILV_SERVER` | `http://127.0.0.1:8765` | 常驻推理地址。设为 `0` 则每次单独启动 |
| `DAILV_WEB_PORT` | `8780` | 页面端口 |

## 访问范围

仓库和模型都是私有的。陌生人、搜索引擎和未登录访问都进不去。

你自己的账号可以。你邀请的协作者可以。你发出去的、并且权限覆盖这个仓库或这个模型的令牌也可以：对方拿到令牌，就等于拿到你授予该令牌的那部分读权限。GitHub 的经典 `repo` 令牌能读你账号下全部私有仓库。Hugging Face 的普通 Read 令牌能读该账号下全部私有模型。若只想给一个人开一扇门，用限定到单个仓库的细粒度令牌，并设过期时间。
