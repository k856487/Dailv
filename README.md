# Dailv

降低中文文本 AI 率的本地改写页面。把中文学术段落改写成更接近人工撰写的表述，保留术语、数字和原意。

目前只支持中文。

本仓库是私有项目。权重不放在这里。

## 目录

```
app/server.py    页面服务
app/index.html   页面
app/dailv.py     改写实现，只由页面调用
requirements.txt 检测器依赖
```

## 模型

权重是私有模型 **Dailv 1.7**（GGUF F16，约 3.4 GB）：

https://huggingface.co/k86574/Dailv1.7

文件名：`Dailv1.7-f16.gguf`

用能读取 `k86574/Dailv1.7` 的账号登录后下载：

```bash
hf auth login
hf download k86574/Dailv1.7 Dailv1.7-f16.gguf --local-dir D:\hf_models\paperpass\gguf
```

默认就读这个路径。放在别处时，把 `DAILV17_MODEL` 设成文件的完整路径。

## 环境

| 项目 | 要求 |
| --- | --- |
| 系统 | Windows 10 或更新版本。Linux 同样用下面的启动命令 |
| Python | 3.10 及以上 |
| 推理 | [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 `llama-server`，放在 `PATH` 里，或用 `DAILV_LLAMA_SERVER` 指向可执行文件 |
| 显卡 | 建议有 Vulkan 或 CUDA。权重会尽量放进显存 |
| 磁盘 | 模型约 3.4 GB，另加检测器缓存 |
| 检测器 | 选优时使用公开模型 `yuchuantian/AIGC_detector_zhv3`，首次运行自动下载 |

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 启动

```bash
python app/server.py
```

浏览器打开 http://127.0.0.1:8780 。页头可以调 AI 率降幅（默认 20%）和原文相似度（默认 70%）。

## 可选配置

启动前设置。不设置就用默认值。

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `DAILV17_MODEL` | `D:\hf_models\paperpass\gguf\Dailv1.7-f16.gguf` | 权重路径 |
| `DAILV_LLAMA_SERVER` | 从 `PATH` 找 `llama-server` | llama.cpp 服务程序 |
| `DAILV_SERVER` | `http://127.0.0.1:8765` | 常驻推理地址。设为 `0` 则由页面自己拉起 |
| `DAILV_WEB_PORT` | `8780` | 页面端口 |

## 访问范围

仓库和模型都是私有的。没有钥匙的人打不开。

你自己的账号可以。你邀请的协作者可以。你发出去的、并且权限覆盖这个仓库或这个模型的令牌也可以。GitHub 的经典 `repo` 令牌能读你账号下全部私有仓库。Hugging Face 的普通 Read 令牌能读该账号下全部私有模型。只想开放一处时，用限定到单个仓库的细粒度令牌，并设过期时间。
