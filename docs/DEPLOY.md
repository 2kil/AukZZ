# 部署与运维

## 1. 硬件与系统要求

| 项目 | 最低 | 推荐 |
|---|---|---|
| GPU | NVIDIA，≥8 GB 显存（GGUF 路线峰值 7.6 GB） | 16 GB（如 4070 Ti SUPER / 4080） |
| 驱动 | 支持 CUDA 12.x | 最新 Game Ready / Studio |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 12 GB（模型 7.7 G + 运行时 2 G + venv 5.4 G） | 20 GB |
| 系统 | Windows 10/11 x64、Ubuntu 22.04+ | — |

> **不能**用 PyTorch 全精度路线的最低配置：官方实测开 CPU offload 仍需 16.98 GiB 显存，
> 且要往内存驻留约 8 GB。本项目的默认路线（GGUF）就是为 16 GB 卡设计的。

## 2. Windows 部署

### 2.1 Python 依赖

```powershell
# uv 默认缓存在 C:\Program Files\uv-python\cache，无写权限时必须指定
$env:UV_CACHE_DIR = "D:\GitSum\AukZZ\.cache\uv"
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

`requirements.txt` 只含 `fastapi` / `uvicorn[standard]` / `python-multipart` / `numpy`。
**GGUF 路线不需要 torch**。

可选：走 PyTorch 路线时才需要（且有显卡前提）：

```powershell
uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu126 `
  --index-strategy unsafe-best-match "torch==2.7.1+cu126" "torchaudio==2.7.1+cu126"
uv pip install --python .venv\Scripts\python.exe -e docs\AuK
```

> PyPI 上的 torch Windows 轮子是 **CPU 版**，必须显式指定 PyTorch 官方源，否则
> `torch.cuda.is_available()` 永远为 False。

### 2.2 模型与运行时

| 用途 | 文件 | 大小 | 位置 |
|---|---|---|---|
| 生成器（默认） | `auk-flash-q8_0.gguf` | 1.6 GB | `models/AuK-GGUF/` |
| 生成器（编辑） | `auk-base-q8_0.gguf` | 1.6 GB | 同上 |
| 文本/音频编码器 | `qwen2.5-omni-3b-q8_0.gguf` | 4.0 GB | 同上 |
| VAE | `auk-vae-f32.gguf` | 608 MB | 同上 |
| 配置与分词器 | `config/*.yaml`、`tokenizer/*` | <1 MB | 同上 |

来源：<https://huggingface.co/audio-cpp/AuK-Base-and-Flash-GGUF>

推理运行时（Windows x64 / CUDA 12.4）：

```
audio-v0.9.5-bin-windows-x64-cuda12.4.zip    469 MB  → vendor/audio.cpp-cu124/
audio-v0.9.5-cudart-windows-x64-cuda12.4.zip 607 MB  → 同上（CUDA 运行时）
```

外加三个 DLL（不在发行包里，必须单独提供，否则报 `requires Soxr HQ resampling`）：

| DLL | 来源 |
|---|---|
| `libsoxr.dll` | MSYS2 UCRT：`mingw-w64-ucrt-x86_64-libsoxr-0.1.3-6` |
| `libgomp-1.dll` | MSYS2 UCRT：`mingw-w64-ucrt-x86_64-libgomp-16.2.0-7` |
| `libwinpthread-1.dll` | MSYS2 UCRT：`mingw-w64-ucrt-x86_64-libwinpthread-git-*` |

已下载的副本保存在 `vendor/soxr-ucrt/`，可直接复制。详细排错见
[TROUBLESHOOTING §1](TROUBLESHOOTING.md#1-requires-soxr-hq-resampling)。

> **强烈建议**把 `D:\GitSum\AukZZ\vendor\audio.cpp-cu124` 加入杀毒软件排除目录。
> 实时扫描会导致首次 `LoadLibrary` 失败，或被静默删除（本项目踩过一次）。

### 2.3 启动

```powershell
.venv\Scripts\python.exe -m app.main --port 8080 --webui
```

启动后会自动拉起常驻推理进程，流程为：加载权重（约 6 s）→ 预热 Instruct 路径 →
预热音色路径 → 对外服务。总耗时约 9 s。

## 3. 启动参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` | `127.0.0.1` | 对外服务用 `0.0.0.0`（**无鉴权**，见 §6） |
| `--port` | `8080` | HTTP 端口 |
| `--webui` / `-webui` | 关 | 开启 `/` 调试控制台，JSON 索引移到 `/api` |
| `--tts-backend {server,cli}` | `server` | `cli` 为每请求加载模型（约 7 s），仅调试 |
| `--server-url` | `http://127.0.0.1:8123` | 常驻服务地址 |
| `--no-server-autostart` | 关 | 只连接不拉起 |
| `--engine {auto,gguf,auk,mock}` | `auto` | 强制引擎；`gguf`/`auk` 预检失败即退出 |
| `--gguf-variant {flash,base}` | `flash` | 默认生成器 |
| `--gguf-backend {cuda,vulkan,cpu}` | `cuda` | CPU/Vulkan 可省 CUDA 运行时，但慢很多 |
| `--mem-saver` | 关 | 阶段间释放权重，省显存但热请求变慢 |
| `--device` / `--dtype` / `--cpu-offload` / `--prefer` / `--preload` | — | 仅 `--engine auk` 使用 |
| `--log-level` | `info` | uvicorn 日志级别 |

## 4. 环境变量（覆盖配置，无需改代码）

路径类：`AUK_ROLE_DIR`、`AUK_DATA_DIR`、`AUK_OUTPUT_DIR`、`AUDIO_CPP_DIR`、`AUDIO_CPP_CLI`、
`AUDIO_CPP_SERVER`、`AUK_GGUF_DIR`、`AUK_VOICE_DIR`、`AUK_SERVER_CONFIG`、`AUK_CKPT_DIR`、`AUK_SRC`

参数类：`AUK_HOST`、`AUK_PORT`、`AUK_ENGINE`、`AUK_TTS_BACKEND`、`AUK_SERVER_URL`、
`AUK_SERVER_MODEL`、`AUK_SERVER_AUTOSTART`、`AUK_SERVER_WARMUP`、`AUK_SERVER_READY_TIMEOUT`、
`AUK_GGUF_VARIANT`、`AUK_GGUF_EDIT_VARIANT`、`AUK_GGUF_BACKEND`、`AUK_GGUF_TIMEOUT`、
`AUK_SAMPLE_RATE`、`AUK_REF_TRIM_SECONDS`、`AUK_MAX_GEN_SECONDS`、`AUK_MAX_CHUNK_SECONDS`、
`AUK_CHUNK_GAP_SCALE`

例：临时放宽单句上限到 28 秒（注意参考音频要短，否则撞 30 秒预算）：

```powershell
$env:AUK_MAX_GEN_SECONDS = "28"; .venv\Scripts\python.exe -m app.main --port 8080
```

## 5. `vendor/server.json`（常驻服务配置）

```json
{
  "host": "127.0.0.1", "port": 8123, "backend": "cuda", "device": 0,
  "threads": 4, "lazy_load": false, "max_loaded_models": 2,
  "voice_dir": "D:/GitSum/AukZZ/models/voice-library",
  "models": [{
    "id": "auk-flash", "family": "auk", "task": "tts", "mode": "offline",
    "path": "D:/GitSum/AukZZ/models/AuK-GGUF",
    "session_options": {
      "auk.variant": "flash",
      "auk.model_gguf": "auk-flash-q8_0.gguf",
      "auk.qwen_gguf": "qwen2.5-omni-3b-q8_0.gguf",
      "auk.vae_gguf": "auk-vae-f32.gguf"
    },
    "default_request_options": {"seed": 42}
  }]
}
```

要点：

- `lazy_load: false` 才会启动即加载（否则首请求多等约 5 s）。
- `path` 与 `voice_dir` 是**绝对路径**，换机器/换系统必须改。
- 只有 Flash 一个模型条目 → `/speech` 永远走 Flash；`/edit` 走 CLI 才能用 Base。
- 想加 Base 需再复制一条 `id: auk-base`、`auk.variant: base`、`auk.model_gguf: auk-base-q8_0.gguf`，
  但两个模型同时驻留约 15 GB，16 GB 卡放不下，建议 `max_loaded_models: 1` + 空闲卸载。

## 6. 上线检查清单

```powershell
# 1) 引擎与依赖
curl -s http://127.0.0.1:8080/health | .venv\Scripts\python.exe -c """
import json,sys; d=json.load(sys.stdin)
print('engine  :', d['engine']['active'], '| loaded:', d['engine']['model_loaded'])
print('missing :', d['gguf']['missing'])
print('soxr    :', d['gguf']['soxr']['loadable'])"""

# 2) 热请求延迟（应 < 1 s）
for ($i=0; $i -lt 3; $i++) {
  curl.exe -s -o NUL -w "%{time_total}s`n" -X POST http://127.0.0.1:8080/speech `
    -H "Content-Type: application/json" `
    -d '{\"text\":\"延迟测试，欢迎来到直播间。\",\"role_name\":\"安若柔\",\"gen_seconds\":3.0}'
}

# 3) 端到端
.venv\Scripts\python.exe tests\smoke_test.py
```

验收标准：`engine=gguf-server`、`model_loaded=true`、`gguf.missing=[]`、热请求 < 1 s、冒烟 8 项全 PASS。

**安全**：服务**没有鉴权**。对外必须满足其一：只监听 `127.0.0.1` 由同机反代；或前置 Nginx/Caddy
做 Basic Auth + TLS + 限流。不要把 8080 直接暴露到公网。

## 7. Linux 部署

### 7.1 与 Windows 的关键差异

| 项目 | Windows | Linux |
|---|---|---|
| 运行时包 | `bin-windows-x64-cuda12.4.zip`（469 MB，含 CLI/Server/gguf 工具） | `bin-ubuntu-x64-cuda12.8-colab.tar.gz`（221 MB，**只有 `audiocpp_server`**） |
| CUDA 库 | 发行包 + `cudart` 包（`.dll`），走 `PATH` | `cudart-ubuntu-x64-cuda12.8.tar.gz`（787 MB）或系统 CUDA 12.8，走 `LD_LIBRARY_PATH` |
| soxr | 三件套 DLL | `apt install libsoxr0`（一个包，`libsoxr.so.0`） |
| ggml | `ggml-*.dll` 84 MB+1.1 GB | **静态链接**，无独立 .so |
| `/edit`、`--tts-backend cli` | 可用 | **不可用**（无 `audiocpp_cli`） |

> 官方预编译支持表只列了 Ubuntu 的 CPU / Vulkan / HIP；CUDA 包名为 `cuda12.8-colab`（面向 Colab）。
> 若跑不通，用源码构建：`scripts/build_linux.sh --deployment-build --backend cuda --cuda-arch native`
> ——Linux 上反而比 Windows 顺利，还能一并得到 CLI 补上 `/edit`。

### 7.2 步骤

```bash
# 1) 系统依赖
sudo apt update && sudo apt install -y python3.10-venv libsoxr0 libgomp1 ffmpeg

# 2) 代码与模型（GGUF 与音频跨平台通用，直接拷）
#    app/  role/  tests/  requirements.txt  models/AuK-GGUF/  models/voice-library/

# 3) 运行时
mkdir -p vendor/audio.cpp-linux && cd vendor/audio.cpp-linux
tar xzf ~/audio-v0.9.5-bin-ubuntu-x64-cuda12.8-colab.tar.gz   # audiocpp_server 在根部
mkdir -p cudart && tar xzf ~/audio-v0.9.5-cudart-ubuntu-x64-cuda12.8.tar.gz -C cudart
find cudart -maxdepth 2 -name 'lib*.so*' -exec cp -a {} . \;   # 摊平到当前目录
chmod +x audiocpp_server && rm -rf cudart

# 4) Python 环境
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 5) 改 vendor/server.json 里的 path 与 voice_dir 为 Linux 绝对路径
#    /opt/auk/models/AuK-GGUF 与 /opt/auk/models/voice-library

# 6) 启动
AUDIO_CPP_DIR=/opt/auk/vendor/audio.cpp-linux \
  .venv/bin/python -m app.main --host 127.0.0.1 --port 8080
```

代码已做平台适配（可执行文件名、`LD_LIBRARY_PATH`、`libcufft.so.11` 探测、`ctypes.CDLL`），
**不需要改代码**。唯一必须手改的是 `vendor/server.json` 的两个路径。

### 7.3 systemd

```ini
# /etc/systemd/system/auk.service
[Unit]
Description=AuK Speech Service
After=network-online.target

[Service]
Type=simple
User=auk
WorkingDirectory=/opt/auk
Environment="AUDIO_CPP_DIR=/opt/auk/vendor/audio.cpp-linux"
Environment="AUK_VOICE_DIR=/opt/auk/models/voice-library"
ExecStart=/opt/auk/.venv/bin/python -m app.main --host 127.0.0.1 --port 8080
Restart=always
RestartSec=10
# 常驻子进程随主进程退出（KillMode=control-group 是默认值，无需改）

[Install]
WantedBy=multi-user.target
```

`sudo systemctl daemon-reload && sudo systemctl enable --now auk && journalctl -u auk -f`

## 8. 运维要点

| 项目 | 说明 |
|---|---|
| 日志 | 服务日志走 stdout（uvicorn）；常驻进程日志 `data/outputs/_tmp/audiocpp_server.log` |
| 显存 | 常驻约 6.2 GB，峰值 7.6 GB；进程被杀后下次请求自动重启（约 10 s） |
| 并发 | 单模型单槽，请求串行排队；吞吐需求靠多实例或 `--parallel-jobs`（显存翻倍） |
| 关闭 | `Ctrl+C` 会回收本服务拉起的常驻进程；外部已存在的实例不动 |
| 数据备份 | 只需备份 `data/roles/`（index.json + audio）与 `role/`；模型可重新下载 |
| 磁盘 | `.cache/uv` 可随时删（6.6 GB）；`data/outputs` 可定期清理 |
| 音色库 | `models/voice-library/*.wav` 由 `/roles` 自动同步，删角色会自动清理 |
