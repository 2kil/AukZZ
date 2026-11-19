# AuK 语音服务（直播场景）

基于腾讯 [AuK](https://github.com/Tencent-Hunyuan/AuK) 的语音合成服务：参考音频零样本克隆 +
指令式音色生成 + 17 类音频编辑，通过 **GGUF 量化 + 常驻推理** 把单句延迟压到 **0.3 秒级**，
可在 16 GB 消费级显卡上运行。

```
你的业务  ──HTTP──▶  FastAPI :8080  ──HTTP──▶  audiocpp_server :8123
                     角色/切块/规范化/日志          权重常驻显存，纯推理
```

| 指标 | 数值 |
|---|---|
| 单句延迟（3–4 秒音频，热请求） | 0.3 – 0.9 s |
| 长文本（90 字 → 18.5 秒音频） | 1.49 s |
| 模型体积 | 6.5 GB（Flash 路线）/ 8.1 GB（含 Base） |
| 常驻显存 | 约 6.2 GB（峰值 7.6 GB） |
| 启动耗时 | 约 9 s（加载 + 双路预热） |
| 输出格式 | 24 kHz 单声道 PCM16 WAV |
| 内置角色 | 15 个 |

> **已部署过？** 只需一条命令：
> `.venv\Scripts\python.exe -m app.main --port 8080 --webui` → <http://127.0.0.1:8080/>

---

# 部署

首次部署三步：**下载模型与运行时 → 准备 Python 环境 → 启动**。全部文件放在项目目录内，
不写系统目录。

## 1. 硬件要求

| 项目 | 最低 | 推荐 |
|---|---|---|
| GPU | NVIDIA ≥8 GB 显存（GGUF 峰值 7.6 GB） | 16 GB（4070 Ti SUPER / 4080） |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 12 GB（模型 7.7 + 运行时 1.1 + venv 5.4） | 20 GB |
| 系统 | Windows 10/11 x64 · Ubuntu 22.04+ | — |

> PyTorch 全精度路线（`--engine auk`）需要 ≥24 GB 显存 + ≥32 GB 内存 + 18.8 GB 权重，
> 官方实测开 CPU offload 仍需 16.98 GiB 显存。16 GB 卡请使用默认的 GGUF 路线。

## 2. 下载模型与运行时

两类文件，均不入版本库，合计约 9 GB。**GGUF 权重跨平台通用**，Windows 与 Linux 用的是同一份；
只有推理运行时按平台区分。

### 2.1 模型权重（约 7.7 GB）

来源：<https://huggingface.co/audio-cpp/AuK-Base-and-Flash-GGUF>

| 文件 | 大小 | 必须 | 说明 |
|---|---|---|---|
| `auk-flash-q8_0.gguf` | 1.6 GB | ✅ | 生成器，4 步推理（默认，0.3 s/句） |
| `auk-base-q8_0.gguf` | 1.6 GB | 建议 | 生成器，32 步；`/edit` 用它，编辑保真度明显更好 |
| `qwen2.5-omni-3b-q8_0.gguf` | 4.0 GB | ✅ | 文本 + 音频编码器（Qwen2.5-Omni-3B） |
| `auk-vae-f32.gguf` | 608 MB | ✅ | VAE 解码器 |
| `config/auk-flash.yaml`、`config/auk-base.yaml` | <1 MB | ✅ | 模型配置 |
| `tokenizer/tokenizer.json`、`tokenizer/tokenizer_config.json` | <1 MB | ✅ | 分词器 |

```powershell
# Windows（curl.exe 自带；-C - 支持断点续传）
$B = "https://huggingface.co/audio-cpp/AuK-Base-and-Flash-GGUF/resolve/main"
$FILES = @(
  "auk-flash-q8_0.gguf", "auk-base-q8_0.gguf",
  "qwen2.5-omni-3b-q8_0.gguf", "auk-vae-f32.gguf",
  "config/auk-flash.yaml", "config/auk-base.yaml",
  "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"
)
New-Item -ItemType Directory -Force models\AuK-GGUF | Out-Null
foreach ($f in $FILES) {
  New-Item -ItemType Directory -Force "models/AuK-GGUF/$(Split-Path $f -Parent)" | Out-Null
  curl.exe -L --retry 3 -C - -o "models/AuK-GGUF/$f" "$B/$f"
}
```

```bash
# Linux / macOS
B=https://huggingface.co/audio-cpp/AuK-Base-and-Flash-GGUF/resolve/main
mkdir -p models/AuK-GGUF/config models/AuK-GGUF/tokenizer
for f in auk-flash-q8_0.gguf auk-base-q8_0.gguf qwen2.5-omni-3b-q8_0.gguf \
         auk-vae-f32.gguf config/auk-flash.yaml config/auk-base.yaml \
         tokenizer/tokenizer.json tokenizer/tokenizer_config.json; do
  curl -L --retry 3 -C - -o "models/AuK-GGUF/$f" "$B/$f"
done
```

**国内网络下载不动时**，把域名换成镜像（文件完全一致，仅 CDN 不同）：

```
https://hf-mirror.com/audio-cpp/AuK-Base-and-Flash-GGUF/resolve/main
```

**校验完整性**（字节数必须完全一致，否则加载会失败）：

```powershell
.venv\Scripts\python.exe -c "
import os
EXPECT = {'auk-flash-q8_0.gguf':1639358080, 'auk-base-q8_0.gguf':1639358080,
          'qwen2.5-omni-3b-q8_0.gguf':4253964480, 'auk-vae-f32.gguf':637364224}
for name, size in EXPECT.items():
    p = os.path.join('models/AuK-GGUF', name)
    got = os.path.getsize(p) if os.path.isfile(p) else -1
    print(('OK  ' if got == size else 'BAD '), name, got, '/', size)"
```

### 2.2 推理运行时（Windows 约 1.1 GB / Linux 约 1.0 GB）

来自 audio.cpp v0.9.5 预编译包：<https://github.com/0xShug0/audio.cpp/releases/tag/v0.9.5>

**Windows（本仓库当前使用）**

| 文件 | 大小 | 解压到 |
|---|---|---|
| `audio-v0.9.5-bin-windows-x64-cuda12.4.zip` | 469 MB | `vendor/audio.cpp-cu124/` |
| `audio-v0.9.5-cudart-windows-x64-cuda12.4.zip` | 607 MB | 同上（CUDA 运行时 DLL） |

```powershell
New-Item -ItemType Directory -Force vendor\audio.cpp-cu124 | Out-Null
curl.exe -L -o vendor\bin.zip "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.5/audio-v0.9.5-bin-windows-x64-cuda12.4.zip"
curl.exe -L -o vendor\cudart.zip "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.5/audio-v0.9.5-cudart-windows-x64-cuda12.4.zip"
Expand-Archive vendor\bin.zip    -DestinationPath vendor\audio.cpp-cu124 -Force
Expand-Archive vendor\cudart.zip -DestinationPath vendor\audio.cpp-cu124 -Force
Remove-Item vendor\bin.zip, vendor\cudart.zip

# 补齐 release 未附带、但 AuK 必需的重采样库（仓库已含副本，见下）
Copy-Item vendor\soxr-ucrt\*.dll vendor\audio.cpp-cu124\ -Force
```

> `vendor/soxr-ucrt/` 内是 `libsoxr.dll` + `libgomp-1.dll` + `libwinpthread-1.dll`，
> 缺任一都会在带参考音频时报 `requires Soxr HQ resampling`。这三个文件已入库，无需另行获取；
> 上游来源是 MSYS2 UCRT 的 `libsoxr`、`libgomp`、`libwinpthread` 包。

**Linux**

| 文件 | 大小 | 说明 |
|---|---|---|
| `audio-v0.9.5-bin-ubuntu-x64-cuda12.8-colab.tar.gz` | 221 MB | **只含 `audiocpp_server`**（无 CLI） |
| `audio-v0.9.5-cudart-ubuntu-x64-cuda12.8.tar.gz` | 787 MB | CUDA 12.8 运行时，也可用系统 CUDA |
| `libsoxr0`（apt） | ~100 KB | 重采样库，`sudo apt install libsoxr0` |

```bash
mkdir -p vendor/audio.cpp-linux && cd vendor/audio.cpp-linux
curl -L -o bin.tar.gz "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.5/audio-v0.9.5-bin-ubuntu-x64-cuda12.8-colab.tar.gz"
curl -L -o cudart.tar.gz "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.5/audio-v0.9.5-cudart-ubuntu-x64-cuda12.8.tar.gz"
tar xzf bin.tar.gz                      # audiocpp_server 在归档根部
mkdir -p cudart && tar xzf cudart.tar.gz -C cudart
find cudart -maxdepth 2 -name 'lib*.so*' -exec cp -a {} . \;   # 摊平（有无顶层目录都适用）
chmod +x audiocpp_server && rm -rf bin.tar.gz cudart.tar.gz cudart
sudo apt install -y libsoxr0
```

### 2.3 目录检查

```
models/AuK-GGUF/          约 7.7 GB   ← 权重 + config/ + tokenizer/
vendor/audio.cpp-cu124/   约 1.1 GB   ← audiocpp_server.exe / audiocpp_cli.exe / *.dll
vendor/server.json                     ← 常驻服务配置（绝对路径，换机器需改）
```

## 3. 准备 Python 环境

```powershell
# 需要 Python 3.10+；项目自带 .venv
.venv\Scripts\python.exe --version

# uv 默认缓存在 C:\Program Files\uv-python\cache（无写权限会失败），指定到项目内
$env:UV_CACHE_DIR = "D:\GitSum\AukZZ\.cache\uv"
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

```bash
# Linux
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

依赖只有 4 个：`fastapi`、`uvicorn[standard]`、`python-multipart`、`numpy`。
**GGUF 路线不需要 torch**；仅 `--engine auk` 才需要 PyTorch + AuK 官方栈（见
[DEPLOY §2.1](docs/DEPLOY.md#21-python-依赖)）。

## 4. 启动

```powershell
# 生产（常驻推理，单句 0.3 s）
.venv\Scripts\python.exe -m app.main --port 8080

# 调试（带 WebUI 控制台）
.venv\Scripts\python.exe -m app.main --port 8080 --webui --log-level info

# 对外提供服务（无鉴权，务必配反向代理，见第 7 节）
.venv\Scripts\python.exe -m app.main --host 0.0.0.0 --port 8080
```

启动会自动拉起常驻推理进程：加载权重（约 6 s）→ 预热 Instruct 与音色两条路径 → 对外服务，
总约 9 s。日志应出现两条“预热…完成”。

浏览器打开 <http://127.0.0.1:8080/> 即为调试控制台（角色试听、合成、编辑、请求日志）。

## 5. 上线验收

```powershell
# 引擎与依赖（三项必须为：gguf-server / True / []）
curl -s http://127.0.0.1:8080/health | .venv\Scripts\python.exe -c "
import json,sys; d=json.load(sys.stdin)
print(d['engine']['active'], d['engine']['model_loaded'], d['gguf']['missing'])"

# 端到端（建角色 → 合成 → 编辑 → 删除，8 项）
.venv\Scripts\python.exe tests\smoke_test.py

# 离线单元测试（秒级，不依赖服务）
.venv\Scripts\python.exe tests\test_payload.py
```

单次调用验证：

```powershell
curl -o out.wav -X POST http://127.0.0.1:8080/speech `
  -H "Content-Type: application/json" `
  -d '{"text":"欢迎来到直播间，今天有三款新品上架。","role_name":"安若柔"}'
```

若报 `requires Soxr HQ resampling`，查
[TROUBLESHOOTING §1](docs/TROUBLESHOOTING.md#1-requires-soxr-hq-resampling)。

## 6. Linux 部署

代码已做平台适配（可执行文件名、`LD_LIBRARY_PATH`、`libcufft.so.11` 探测、`ctypes.CDLL`），
**不需要改代码**，只需三步：

1. 用 Linux 运行时包（见 2.2），`export AUDIO_CPP_DIR=/opt/auk/vendor/audio.cpp-linux`
2. 改 `vendor/server.json` 里的 `path` 与 `voice_dir` 为 Linux 绝对路径（唯一必须手改的配置）
3. `apt install libsoxr0`，然后 `python -m app.main --host 127.0.0.1 --port 8080`

注意：Linux CUDA 预编译包**只有 `audiocpp_server`，没有 `audiocpp_cli`**，因此 `/edit` 与
`--tts-backend cli` 不可用；需要时用源码构建（`scripts/build_linux.sh --backend cuda`）或改走
`/v1/tasks/run`。完整步骤、systemd 单元见 [DEPLOY §7](docs/DEPLOY.md#7-linux-部署)。

## 7. 安全与运维

- **服务没有鉴权**：对外必须 `--host 127.0.0.1` + 本机反向代理，或在前置 Nginx/Caddy 上加
  Basic Auth、TLS 与限流。切勿把 8080 直接暴露公网。
- **显存常驻**约 6.2 GB；进程被杀后下次请求会自动重启（约 10 s），无需人工干预。
- **并发**：单模型单槽，请求串行排队；高吞吐需多实例或 `--parallel-jobs`（显存翻倍）。
- **备份**只需 `data/roles/`（自定义角色索引与音频）与 `role/`；模型可重新下载。
- **日志**：服务走 stdout；常驻进程日志在 `data/outputs/_tmp/audiocpp_server.log`。
- **杀软排除**：把 `vendor/audio.cpp-cu124` 加入白名单，实时扫描会导致 DLL 首次加载失败甚至被删。

---

# 参考手册

## 目录结构

```
app/                    ★ 服务代码（业务逻辑全部在这里）
  main.py       467 行   FastAPI 路由、请求模型、响应头、CLI 参数
  engine.py    1159 行   四个引擎 + 预检（server / cli / python / mock）
  audio.py      491 行   WAV 解析、时长估算、文本规范化、切块与拼接、回显检测
  audioserver.py 200 行  常驻进程生命周期（拉起/复用/预热/停止）
  roles.py      206 行   角色库（JSON 持久化 + 音频归一化）
  config.py     154 行   全部路径与参数，可用环境变量覆盖
  webui/index.html      单文件调试控制台（原生 JS，无构建）
tests/                  smoke_test.py（端到端 8 项）、test_payload.py（离线 6 项）
role/                   ★ 内置角色参考音频（15 个 WAV，12 MB，已入库）
models/AuK-GGUF/        ★ 模型权重 7.7 GB（GGUF，跨平台，需下载）
models/voice-library/   运行时音色库，服务端热扫描（角色名.wav，自动生成）
vendor/audio.cpp-cu124/ ⚙ 第三方推理运行时 2.0 GB（需下载，不入库）
vendor/soxr-ucrt/       ⚙ 三个重采样 DLL（740 KB，已入库）
vendor/server.json      ⚙ 常驻服务配置（绝对路径，换机器需改）
docs/AuK/               📖 上游资料（只读，含 PyTorch 参考实现）
data/                   💾 运行产物：角色索引、输出音频、临时文件与日志（不入库）
```

**缓存（可删，不影响运行）**：`.cache/uv`（6.6 GB，uv 包缓存）、`.cache/*.zst|json|md`、
`.playwright-cli/`、`app/__pycache__/`。
`.venv/`（5.4 GB）不是缓存，但可重建。

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 引擎、显存、依赖、预检（缺什么一目了然） |
| GET | `/roles` | 角色列表，`?source=builtin\|custom` |
| POST | `/roles` | 创建角色（multipart：`name` + `ref_audio`） |
| GET | `/roles/{id}` · `/roles/{id}/audio` | 角色详情 / 参考音频 |
| DELETE | `/roles/{id}` | 删除自建角色（内置返回 403） |
| POST | `/speech` | 语音合成，JSON，返回 `audio/wav` |
| POST | `/edit` | 指令式编辑/增强/分离，multipart，返回 `audio/wav` |
| GET | `/` · `/api` · `/docs` | WebUI / JSON 索引 / OpenAPI |

完整参数、响应头、错误码见 [API 手册](docs/API.md)。

## 常用启动参数

```powershell
--port 8080                 # 监听端口
--host 0.0.0.0              # 对外提供服务（默认仅本机）
--webui                     # 开启调试控制台
--tts-backend server|cli    # 默认 server（0.3s）；cli 为每请求加载（约 7s，仅调试）
--server-url http://127.0.0.1:8123
--no-server-autostart       # 只连接不拉起常驻进程
--gguf-variant flash|base   # 默认 flash（4 步）；base 更慢但编辑更准
--gguf-backend cuda|vulkan|cpu
--mem-saver                 # 阶段间释放权重，省显存但热请求变慢
```

`--engine` 可选 `auto|gguf|auk|mock`（默认 `auto`）：`gguf` 走 GGUF 运行时，`auk` 走 PyTorch
全精度栈，`mock` 输出占位音用于联调。常驻/CLI 的选择由 `--tts-backend` 控制。

## 文档

| 文档 | 内容 |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | 进程模型、模块职责、数据流、与上游 `auk.md` 逐条对照、关键不变量 |
| [API.md](docs/API.md) | 接口手册：参数、响应头、错误码、curl / Python 示例 |
| [DEPLOY.md](docs/DEPLOY.md) | 部署详解：依赖、环境变量全表、`server.json`、systemd、运维 |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | 已踩过的 10 个坑：症状 → 根因 → 处理 |

## 已知限制

- **不支持流式**（上游明确限制），首字延迟 = 整句生成时间；好在整句只需 0.3 s。
- **单句约 30 秒预算**（参考音频 + 目标合计）。长文本自动切块（每段 ≤15 s），
  语速保持自然（实测 4.86 字/秒）。
- **常驻模式单模型单槽**，请求串行排队；并发需求需多实例或 `--parallel-jobs`（显存翻倍）。
- **编辑走 CLI**（`/edit`，每请求约 7 s）：Linux CUDA 预编译包不含 CLI，需源码构建或改走
  `/v1/tasks/run`，详见 [DEPLOY](docs/DEPLOY.md#7-linux-部署)。

## 参考

- AuK 上游：<https://github.com/Tencent-Hunyuan/AuK>（`docs/AuK/` 有本地副本）
- GGUF 运行时：<https://github.com/0xShug0/audio.cpp>（模型文档 `docs/models/auk.md`）
- 量化权重：<https://huggingface.co/audio-cpp/AuK-Base-and-Flash-GGUF>
