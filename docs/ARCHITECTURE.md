# 架构与实现说明

本文说明服务怎么组织、一次请求怎么流动、哪些设计是“必须保持的不变量”，以及
与上游 `audio.cpp/docs/models/auk.md` 的逐条对应关系。

## 1. 进程模型

```
┌──────────────────────────────┐         ┌─────────────────────────────────┐
│  FastAPI (app.main)          │  HTTP   │  audiocpp_server (C++, ggml)    │
│  127.0.0.1:8080              │ ──────▶ │  127.0.0.1:8123                 │
│                              │         │                                 │
│  · 角色库 / 音色库同步        │         │  · AuK-Flash q8_0 常驻显存       │
│  · 文本规范化 / 时长估算      │         │  · /v1/audio/speech             │
│  · 长文本切块 / 拼接 / 对齐   │         │  · 音色库热扫描（voice_dir）     │
│  · 回显检测 / 进程自愈        │         │                                 │
│  · WebUI / 日志              │         │  约 6.2 GB 显存                  │
└──────────────────────────────┘         └─────────────────────────────────┘
                                                        ▲
                          编辑路径（低频）              │ subprocess
                          ┌─────────────────────────────┘
                          │  audiocpp_cli --task gen（每次加载模型，约 7 s）
```

为什么拆两个进程：AuK 的 Python 栈需要 18.8 GB 权重与 ≥16.98 GiB 显存（开 CPU offload 的官方实测值），
16 GB 卡跑不起来；GGUF 量化后只要 6.5 GB 权重、7.6 GB 显存，而执行 GGUF 需要 C++ 运行时。
业务逻辑（角色、切块、规范化）仍全部在 Python，C++ 只负责“文本 → 波形”。

常驻进程由 `app/audioserver.py` 管理：复用已运行实例 → 否则拉起 → 轮询 `/v1/models` 的
`loaded` → 预热 → 退出时只回收自己拉起的进程。

## 2. 模块职责

| 文件 | 行数 | 职责 | 关键点 |
|---|---|---|---|
| `app/main.py` | 467 | 路由、请求/响应模型、响应头、CLI 参数、lifespan | 上传大小限制 25 MiB；错误码映射 |
| `app/engine.py` | 1159 | 四个引擎 + 两个预检 | 引擎选择与降级；`_request` 重试；`_guard_echo` |
| `app/audio.py` | 491 | WAV 解析/归一化、时长估算、文本规范化、切块、拼接 | 纯函数，无副作用，可单测 |
| `app/audioserver.py` | 200 | 常驻进程生命周期 | `ensure/spawn/wait_ready/warmup/stop` |
| `app/roles.py` | 206 | 角色库 | 内置扫描 `role/` + 自建存 `data/roles/` |
| `app/config.py` | 154 | 路径与参数 | 全部可用环境变量覆盖；平台分支 |
| `app/webui/index.html` | 524 | 调试控制台 | 单文件、原生 JS、无构建、无新依赖 |

## 3. 一次 `/speech` 的完整流程

```
1. 解析请求          role_id / role_name / voice_description 三选一
2. 查找角色          RoleStore.get()；找不到 → 404
3. 文本规范化        normalize_text()：30%→百分之30，10.8s→10.8秒，去掉汉字与数字间空格
4. 描述清洗          clean_description()：剥引号、折换行、>80 字截断（仅 instruct 路径）
5. 时长估算          estimate_seconds()：汉字/4.6 + 英文词/2.5 + 标点×0.12 + 数字音节
6. 是否切块          未指定 gen_seconds 且估算 > 15 s 且字数 > 40 → _synthesize_chunked()
7. 音色库同步        prepare_voice()：角色 WAV → 单声道 PCM16、截到 10 s → voice-library/<角色名>.wav
8. 组请求            有 voice → input 套 "Say the following with the same voice: \"…\""
                     无 voice → input 裸文本 + options.instruct
9. 转发常驻服务      POST /v1/audio/speech {model, input, voice?, seed, options{duration_sec[,instruct]}}
10. 失败重试         5xx 且消息含 soxr/resampling → 0.8 s 后重试，最多 3 次；
                     ConnectionError → 重新拉起常驻进程后重试
11. 回显检测         _guard_echo()：与 voice-library 参考音频相关度 ≥0.85 → 判定 soxr 未初始化
                     → 重启进程重试；仍 ≥0.85 → 503
12. 切块拼接（若切） join_audio()：裁首尾静音 → ±6 dB 内 RMS 对齐 → 15 ms 淡入淡出 → 按标点插停顿
13. 返回             audio/wav + X-Engine / X-Mode / X-Elapsed-MS / X-Notes / X-Warning
```

## 4. 引擎矩阵

| 引擎 | 触发 | 速度 | 用途 |
|---|---|---|---|
| `AukServerEngine` (`gguf-server`) | 默认（`--tts-backend server`） | ~0.3 s/句 | 生产、直播 |
| `AukGgufEngine` (`gguf`) | `--tts-backend cli` 或常驻不可用 | ~7 s/句 | 调试、`/edit` |
| `AukEngine` (`auk`) | `--engine auk`，需 PyTorch 权重 | 数秒 | 有 ≥24 GB 显存时的全精度路线 |
| `MockEngine` (`mock`) | `--engine mock` 或全部不可用 | ~0 | 联调接口，输出占位音 |

选择顺序（`build_engine`）：`mock` → `server`（常驻就绪则用，失败可回退）→ `gguf`（CLI 预检通过）
→ `auk`（PyTorch 权重齐备）→ `mock`（降级，并在 `/health` 与响应头标注原因）。

`--engine gguf|auk` 为强约束：预检不过直接 `SystemExit`，不会静默降级——避免把占位音
当成真实合成结果发给业务方。

## 5. 必须保持的不变量

这几条都是踩过坑换来的，改动时务必保持：

1. **零样本必须套完整指令**：`Say the following with the same voice: "{text}"`。
   常驻服务的 `input` 会原样转发给模型，裸文本会被当作“指令”而非“要说的内容”。
   实测：裸文本与 CLI 基准相关度 0.088，套模板后 1.000。
2. **instruct 路径不要套模板**：`input` 保持裸文本 + `options.instruct`，服务端自己会套。
   手动套中文模板相关度掉到 0.116。
3. **编辑不传 `instruct`**（上游约定），`/edit` 走 `--task gen --audio … --text <指令>`。
4. **Flash 与 Base 的分工**：TTS 用 Flash（4 步、CFG=0），编辑用 Base（32 步、CFG=2）。
   Flash 做情绪/音色编辑区分度只有 Base 的 1/3（谱质心差 68 Hz vs 173 Hz）。
5. **切块上限 15 s**，留出 20 s 硬上限的余量；每块的种子按 `seed + 段序号` 递增。
6. **拼接前必须裁首尾静音**，否则每处接缝会多出约 0.9 s 空档。
7. **回显阈值 0.85**：同一文件相关度 1.00，正常合成 -0.05，余量充足，不会误杀。

## 6. 与上游 `docs/models/auk.md` 的对照

| 上游文档 | 本项目 |
|---|---|
| `--task tts --family auk --model <dir> --backend cuda` | 一致 |
| 组件选择 `auk.variant` / `auk.model_gguf` / `auk.qwen_gguf` / `auk.vae_gguf` | 四项都显式传 |
| 指令 TTS：`--text 词句` + `--request-option instruct=描述` | 一致 |
| 零样本：`--voice-ref`；不传 instruct 时 `--text` 必须含完整指令 | 一致（见不变量 1） |
| 编辑：`--task gen --audio` + `--text 指令`，不得传 instruct | 一致 |
| `duration_sec` / `num_inference_steps`(32) / `guidance_scale`(2) / `sway_sampling_coef`(-1) | 前三个按变体传；`sway` 用默认 -1 |
| `--seed`，`-1` 为随机 | 一致 |
| 输出单声道 24 kHz | 一致 |
| 输入音频内部重采样（soxr） | 依赖 `libsoxr`，见 TROUBLESHOOTING |
| 不支持流式 | 已确认 |
| `auk.mem_saver` | `--mem-saver` 开关 |

**三处刻意偏离**（均在文档允许的参数范围内）：

- Qwen 用 `q8_0` 而非默认 BF16（省约 1.7 GB 显存）；默认变体用 `flash` 而非 `base`（0.3 s 的关键）。
- 走常驻 HTTP 服务而非每次 CLI（文档只写 CLI 用法，HTTP 端点在 audio.cpp 的 server 文档里）。

**文档之外补的服务层**（与模型能力无关，直播场景必需）：时长估算、数字/单位规范化、
长文本切块与拼接、WAV 格式归一化、参考音频截断、回显检测。

## 7. 失败与自愈

| 情况 | 行为 |
|---|---|
| 常驻进程未启动 | 自动拉起（`--no-server-autostart` 可关闭），等待模型 loaded 后预热 |
| 首次加载 soxr 失败 | 0.8 s 后重试 ×3；仍失败返回 503，消息指明检查 `libsoxr.dll` 等 |
| 常驻进程被杀 / OOM | 下次请求捕获 `ConnectionError` → 重新拉起 → 重试（实测约 10 s 恢复） |
| 输出等于参考音频 | 判定为静默降级 → 重启进程重试 → 仍失败 503 |
| 模型级异常（如 CUDA OOM） | 500 + 异常类型与消息；堆栈写入服务日志 |

## 8. 测试

| 文件 | 类型 | 覆盖 |
|---|---|---|
| `tests/smoke_test.py` | 端到端（需服务运行） | health、角色列表、建角色、合成、音频下载、编辑、删角色（8 项） |
| `tests/test_payload.py` | 离线（不依赖服务） | 零样本必须套模板、instruct 必须裸文本（6 项） |

```powershell
.venv\Scripts\python.exe tests\test_payload.py            # 离线，秒级
.venv\Scripts\python.exe tests\smoke_test.py --base http://127.0.0.1:8080
```
