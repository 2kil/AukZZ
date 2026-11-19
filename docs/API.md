# API 手册

服务地址默认 `http://127.0.0.1:8080`。除 `/roles`、`/edit` 为 `multipart/form-data` 外，
其余均为 JSON。交互式文档：`GET /docs`（OpenAPI），JSON 索引：`GET /api`。

## 通用约定

- 音频统一返回 `Content-Type: audio/wav`，24 kHz 单声道 PCM16。
- 错误统一为 `{"detail": "..."}`。`detail` 为中文时可直接展示给用户。
- 中文响应头（如 `X-Role-Name`、`X-Notes`）会做 percent-encoding，客户端需 `decodeURIComponent`。

### 响应头（所有返回音频的接口）

| 头 | 示例 | 说明 |
|---|---|---|
| `X-Engine` | `gguf-server` | 实际使用的引擎：`gguf-server` / `gguf` / `auk` / `mock` |
| `X-Mode` | `zero_shot_tts` | `zero_shot_tts` / `instruct_tts` / `edit`，切块时追加 `+chunked` |
| `X-Gen-Seconds` | `3.0` | 请求的目标时长 |
| `X-Audio-Duration` | `3.0` | 实际输出时长 |
| `X-Elapsed-MS` | `312` | 端到端耗时 |
| `X-Sample-Rate` | `24000` | 采样率 |
| `X-Role-Id` · `X-Role-Name` | `builtin_8ba19b2c` · 安若柔 | 使用的角色（有角色时） |
| `X-Source-Duration` | `8.64` | 源音频时长（仅 `/edit`） |
| `X-Notes` | 长文本已切分为 2 段… | **信息性提示**，请求成功 |
| `X-Warning` | 参考 24.6 秒 + 目标 8 秒超过约 30 秒预算… | **需要关注**，建议处理 |

> `X-Notes` 与 `X-Warning` 的分工很重要：前者是正常处理说明（截断、切块），后者才值得报警。

---

## GET /health

服务、引擎、依赖与预检状态。**上线前必看**。

```bash
curl -s http://127.0.0.1:8080/health | .venv\Scripts\python.exe -m json.tool
```

```json
{
  "status": "ok",
  "engine": {
    "active": "gguf-server",
    "degraded": false,
    "model_loaded": true,
    "server": "http://127.0.0.1:8123",
    "model": "auk-flash",
    "variant": "flash",
    "backend": "cuda",
    "voice_dir": "D:\\GitSum\\AukZZ\\models\\voice-library",
    "soxr_dll": true, "gomp_dll": true, "winpthread_dll": true, "soxr_loadable": true,
    "edit_backend": "cli"
  },
  "engine_requested": "auto",
  "roles": {"builtin": 15, "custom": 0, "library_dir": "...\\role", "warnings": []},
  "gguf":  {"ready": true, "missing": [], "soxr": {"loadable": true, "error": null}},
  "preflight": {"ready": false, "missing": ["AuK checkpoint ...", "Qwen2.5-Omni-3B ..."]}
}
```

- `engine.degraded=true` 表示未用上真实推理，`reason` 会给出原因。
- `gguf.missing` 非空 → GGUF 路线不可用（缺模型 / 缺 soxr / 缺 CUDA）。
- `preflight.missing` 非空 → PyTorch 全精度路线不可用（正常，16 GB 卡本就走 GGUF）。

---

## 角色

### GET /roles

```
GET /roles?source=builtin|custom
```

```json
{"total": 15, "roles": [{
  "id": "builtin_8ba19b2c", "name": "安若柔", "source": "builtin",
  "duration": 8.64, "sample_rate": 24000, "channels": 1,
  "description": "", "gender": "", "language": "", "tags": [], "created_at": ""
}]}
```

| 字段 | 说明 |
|---|---|
| `id` | 内置为 `builtin_<hash>`（随文件名稳定），自建为 `role_<hash>` |
| `source` | `builtin`（来自 `role/`）或 `custom`（`data/roles/`） |
| `name` | 角色名，也是音色库文件名 |

### POST /roles

`multipart/form-data`，创建自建角色。参考音频会被归一化为单声道 PCM16（支持 PCM 8/16/24/32
与 IEEE float 32/64、`WAVE_FORMAT_EXTENSIBLE`，即 48 kHz 立体声浮点也可直接上传）。

| 字段 | 必填 | 说明 |
|---|---|---|
| `name` | ✅ | 角色名；同名会返回 409 |
| `ref_audio` | ✅ | 参考音频（WAV），建议 3–15 秒，≤25 MiB |
| `description` / `gender` / `language` / `tags` | — | 元信息，`tags` 逗号分隔 |
| `trim_seconds` | — | 截取时长，默认 10 秒，`0` 表示不截断 |

```bash
curl -X POST http://127.0.0.1:8080/roles \
  -F "name=深夜电台" -F "gender=female" -F "language=zh" -F "tags=直播,温柔" \
  -F "ref_audio=@role/安若柔.wav;type=audio/wav"
```

`201` 返回角色对象。侧写：参考音频同时被复制进 `models/voice-library/<name>.wav`，
服务端热扫描，**新角色立即可用，无需重启**。

### GET /roles/{id} · GET /roles/{id}/audio · DELETE /roles/{id}

详情 / 下载参考音频 / 删除。删除内置角色返回 **403**；删除自建角色会同时清理音色库文件。

---

## POST /speech

语音合成。两种模式：

**A. 参考音频零样本克隆**（推荐，音色最稳）

```json
{
  "text": "欢迎来到直播间，今天有三款新品上架。",
  "role_name": "安若柔",
  "gen_seconds": 4.0,
  "seed": 42
}
```

**B. 指令式音色**（无参考音频）

```json
{
  "text": "今天天气真好。",
  "voice_description": "一位二十多岁的女生，声音清甜自然，语速稍慢"
}
```

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `text` | string | ✅ | 1–2000 字；数字与单位会自动规范化 |
| `role_id` / `role_name` | string | — | 二者择一；找不到返回 404 |
| `voice_description` | string | — | 无角色时必填，否则 422；>80 字自动截断 |
| `gen_seconds` | float | 自动估算 | 0.2–20；不传则按 `字数/4.6 + 标点×0.12 + 数字音节` 估算 |
| `speed` | float | 1.0 | 0.3–3.0，仅影响时长估算（不是模型语速控制） |
| `seed` | int | 随机 | 固定后同文本可复现 |
| `max_chunk_seconds` | float | 15 | 5–20，长文本切块时每段上限 |
| `nfe` / `cfg_strength` | int/float | 32 / 2.0 | 仅 Base 变体生效；Flash 固定 4 步、CFG=0 |

```bash
curl -o out.wav -D headers.txt -X POST http://127.0.0.1:8080/speech \
  -H "Content-Type: application/json" \
  -d '{"text":"欢迎来到直播间，今天有三款新品上架。","role_name":"安若柔","gen_seconds":4.0}'
```

行为说明：

- 估算时长 > 15 秒且字数 > 40 时自动切块，`X-Mode` 变为 `...+chunked`，
  `X-Notes` 说明切了几段；段间停顿按标点决定（句号 0.45 s、逗号 0.18 s）。
- 参考音频超过 10 秒会被截断（`X-Notes` 提示），因为音色编码耗时与参考长度正相关。
- 参考 + 目标合计超过约 30 秒预算时给 `X-Warning`，音质可能下降。

错误码：`404` 角色不存在 · `422` 文本为空 / 既无角色也无描述 · `503` 常驻服务不可用 · `500` 模型异常。

---

## POST /edit

指令式音频编辑，覆盖上游 17 类能力：内容/歌词编辑、音高/语速/音量、情绪/音色/去口音/非语声/耳语、
语音增强、人声与音乐分离、目标说话人提取。

`multipart/form-data`：

| 字段 | 必填 | 说明 |
|---|---|---|
| `instruction` | ✅ | 自然语言指令，英文模板效果最佳 |
| `audio` | ✅ | 源音频 WAV，≥0.3 秒、≤25 MiB（超过 30 秒会给预算风险提示） |
| `gen_seconds` | — | 目标时长（语速类编辑用），默认与源等长 |
| `seed` | — | 随机种子 |

```bash
curl -o enhanced.wav -X POST http://127.0.0.1:8080/edit \
  -F "instruction=Remove the background noise and reverb, keep the speech clean and natural." \
  -F "audio=@noisy.wav;type=audio/wav"
```

常用指令模板：

| 任务 | 指令 |
|---|---|
| 音高 +2 半音 | `Raise the pitch by 2 semitones.` |
| 音量 +10 dB | `Increase the volume by 10 dB.` |
| 语速放慢 | `Slow down the speech slightly.`（配合 `gen_seconds`） |
| 情绪 | `Change the emotion to happy.` / `sad` / `angry` |
| 音色 | `Change the timbre to a deeper, warmer voice.` |
| 去口音 | `Remove the accent and speak in standard Mandarin.` |
| 耳语 | `Convert the speech to a whisper.` |
| 语音增强 | `Remove the background noise and make the voice cleaner.` |
| 人声分离 | `Extract the vocals and remove the background music.` |
| 目标说话人 | `Keep only the first speaker and remove the others.` |

**注意**：编辑固定走 CLI 引擎（`X-Engine: gguf`），单次约 7 秒。指令里不要传 `instruct` 风格描述。

错误码：`415` 非 WAV · `422` 指令为空 / 音频过短 · `413` 超 25 MiB · `500` 模型异常。

---

## 其他

| 路径 | 说明 |
|---|---|
| `GET /` | WebUI 调试控制台（`--webui` 开启时），否则返回 JSON 索引 |
| `GET /api` | JSON 索引，始终可用 |
| `GET /docs` | Swagger UI |
| `GET /sw.js` | `204`，静默浏览器探测 |

## Python 调用示例

```python
import requests

BASE = "http://127.0.0.1:8080"

def speak(text: str, role: str = "安若柔", seconds: float | None = None) -> bytes:
    r = requests.post(f"{BASE}/speech", json={
        "text": text, "role_name": role,
        **({"gen_seconds": seconds} if seconds else {}),
    }, timeout=120)
    r.raise_for_status()
    print(r.headers.get("X-Engine"), r.headers.get("X-Elapsed-MS"), "ms")
    return r.content

open("out.wav", "wb").write(speak("欢迎来到直播间。"))
```

直播场景建议：固定 `seed` 便于复现；同一句复用缓存；播当前句时预取下一句。
