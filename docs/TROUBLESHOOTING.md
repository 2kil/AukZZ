# 排错手册

每条都是本项目实际踩过的坑，按“症状 → 根因 → 处理”记录，附可复现的诊断命令。

---

## 1. `requires Soxr HQ resampling`

**症状**：`/speech` 带角色（参考音频）时返回 500/503，消息
`AuK audio conditioning requires Soxr HQ resampling`；Instruct 路径（无参考音频）正常。

**根因**：audio.cpp 在运行时 dlopen `libsoxr`，整条依赖链任何一环缺失都会报这个错：

```
libsoxr.dll  →  libgomp-1.dll  →  libwinpthread-1.dll
```

只缺最后一环也会报“soxr 不可用”，很容易误判成“懒加载竞态”。

**诊断**（不启动服务）：

```powershell
# 1) 看真实依赖链
C:\TDM-GCC-64\bin\objdump.exe -p vendor\audio.cpp-cu124\libsoxr.dll  | findstr "DLL Name"
C:\TDM-GCC-64\bin\objdump.exe -p vendor\audio.cpp-cu124\libgomp-1.dll | findstr "DLL Name"

# 2) 真实尝试加载
.venv\Scripts\python.exe -c "import ctypes;ctypes.WinDLL(r'vendor\audio.cpp-cu124\libsoxr.dll');print('ok')"

# 3) 服务内自检
curl -s http://127.0.0.1:8080/health | .venv\Scripts\python.exe -c """
import json,sys; d=json.load(sys.stdin)
print('soxr     :', d['gguf']['soxr'])
print('loadable :', d['engine']['soxr_loadable'])"""
```

**处理**：补齐三个 DLL（`vendor/soxr-ucrt/` 有副本），或 Linux 上 `apt install libsoxr0`。
`/health` 的 `gguf.soxr.loadable` 必须为 `true`。

**预防**：把 `vendor/audio.cpp-cu124` 加入杀软排除目录（本项目有一次 DLL 被静默删除）。

---

## 2. 输出是参考音频，或像“参考音频拼凑”

**症状**：能返回 200，但听起来不是要合成的句子，而是参考录音的内容/碎片。

**根因**（两种，必须区分）：

| 现象 | 根因 | 判定 |
|---|---|---|
| 输出**波形等于**参考音频 | soxr 未初始化，服务端静默降级返回 prompt 录音 | 与参考相关系数 ≈ 1.00 |
| 输出与文本无关但也不是参考 | 零样本路径**漏了 AuK 指令模板**，模型把台词当指令 | 与 CLI 基准相关系数 < 0.2 |

AuK 是指令跟随模型：带 `--voice-ref` 且不给 `instruct` 时，`--text` 必须是完整指令
（`Say the following with the same voice: "…"`）。常驻服务的 `input` 会**原样**转发，
所以模板必须由调用方拼好。

**诊断**：

```powershell
# 用 CLI（已知正确）生成基准，再比对
.venv\Scripts\python.exe -c "
from app import audio as A
print('与参考相关:', round(A.similarity('out.wav','models/voice-library/安若柔.wav'), 3))
print('与时基相关:', round(A.similarity('out.wav','baseline.wav'), 3))"
```

判读：与参考 1.00 → 走 §1；与基准 <0.2 且与参考 ≈0 → 检查指令模板。

**处理**：`app/engine.py::AukServerEngine._request()` 中的
`ZERO_SHOT_TEMPLATE.format(text=text) if voice else text`。已有防护：
`_guard_echo()` 在相关度 ≥0.85 时重启进程重试，仍失败返回 503；
`tests/test_payload.py` 离线断言该模板，防止回归。

---

## 3. `audiocpp_cli exited with code 3221225477`（0xC0000005）

**症状**：`X-Warning` 里出现该消息，但音频文件是完整可用的。

**根因**：audio.cpp 0.9.5 在写盘**之后**销毁阶段崩溃（Windows 上的堆损坏）。
本项目最初用 mingw(msvcrt) 自编 libsoxr，与 MSVC(UCRT) 宿主混用 CRT 导致；换成 MSYS2 UCRT 版后消失。

**处理**：

- 确认 soxr 是 UCRT 构建（`vendor/soxr-ucrt/`），不要用 TDM-GCC 自编版本。
- 引擎已按“产物有效性”判定成功：WAV 完整即返回 200，并在 `X-Warning` 如实说明。
- 若音频不完整则抛 500，此时才需要排查。

---

## 4. 长文本语速明显变快

**症状**：短句正常（约 4.8 字/秒），长段落读到 6.5+ 字/秒，像赶时间。

**根因**：AuK 需要显式目标时长，服务端估算是 `字数/4.6 + 标点×0.12 + 数字音节`，
但会被 `MAX_GEN_SECONDS=20` 截断；28 秒的内容塞进 20 秒 → 语速提高约 40%。

**诊断**：

```powershell
.venv\Scripts\python.exe -c "
from app.audio import estimate_seconds, split_text
t='（你的长文本）'
print('估算', round(estimate_seconds(t),2), 's -> 切块', split_text(t, max_seconds=15))"
```

**处理**：

- 默认已自动切块（`X-Mode` 末尾带 `+chunked`，`X-Notes` 说明段数），无需手工处理。
- 想放宽单句上限：`set AUK_MAX_GEN_SECONDS=28`，同时把参考音频换成更短的角色（≤5 s）。
- 注意硬上限不止 20 秒，还有“参考 + 目标 ≈ 30 秒”的模型预算。

---

## 5. 数字读得怪、带空格就卡顿

**症状**：`现在下单只要199元` 读得飞快；`优惠30%` 几乎挤在一起；`续航10.8s` 把 “s” 读成字母。

**根因**：早期估算器只统计汉字和英文词，**数字、百分号、单位一个都不计**，给的时长严重不足。

**处理**（已内置，无需操作）：

- `normalize_text()`：`30%` → `百分之30`、`10.8s` → `10.8秒`、`kg/ml/℃/min/h` 转中文单位、
  删除汉字与数字之间的多余空格。
- `estimate_seconds()`：按数字位数折算音节，`199` 从 1.52 s 修正为 2.38 s。

若自建引擎或改代码，务必保留这两个函数——**常驻服务不做任何文本规整**。

---

## 6. 音色描述被朗读出来

**症状**：Instruct 模式下（`voice_description`），输出里出现“一位二十多岁的女生…”这类描述内容。

**根因**：模板是 `请基于下面的描述: "{desc}",生成语音内容"{text}"`，描述里自带引号会破坏结构，
描述过长也容易被“续读”。

**处理**（已内置）：

- `clean_description()`：剥离 `" ' “ ” ‘ ’ 「 」`、折平换行、**超过 80 字截断**并给 `X-Notes`。
- 泄漏检测：输出时长 > max(2.2×估算, 估算+2 s) 时给 `X-Warning`，提示缩短描述或改用角色。

**根治**：对稳定性要求高时用角色参考音频（零样本克隆），它根本不经过描述。

---

## 7. `503 常驻语音服务不可用`

**排查顺序**：

```powershell
curl -s -m 2 http://127.0.0.1:8123/v1/models   # 常驻服务是否活着、是否 loaded
curl -s http://127.0.0.1:8080/health | .venv\Scripts\python.exe -c """
import json,sys; print(json.load(sys.stdin)['engine'])"""   # degraded / reason
tasklist | findstr audiocpp                    # 进程是否存在
```

| 情况 | 处理 |
|---|---|
| 8123 无响应 | 用 `--no-server-autostart` 启动的，需手动拉起 `audiocpp_server.exe --config vendor\server.json` |
| `model_loaded=false` | 还在加载（约 6 s），或 `lazy_load` 生效导致未预热，等首请求 |
| 进程存在但请求全 503 | 多为 soxr 重试耗尽，见 §1，重启常驻进程 |
| 进程被杀 | 引擎会自动重新拉起（实测约 10 s 恢复），不必手工干预 |

---

## 8. 启动慢、首请求慢

**正常值**：启动约 9 s（加载 + 双路预热）；热请求 0.25–0.61 s（3 秒音频）。

| 现象 | 原因 | 处理 |
|---|---|---|
| 首请求多等约 5 s | `server.json` 里 `lazy_load: true`，或预热失败 | 设 `lazy_load: false`，看启动日志是否两条“预热完成” |
| 每请求都 7 s 左右 | 走了 CLI 后端 | 检查 `/health` 的 `engine.active` 是否为 `gguf-server`；`--tts-backend server` |
| 启动要 20 s+ | 显卡正被其他程序占用，或首次加载需编译内核 | 关掉占显存的程序；后续启动会更快 |
| 编辑慢（约 7 s） | `/edit` 固定走 CLI | 属预期；如需加速见 [DEPLOY §7](DEPLOY.md#7-linux-部署) |

---

## 9. 某个角色被跳过 / 上传浮点 WAV 报 415

**症状**：`/health` 的 `roles.warnings` 出现 `skipped 潘雪柔.wav: unknown format: 3`。

**根因**：Python 标准库 `wave` 只支持 PCM，读不了 IEEE float（format 3）与 48 kHz 立体声。
早期版本因此漏掉内置角色，也拒绝用户上传的浮点 WAV。

**处理**（已内置）：`app/audio.py` 自带 RIFF 解析，支持 PCM 8/16/24/32、IEEE float 32/64、
`WAVE_FORMAT_EXTENSIBLE`，统一归一化为单声道 PCM16。若仍被跳过，用下列命令确认格式：

```powershell
.venv\Scripts\python.exe -c "
from app.audio import probe_wav; print(probe_wav('role/潘雪柔.wav'))"
```

---

## 10. PyTorch 全精度路线（`--engine auk`）跑不起来

**症状**：启动即 `[fatal] AuK inference unavailable: missing ...`，或加载后 CUDA OOM。

**这是预期行为**，不是 bug：

| 项目 | 需求 | 本机 |
|---|---|---|
| 权重 | 18.8 GB（base 6.12 + vae 0.64 + Qwen 12） | 未下载 |
| 显存 | 24.78 GiB（不开 offload）/ 16.98 GiB（开） | 15.99 GiB |
| 内存 | 开 offload 需驻留约 8 GB | 15.4 GB |

**处理**：换 ≥24 GB 显存的机器，然后

```powershell
.venv\Scripts\python.exe -m app.main --engine auk --cpu-offload --port 8080
```

代码无需改动。在 16 GB 卡上请继续用默认的 GGUF 路线。

---

## 诊断工具箱

```powershell
# 服务与依赖总览
curl -s http://127.0.0.1:8080/health | .venv\Scripts\python.exe -m json.tool

# 只看关键项
curl -s http://127.0.0.1:8080/health | .venv\Scripts\python.exe -c """
import json,sys; d=json.load(sys.stdin)
print({'engine': d['engine']['active'], 'loaded': d['engine']['model_loaded'],
       'missing': d['gguf']['missing'], 'soxr': d['gguf']['soxr']['loadable']})"""

# 单次请求的所有诊断头
curl -s -D - -o out.wav -X POST http://127.0.0.1:8080/speech ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"测试\",\"role_name\":\"安若柔\"}" | findstr /i "^x-"

# 音频是否真的是合成结果（而非参考回显）
.venv\Scripts\python.exe -c "
from app import audio as A
print('相关度', round(A.similarity('out.wav','models/voice-library/安若柔.wav'),3))"

# 离线回归（秒级，不依赖服务）
.venv\Scripts\python.exe tests\test_payload.py

# 端到端回归
.venv\Scripts\python.exe tests\smoke_test.py

# 常驻进程日志
Get-Content data\outputs\_tmp\audiocpp_server.log -Tail 40
```

## 一句话原则

遇到“没有报错但结果不对”，先做**波形比对**（与参考音频、与 CLI 基准各算一次相关度）。
这个项目里两次“听起来不对”都是这样定位的：一次是 soxr 静默降级（相关 1.00），
一次是指令模板缺失（相关 0.088）。日志和状态码在这两种情况下都是 200。
