"""AuK speech generation HTTP service.

Run:
    .venv\\Scripts\\python.exe -m app.main --host 127.0.0.1 --port 8080

Endpoints
---------
GET    /health                 service + engine + GPU status
GET    /roles                  list built-in (./role) and custom roles
POST   /roles                  create a role (multipart: metadata + reference audio)
GET    /roles/{role_id}        role detail
GET    /roles/{role_id}/audio  download the role's reference audio
DELETE /roles/{role_id}        delete a custom role
POST   /speech                 synthesize WAV (zero-shot TTS with a role, or instruct TTS)
GET    /docs                   interactive OpenAPI UI
"""

from __future__ import annotations

import argparse
import logging
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from . import audio as audio_utils
from . import audioserver
from . import config
from .engine import EngineUnavailable, MockEngine, build_engine, gguf_preflight, preflight
from .roles import RoleStore

logger = logging.getLogger("auk.service")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
WEBUI_DIR = Path(__file__).resolve().parent / "webui"


class SpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="要合成的文本")
    role_id: str | None = Field(None, description="角色 id（内置或自建），与 role_name 二选一")
    role_name: str | None = Field(None, description="角色名称，例如「龙安雅」")
    voice_description: str | None = Field(
        None, description="不提供参考音频时使用的音色描述（AuK Instruct TTS）"
    )
    speed: float = Field(1.0, gt=0.3, le=3.0, description="语速系数，仅影响时长估算")
    gen_seconds: float | None = Field(
        None, gt=0.2, le=config.MAX_GEN_SECONDS, description="显式目标时长（秒）"
    )
    seed: int | None = Field(None, description="随机种子")
    max_chunk_seconds: float | None = Field(
        None, ge=5, le=20,
        description="长文本自动分段时每段的目标上限（秒）；不传则用服务端默认 15 秒",
    )
    nfe: int = Field(32, ge=1, le=128, description="扩散采样步数（AuK-Flash 固定 4）")
    cfg_strength: float = Field(2.0, ge=0.0, le=10.0, description="CFG 强度（AuK-Flash 固定 0）")


def _ascii_header(value: str) -> str:
    """HTTP headers are latin-1; percent-encode anything else."""
    try:
        value.encode("latin-1")
        return value
    except UnicodeEncodeError:
        return urllib.parse.quote(value, safe=" ,:;/()[]'")


def create_app(*, engine_mode: str = "auto", device: str = "cuda", dtype: str = "bf16",
               cpu_offload: bool = False, prefer: str = "flash", preload: bool = False,
               gguf_variant: str = "flash", gguf_backend: str = "cuda",
               mem_saver: bool = False, webui: bool = False,
               tts_backend: str = "server", server_url: str | None = None,
               server_autostart: bool | None = None) -> FastAPI:
    config.ensure_dirs()
    store = RoleStore()

    try:
        engine, requested, fallback_reason = build_engine(
            engine_mode, device=device, dtype=dtype, cpu_offload=cpu_offload,
            prefer=prefer, preload=preload, gguf_variant=gguf_variant,
            gguf_backend=gguf_backend, mem_saver=mem_saver,
            tts_backend=tts_backend, server_url=server_url,
            server_autostart=server_autostart,
        )
    except EngineUnavailable as exc:
        # explicit --engine with a broken stack: refuse to start rather than serve mock audio
        raise SystemExit(f"[fatal] {exc}") from exc

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        yield
        proc = getattr(engine, "server_process", None)
        if proc is not None and getattr(engine, "server_spawned", False):
            audioserver.stop(proc)

    app = FastAPI(
        title="AuK Speech Service",
        version="0.1.0",
        description="基于 Tencent AuK 的语音生成服务：角色管理 + 零样本 TTS",
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )
    app.state.store = store
    app.state.engine = engine

    app.state.engine_requested = requested
    app.state.fallback_reason = fallback_reason

    if fallback_reason:
        logger.warning("running in DEGRADED mode — %s", fallback_reason)

    def _audio_response(result, *, role=None, extra: dict | None = None) -> Response:
        """Shared WAV response builder: audio body + diagnostics headers."""
        headers = {
            "X-Engine": result.engine,
            "X-Mode": result.mode,
            "X-Gen-Seconds": f"{result.gen_seconds}",
            "X-Audio-Duration": f"{result.duration}",
            "X-Elapsed-MS": f"{int(result.elapsed * 1000)}",
            "X-Sample-Rate": f"{result.sample_rate}",
        }
        if role is not None:
            headers["X-Role-Id"] = _ascii_header(role.id)
            headers["X-Role-Name"] = _ascii_header(role.name)
        if result.notes:
            headers["X-Notes"] = _ascii_header(" | ".join(result.notes))
        if result.warnings:
            headers["X-Warning"] = _ascii_header(" | ".join(result.warnings))
        for key, value in (extra or {}).items():
            headers[key] = _ascii_header(str(value))

        headers["Content-Disposition"] = f'inline; filename="auk_{uuid.uuid4().hex[:8]}.wav"'
        return Response(content=result.wav_bytes, media_type="audio/wav", headers=headers)

    # ------------------------------------------------------------------ health
    def api_index() -> dict:
        body = {
            "service": "AuK Speech Service",
            "version": app.version,
            "engine": app.state.engine.status(),
            "webui": bool(webui),
            "endpoints": {
                "GET /health": "服务与引擎状态",
                "GET /roles": "角色列表（内置 + 自建）",
                "POST /roles": "创建角色（multipart，含参考音频）",
                "GET /roles/{role_id}": "角色详情",
                "GET /roles/{role_id}/audio": "角色参考音频",
                "DELETE /roles/{role_id}": "删除自建角色",
                "POST /speech": "语音合成（零样本克隆 / Instruct TTS），返回 audio/wav",
                "POST /edit": "音频编辑/增强/分离（指令 + 源音频），返回 audio/wav",
                "GET /api": "本索引（JSON）",
            },
        }
        if webui:
            body["endpoints"]["GET /"] = "WebUI 控制台（HTML）"
        return body

    @app.get("/sw.js", include_in_schema=False)
    def _no_service_worker() -> Response:
        """Silence browsers/extensions probing for a service worker."""
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    @app.get("/api", summary="服务索引（JSON）")
    def api_json() -> dict:
        return api_index()

    if webui:
        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def webui_console() -> HTMLResponse:
            page = WEBUI_DIR / "index.html"
            if not page.is_file():
                raise HTTPException(status_code=500, detail=f"webui asset missing: {page}")
            return HTMLResponse(page.read_text(encoding="utf-8"))
    else:
        @app.get("/", summary="服务索引")
        def root_index() -> dict:
            return api_index()

    @app.get("/health", summary="健康检查")
    def health() -> dict:
        return {
            "status": "ok",
            "engine": app.state.engine.status(),
            "engine_requested": app.state.engine_requested,
            "roles": store.stats(),
            "gguf": gguf_preflight(),
            "preflight": preflight(),
        }

    # ------------------------------------------------------------------- roles
    @app.get("/roles", summary="角色列表")
    def list_roles(source: str | None = Query(None, description="builtin / custom")) -> dict:
        roles = store.list()
        if source:
            roles = [r for r in roles if r.source == source]
        return {"total": len(roles), "roles": [r.to_public() for r in roles]}

    @app.post("/roles", status_code=201, summary="创建角色")
    async def create_role(
        name: str = Form(..., description="角色名称"),
        ref_audio: UploadFile = File(..., description="参考音频（WAV，建议 3-15 秒）"),
        description: str = Form("", description="角色描述"),
        gender: str = Form("", description="性别（可选）"),
        language: str = Form("zh", description="语言标签"),
        tags: str = Form("", description="逗号分隔的标签"),
        trim_seconds: float = Form(
            config.REF_TRIM_SECONDS, description="参考音频截取时长（秒），0 表示不截断"
        ),
    ) -> dict:
        name = name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="name must not be empty")

        suffix = Path(ref_audio.filename or "").suffix.lower()
        payload = await ref_audio.read()
        await ref_audio.close()
        if not payload:
            raise HTTPException(status_code=422, detail="reference audio is empty")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"reference audio exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB",
            )
        if suffix and suffix != ".wav":
            raise HTTPException(
                status_code=415,
                detail=f"unsupported format '{suffix}'; upload 16-bit PCM WAV (audio/convert to wav first)",
            )

        staging = config.TMP_DIR / f"upload_{uuid.uuid4().hex}.wav"
        staging.write_bytes(payload)
        try:
            if not audio_utils.is_riff_wave(staging):
                raise HTTPException(status_code=415, detail="file is not a valid RIFF/WAVE audio")
            probe = audio_utils.probe_wav(staging)
            try:
                role = store.create(
                    name=name,
                    upload_path=staging,
                    description=description,
                    gender=gender,
                    language=language,
                    tags=[t.strip() for t in tags.split(",")],
                    trim_seconds=trim_seconds,
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            staging.unlink(missing_ok=True)

        warnings = []
        if probe["sample_rate"] != config.TARGET_SAMPLE_RATE:
            warnings.append(
                f"reference is {probe['sample_rate']} Hz; AuK resamples to {config.TARGET_SAMPLE_RATE} Hz internally"
            )
        if role.duration < 2:
            warnings.append("reference shorter than 2s may reduce voice-cloning quality")
        body = role.to_public()
        body["warnings"] = warnings
        return body

    @app.get("/roles/{role_id}", summary="角色详情")
    def get_role(role_id: str) -> dict:
        role = store.get(role_id)
        if role is None:
            raise HTTPException(status_code=404, detail=f"role not found: {role_id}")
        return role.to_public()

    @app.get("/roles/{role_id}/audio", summary="角色参考音频")
    def get_role_audio(role_id: str) -> Response:
        role = store.get(role_id)
        if role is None:
            raise HTTPException(status_code=404, detail=f"role not found: {role_id}")
        path = Path(role.audio_path)
        if not path.is_file():
            raise HTTPException(status_code=410, detail="reference audio is no longer available")
        return Response(
            content=path.read_bytes(),
            media_type="audio/wav",
            headers={"Content-Disposition": f'inline; filename="{_ascii_header(role.name)}.wav"'},
        )

    @app.delete("/roles/{role_id}", status_code=200, summary="删除自建角色")
    def delete_role(role_id: str) -> dict:
        role = store.get(role_id)
        if role is None:
            raise HTTPException(status_code=404, detail=f"role not found: {role_id}")
        if role.source == "builtin":
            raise HTTPException(status_code=403, detail="built-in roles from ./role cannot be deleted")
        store.delete(role.id)
        forget = getattr(app.state.engine, "forget_voice", None)
        if callable(forget):
            forget(role.name)
        return {"deleted": role.id, "name": role.name}

    # ------------------------------------------------------------------ speech
    @app.post("/speech", summary="语音合成（返回 audio/wav）")
    def speech(payload: SpeechRequest) -> Response:
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="text must not be empty")

        role = None
        if payload.role_id:
            role = store.get(payload.role_id)
            if role is None:
                raise HTTPException(status_code=404, detail=f"role not found: {payload.role_id}")
        elif payload.role_name:
            role = store.get(payload.role_name)
            if role is None:
                raise HTTPException(status_code=404, detail=f"role not found: {payload.role_name}")
        if role is None and not payload.voice_description:
            raise HTTPException(
                status_code=422,
                detail="provide role_id/role_name, or voice_description for instruct TTS",
            )

        ref_audio = Path(role.audio_path) if role else None
        try:
            result = app.state.engine.synthesize(
                text,
                ref_audio=ref_audio,
                voice_description=payload.voice_description,
                gen_seconds=payload.gen_seconds,
                speed=payload.speed,
                seed=payload.seed,
                nfe=payload.nfe,
                cfg_strength=payload.cfg_strength,
                max_chunk_seconds=payload.max_chunk_seconds,
                role_name=role.name if role else None,
            )
        except EngineUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ConnectionError as exc:
            raise HTTPException(status_code=503, detail=f"常驻语音服务不可用：{exc}") from exc
        except HTTPException:
            raise
        except Exception as exc:  # model-level failure (e.g. CUDA OOM)
            logger.exception("synthesis failed")
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

        return _audio_response(result, role=role)

    # -------------------------------------------------------------------- edit
    @app.post("/edit", summary="音频编辑/增强/分离（指令驱动，返回 audio/wav）")
    async def edit_audio(
        instruction: str = Form(..., description="自然语言编辑指令（英文模板效果最佳）"),
        audio: UploadFile = File(..., description="源音频（WAV）"),
        gen_seconds: float | None = Form(None, description="目标时长（秒），默认与源等长"),
        seed: int | None = Form(None, description="随机种子"),
    ) -> Response:
        instruction = instruction.strip()
        if not instruction:
            raise HTTPException(status_code=422, detail="instruction must not be empty")

        payload = await audio.read()
        await audio.close()
        if not payload:
            raise HTTPException(status_code=422, detail="source audio is empty")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"source audio exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB",
            )

        staging = config.TMP_DIR / f"edit_{uuid.uuid4().hex}.wav"
        staging.write_bytes(payload)
        try:
            if not audio_utils.is_riff_wave(staging):
                raise HTTPException(status_code=415, detail="file is not a valid RIFF/WAVE audio")
            source = audio_utils.probe_wav(staging)
            if source["duration"] < 0.3:
                raise HTTPException(status_code=422, detail="source audio shorter than 0.3s")
            result = app.state.engine.edit(
                instruction,
                staging,
                duration_seconds=gen_seconds,
                seed=seed,
            )
        except EngineUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("edit failed")
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        finally:
            staging.unlink(missing_ok=True)

        return _audio_response(result, extra={"X-Source-Duration": f"{source['duration']}"})

    @app.exception_handler(EngineUnavailable)
    def _engine_unavailable(_request, exc: EngineUnavailable) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="AuK speech generation service")
    parser.add_argument("--host", default=config.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=config.DEFAULT_PORT)
    parser.add_argument("--engine", default=config.DEFAULT_ENGINE,
                        choices=["auto", "gguf", "auk", "mock"],
                        help="auto: gguf -> auk -> mock; gguf: audio.cpp; auk: python stack; mock: placeholder")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--prefer", default="flash", choices=["flash", "base"],
                        help="preferred AuK variant when both checkpoints exist")
    parser.add_argument("--preload", action="store_true", help="load the model during startup")
    parser.add_argument("--gguf-variant", default=config.GGUF_VARIANT, choices=["flash", "base"],
                        help="AuK-Flash (4-step) or AuK base schedule for the GGUF backend")
    parser.add_argument("--gguf-backend", default=config.GGUF_BACKEND,
                        choices=["cuda", "vulkan", "cpu"], help="audio.cpp compute backend")
    parser.add_argument("--mem-saver", action="store_true",
                        help="GGUF: release weights between stages (lower VRAM, slower)")
    parser.add_argument("--tts-backend", default=config.TTS_BACKEND,
                        choices=["server", "cli"],
                        help="server: 常驻 audiocpp_server（快，推荐）；cli: 每次调用加载（调试用）")
    parser.add_argument("--server-url", default=config.SERVER_URL,
                        help="常驻服务地址，默认 http://127.0.0.1:8123")
    parser.add_argument("--no-server-autostart", action="store_true",
                        help="不允许自动拉起常驻服务；未就绪时回退 CLI")
    parser.add_argument("--webui", "-webui", action="store_true",
                        help="在 / 提供 WebUI 调试控制台（JSON 索引移到 /api）")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    application = create_app(
        engine_mode=args.engine,
        device=args.device,
        dtype=args.dtype,
        cpu_offload=args.cpu_offload,
        prefer=args.prefer,
        preload=args.preload,
        gguf_variant=args.gguf_variant,
        gguf_backend=args.gguf_backend,
        mem_saver=args.mem_saver,
        webui=args.webui,
        tts_backend=args.tts_backend,
        server_url=args.server_url,
        server_autostart=not args.no_server_autostart,
    )
    status = application.state.engine.status()
    logger.info("listening on http://%s:%s (engine=%s, backend=%s)", args.host, args.port,
                status.get("active"), args.tts_backend)
    if args.webui:
        logger.info("webui console: http://%s:%s/", args.host, args.port)
    uvicorn.run(application, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
