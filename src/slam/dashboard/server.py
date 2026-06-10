import asyncio
import json
import os
import threading
import numbers
from dataclasses import dataclass

from aiohttp import web

from .client import build_hud_html


@dataclass(frozen=True)
class HudServerContext:
    stream_state: dict
    stop_event: threading.Event
    recording_event: threading.Event
    hud_telemetry: dict
    hud_lock: threading.Lock
    cam_ctrl_lock: threading.Lock
    cam_state: dict
    layout_file: str
    exposure_time_us: int
    iso_sensitivity: int
    target_score_good: float
    laplacian_pass_threshold: float
    target_depth_pct: float


def _create_app(ctx: HudServerContext) -> web.Application:
    routes = web.RouteTableDef()

    def _jsonify(value):
        if isinstance(value, dict):
            return {k: _jsonify(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonify(v) for v in value]
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            return float(value)
        if hasattr(value, "item"):
            try:
                return _jsonify(value.item())
            except Exception:
                pass
        return str(value)

    @routes.get("/")
    async def index(request):
        return web.Response(text=build_hud_html(ctx), content_type="text/html")

    @routes.get("/set_wb")
    async def set_wb(request):
        try:
            val = int(request.query.get("v", 4600))
            with ctx.cam_ctrl_lock:
                ctx.cam_state["wb"] = max(2500, min(8000, val))
        except Exception:
            pass
        return web.Response(text="OK")

    @routes.get("/set_exp")
    async def set_exp(request):
        try:
            val = int(request.query.get("v", 15000))
            with ctx.cam_ctrl_lock:
                ctx.cam_state["exp"] = max(1000, min(33000, val))
        except Exception:
            pass
        return web.Response(text="OK")

    @routes.get("/set_iso")
    async def set_iso(request):
        try:
            val = int(request.query.get("v", 800))
            with ctx.cam_ctrl_lock:
                ctx.cam_state["iso"] = max(100, min(1600, val))
        except Exception:
            pass
        return web.Response(text="OK")

    @routes.get("/stream")
    async def stream(request):
        response = web.StreamResponse(headers={
            "Cache-Control": "no-cache,private",
            "Content-Type": "multipart/x-mixed-replace;boundary=FRAME",
        })
        await response.prepare(request)
        try:
            while not ctx.stop_event.is_set():
                frame = ctx.stream_state.get("latest_jpeg")
                if frame:
                    await response.write(b"--FRAME\r\n")
                    await response.write(b"Content-Type: image/jpeg\r\n")
                    await response.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    await response.write(frame)
                    await response.write(b"\r\n")
                await asyncio.sleep(0.033)
        except Exception as e:
            import traceback
            print(f"[STREAM ERROR] {e}")
            traceback.print_exc()
        return response

    @routes.get("/telemetry")
    async def telemetry(request):
        with ctx.hud_lock:
            payload = ctx.hud_telemetry.copy()
        payload["recording"] = ctx.recording_event.is_set()
        return web.json_response(_jsonify(payload))

    @routes.get("/layout")
    async def get_layout(request):
        if not os.path.exists(ctx.layout_file):
            return web.json_response({})
        try:
            with open(ctx.layout_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return web.json_response(data)
        except (OSError, json.JSONDecodeError):
            pass
        return web.json_response({})

    @routes.post("/layout")
    async def set_layout(request):
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")

        if not isinstance(payload, dict):
            return web.Response(status=400, text="Layout must be an object")

        sanitized = {}
        for panel_id, state in payload.items():
            if not isinstance(panel_id, str) or not isinstance(state, dict):
                continue
            width = state.get("width", "")
            height = state.get("height", "")
            transform = state.get("transform", "translate(0px, 0px)")
            if not isinstance(width, str) or not isinstance(height, str) or not isinstance(transform, str):
                continue
            sanitized[panel_id] = {
                "width": width[:32],
                "height": height[:32],
                "transform": transform[:64],
            }

        try:
            with open(ctx.layout_file, "w", encoding="utf-8") as f:
                json.dump(sanitized, f)
        except OSError as e:
            return web.Response(status=500, text=f"Failed to save layout: {e}")

        return web.json_response({"ok": True})

    app = web.Application()
    app.add_routes(routes)
    return app


def _run_server(ctx: HudServerContext) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = _create_app(ctx)
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    loop.run_until_complete(site.start())

    async def watch_stop():
        while not ctx.stop_event.is_set():
            await asyncio.sleep(1)
        await runner.cleanup()
        loop.stop()

    loop.create_task(watch_stop())
    try:
        loop.run_forever()
    finally:
        loop.close()


def start_hud_server(ctx: HudServerContext) -> threading.Thread:
    thread = threading.Thread(target=_run_server, args=(ctx,), daemon=True)
    thread.start()
    return thread
