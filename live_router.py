"""
Live audio WebSocket router for the Iraqi Arabic Whisper/Qwen FastAPI server.

Install in the same directory as iraqi_listener.py. At the END of
iraqi_listener.py, after summarize_transcript and app are defined:

    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from live_router import build_live_router

    app.include_router(
        build_live_router(
            whisper_getter=lambda: whisper_model,
            summarize_fn=summarize_transcript,
            whisper_lock=whisper_lock,
        )
    )
    app.mount("/web", StaticFiles(directory="web"), name="web")

    @app.get("/", include_in_schema=False)
    def web_home():
        return FileResponse("web/index.html")

faster-whisper accepts a float32 NumPy waveform. This router converts browser
PCM16 audio to normalized float32 samples and uses the server's existing
whisper_lock to serialize CPU inference with /transcribe.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Callable

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect


def _resample_pcm16(pcm: np.ndarray, source_rate: int, target_rate: int = 16000) -> np.ndarray:
    if pcm.size == 0:
        return np.empty(0, dtype=np.float32)
    # Normalize PCM16 to the float32 range expected by faster-whisper.
    signal = pcm.astype(np.float32) / 32768.0
    if source_rate == target_rate:
        return signal
    new_len = max(1, round(len(signal) * target_rate / source_rate))
    old_x = np.arange(len(signal), dtype=np.float64)
    new_x = np.linspace(0, len(signal) - 1, new_len)
    return np.interp(new_x, old_x, signal).astype(np.float32)


def build_live_router(
    whisper_getter: Callable,
    summarize_fn: Callable,
    whisper_lock=None,
):
    router = APIRouter()

    @router.websocket("/ws/live")
    async def live_audio(websocket: WebSocket):
        await websocket.accept()
        client = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "?"
        print(f"[live {client}] WebSocket connected")
        model = whisper_getter()
        if model is None:
            await websocket.send_json({
                "type": "error",
                "message": "Whisper model is not loaded.",
            })
            await websocket.close()
            return

        language = "ar"
        sample_rate = 48000
        source = "microphone"
        pcm_parts: list[np.ndarray] = []
        buffered_samples = 0
        session_transcript: list[str] = []
        processed_seconds = 0.0
        last_summary_at = 0.0
        chunk_seconds = 15
        summary_interval_seconds = 60
        stopped = False
        received_bytes = 0
        last_level_report = 0.0

        async def send_summary(final: bool = False):
            nonlocal last_summary_at
            transcript = "\n".join(session_transcript).strip()
            if not transcript:
                return
            if not final and processed_seconds - last_summary_at < summary_interval_seconds:
                return

            await websocket.send_json({
                "type": "status",
                "message": "Updating rolling summary...",
            })
            try:
                # summarize_transcript already uses the application's summary_lock.
                summary = await asyncio.to_thread(summarize_fn, transcript)
                if summary:
                    await websocket.send_json({
                        "type": "summary",
                        "summary": summary,
                    })
                last_summary_at = processed_seconds
            except Exception as exc:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Summary: {type(exc).__name__}: {exc}",
                })

        async def transcribe_buffer():
            nonlocal pcm_parts, buffered_samples, processed_seconds
            if not pcm_parts:
                return

            joined = np.concatenate(pcm_parts)
            pcm_parts = []
            buffered_samples = 0

            audio = _resample_pcm16(joined, sample_rate, 16000)
            if audio.size < 1600:
                return

            seconds = audio.size / 16000.0
            processed_seconds += seconds
            rms = float(np.sqrt(np.mean(audio ** 2)))
            print(f"[live {client}] Transcribing {seconds:.1f}s chunk (rms={rms:.4f})...")
            started = time.perf_counter()

            def run_whisper():
                def consume():
                    segments, info = model.transcribe(
                        audio,
                        language=None if language == "auto" else language,
                        beam_size=1,
                        temperature=0,
                        condition_on_previous_text=False,
                        vad_filter=True,
                    )
                    # Consume generator while still holding the model lock.
                    return [
                        segment.text.strip()
                        for segment in segments
                        if segment.text and segment.text.strip()
                    ]

                if whisper_lock is None:
                    return consume()
                with whisper_lock:
                    return consume()

            try:
                texts = await asyncio.to_thread(run_whisper)
            except Exception as exc:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Whisper: {type(exc).__name__}: {exc}",
                })
                return

            text = " ".join(texts).strip()
            elapsed = time.perf_counter() - started
            if text:
                print(f"[live {client}] Whisper {elapsed:.1f}s -> {text!r}")
            else:
                print(f"[live {client}] Whisper {elapsed:.1f}s -> no speech detected")
                await websocket.send_json({
                    "type": "status",
                    "message": f"No speech in the last {seconds:.0f}s"
                    + (" (audio is silent)" if rms < 0.001 else ""),
                })
            if text:
                session_transcript.append(text)
                await websocket.send_json({
                    "type": "transcript",
                    "text": text,
                    "source": source,
                })

            await send_summary(final=False)

        try:
            first = await websocket.receive_text()
            config = json.loads(first)
            if config.get("type") != "start":
                await websocket.send_json({
                    "type": "error",
                    "message": "Send start configuration first.",
                })
                await websocket.close()
                return

            sample_rate = int(config.get("sample_rate", 48000))
            if not 8000 <= sample_rate <= 192000:
                raise ValueError("Unsupported browser sample rate.")

            language = config.get("language", "ar")
            if language not in ("ar", "en", "auto"):
                language = "ar"

            source = config.get("source", "microphone")
            print(f"[live {client}] Start: source={source} language={language} sample_rate={sample_rate}")
            await websocket.send_json({
                "type": "status",
                "message": "Live audio connected.",
            })

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    print(f"[live {client}] Client disconnected")
                    break

                if message.get("bytes") is not None:
                    raw = message["bytes"]
                    # Browser sends signed 16-bit little-endian mono PCM.
                    if len(raw) % 2:
                        raw = raw[:-1]
                    chunk = np.frombuffer(raw, dtype="<i2").copy()
                    if chunk.size:
                        pcm_parts.append(chunk)
                        buffered_samples += len(chunk)
                        received_bytes += len(raw)
                        now = time.monotonic()
                        if now - last_level_report >= 5:
                            last_level_report = now
                            peak = int(np.abs(chunk.astype(np.int32)).max())
                            buffered = buffered_samples / sample_rate
                            print(f"[live {client}] Received {received_bytes / 1024:.0f} KB, "
                                  f"buffered {buffered:.1f}s, peak={peak}")
                            await websocket.send_json({
                                "type": "status",
                                "message": (f"Receiving audio · {buffered:.0f}/{chunk_seconds}s buffered"
                                            + (" · SILENT input" if peak < 30 else "")),
                            })

                    if buffered_samples >= sample_rate * chunk_seconds:
                        await transcribe_buffer()

                elif message.get("text") is not None:
                    control = json.loads(message["text"])
                    if control.get("type") == "stop":
                        await transcribe_buffer()
                        await send_summary(final=True)
                        final_text = "\n".join(session_transcript).strip()
                        await websocket.send_json({
                            "type": "finished",
                            "transcript": final_text,
                        })
                        stopped = True
                        break

        except WebSocketDisconnect:
            print(f"[live {client}] Client disconnected")
        except Exception as exc:
            print(f"[live {client}] Error: {type(exc).__name__}: {exc}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                })
            except Exception:
                pass
        finally:
            if not stopped:
                try:
                    await websocket.close()
                except Exception:
                    pass

    return router
