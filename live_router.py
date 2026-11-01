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

# Sessions still finishing after their page closed. Holding a reference keeps
# the tasks from being garbage-collected before they complete.
_background_sessions: set[asyncio.Task] = set()

# Limits against one account tying up the server.
MAX_LIVE_SESSIONS_PER_USER = 1
MAX_BACKLOG_SECONDS = 20 * 60  # audio waiting for Whisper before capture is ended
_live_sessions: dict = {}


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
    authenticate: Callable | None = None,
    create_note: Callable | None = None,
    save_note: Callable | None = None,
):
    """
    authenticate(websocket) -> user dict or None; when given, unauthenticated
    connections are rejected. create_note(user_id, title, source, language) -> id
    and save_note(note_id, transcript=..., summary=...) persist the session.
    """
    router = APIRouter()

    @router.websocket("/ws/live")
    async def live_audio(websocket: WebSocket):
        await websocket.accept()
        client = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "?"
        user = authenticate(websocket) if authenticate else None
        if authenticate and user is None:
            await websocket.send_json({"type": "error", "message": "Please sign in again."})
            await websocket.close(code=4401)
            return
        print(f"[live {client}] WebSocket connected" + (f" (user={user['username']})" if user else ""))
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
        note_id = None
        received_bytes = 0
        last_level_report = 0.0

        # The receive loop must never wait for Whisper or the summary model:
        # if it does, stop/ping/disconnect frames are not read while inference
        # runs and the connection drops once Whisper is slower than real time.
        # Audio chunks are queued for a worker task; summaries run as their
        # own background task, one at a time.
        chunks: asyncio.Queue = asyncio.Queue()
        queued_seconds = 0.0
        summary_task: asyncio.Task | None = None
        worker_task: asyncio.Task | None = None
        send_lock = asyncio.Lock()
        client_gone = False

        async def send(payload: dict):
            # After the page closes the session keeps running and saving;
            # messages for the browser are simply dropped.
            nonlocal client_gone
            if client_gone:
                return
            try:
                async with send_lock:
                    await websocket.send_json(payload)
            except Exception:
                client_gone = True

        async def run_summary(transcript: str):
            await send({
                "type": "status",
                "message": "Updating rolling summary...",
            })
            started = time.perf_counter()
            try:
                # summarize_transcript already uses the application's summary_lock.
                summary = await asyncio.to_thread(summarize_fn, transcript)
                print(f"[live {client}] Summary updated in {time.perf_counter() - started:.1f}s")
                if summary:
                    if note_id is not None:
                        save_note(note_id, summary=summary)
                    await send({
                        "type": "summary",
                        "summary": summary,
                    })
            except Exception as exc:
                print(f"[live {client}] Summary failed: {type(exc).__name__}: {exc}")
                await send({
                    "type": "error",
                    "message": "The summary could not be updated. The transcript is still being saved.",
                })

        def maybe_start_summary():
            nonlocal summary_task, last_summary_at
            if summary_task is not None and not summary_task.done():
                return
            if processed_seconds - last_summary_at < summary_interval_seconds:
                return
            transcript = "\n".join(session_transcript).strip()
            if not transcript:
                return
            last_summary_at = processed_seconds
            summary_task = asyncio.create_task(run_summary(transcript))

        async def transcribe_chunk(joined: np.ndarray):
            nonlocal processed_seconds
            audio = _resample_pcm16(joined, sample_rate, 16000)
            if audio.size < 1600:
                return

            seconds = audio.size / 16000.0
            processed_seconds += seconds
            rms = float(np.sqrt(np.mean(audio ** 2)))
            print(f"[live {client}] Transcribing {seconds:.1f}s chunk (rms={rms:.4f}, "
                  f"{queued_seconds:.0f}s more queued)...")
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
                print(f"[live {client}] Whisper failed: {type(exc).__name__}: {exc}")
                await send({
                    "type": "error",
                    "message": "Part of the audio could not be transcribed.",
                })
                return

            text = " ".join(texts).strip()
            elapsed = time.perf_counter() - started
            if elapsed > seconds:
                print(f"[live {client}] WARNING: Whisper is slower than real time "
                      f"({elapsed:.1f}s for {seconds:.1f}s of audio); transcripts will lag.")
            if text:
                print(f"[live {client}] Whisper {elapsed:.1f}s -> {text!r}")
                session_transcript.append(text)
                if note_id is not None:
                    save_note(note_id, transcript="\n".join(session_transcript))
                await send({
                    "type": "transcript",
                    "text": text,
                    "source": source,
                })
            else:
                print(f"[live {client}] Whisper {elapsed:.1f}s -> no speech detected")
                await send({
                    "type": "status",
                    "message": f"No speech in the last {seconds:.0f}s"
                    + (" (audio is silent)" if rms < 0.001 else ""),
                })

            maybe_start_summary()

        async def transcription_worker():
            nonlocal queued_seconds
            while True:
                joined = await chunks.get()
                if joined is None:
                    return
                queued_seconds -= joined.size / sample_rate
                await transcribe_chunk(joined)

        async def finish_session() -> str:
            """Transcribe whatever is still queued, then write the final summary."""
            flush_buffer()
            chunks.put_nowait(None)
            try:
                await worker_task
            except Exception as exc:
                print(f"[live {client}] Transcription worker failed: {type(exc).__name__}: {exc}")
            # Read summary_task only now: the worker may have started a new one.
            if summary_task is not None:
                await asyncio.gather(summary_task, return_exceptions=True)
            final_text = "\n".join(session_transcript).strip()
            if final_text:
                await run_summary(final_text)
            if note_id is not None:
                save_note(note_id, status="ready")
            print(f"[live {client}] Session finished ({len(final_text)} characters)")
            return final_text

        def flush_buffer():
            nonlocal pcm_parts, buffered_samples, queued_seconds
            if not pcm_parts:
                return
            joined = np.concatenate(pcm_parts)
            pcm_parts = []
            buffered_samples = 0
            queued_seconds += joined.size / sample_rate
            chunks.put_nowait(joined)

        user_key = user["id"] if user else client
        if _live_sessions.get(user_key, 0) >= MAX_LIVE_SESSIONS_PER_USER:
            await websocket.send_json({
                "type": "error",
                "message": "You already have a live session running. Stop it before starting another.",
            })
            await websocket.close(code=4429)
            return
        _live_sessions[user_key] = _live_sessions.get(user_key, 0) + 1

        try:
            first = await websocket.receive_text()
            config = json.loads(first)
            if config.get("type") != "start":
                await send({
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
            if source not in ("microphone", "meeting-tab"):
                source = "microphone"
            print(f"[live {client}] Start: source={source} language={language} sample_rate={sample_rate}")
            if user is not None and create_note is not None:
                label = "Meeting" if source == "meeting-tab" else "Recording"
                note_id = create_note(user["id"], f"{label} · {time.strftime('%Y-%m-%d %H:%M')}", source, language,
                                      status="live")
                await send({"type": "session", "note_id": note_id})
            worker_task = asyncio.create_task(transcription_worker())
            await send({
                "type": "status",
                "message": "Live audio connected.",
            })

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    print(f"[live {client}] Client disconnected (code={message.get('code')})")
                    break
                if worker_task.done() and worker_task.exception():
                    raise worker_task.exception()

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
                                  f"buffered {buffered:.1f}s, waiting for Whisper {queued_seconds:.0f}s, "
                                  f"peak={peak}")
                            status = f"Receiving audio · {buffered:.0f}/{chunk_seconds}s buffered"
                            if queued_seconds >= chunk_seconds:
                                status += f" · transcription {queued_seconds:.0f}s behind"
                            if peak < 30:
                                status += " · SILENT input"
                            await send({"type": "status", "message": status})

                    if buffered_samples >= sample_rate * chunk_seconds:
                        flush_buffer()
                        if queued_seconds > MAX_BACKLOG_SECONDS:
                            # Whisper cannot keep up; stop taking audio instead of
                            # queueing without limit. What was received is still saved.
                            print(f"[live {client}] Backlog {queued_seconds:.0f}s; ending capture")
                            await send({
                                "type": "error",
                                "message": "The server is too far behind to keep transcribing live. "
                                           "Capture stopped; the audio received so far is being saved.",
                            })
                            break

                elif message.get("text") is not None:
                    control = json.loads(message["text"])
                    if control.get("type") == "stop":
                        remaining = queued_seconds + buffered_samples / sample_rate
                        print(f"[live {client}] Stop requested; finishing {remaining:.0f}s of queued audio")
                        await send({
                            "type": "status",
                            "message": f"Finishing the last {remaining:.0f}s of audio...",
                        })
                        final_text = await finish_session()
                        await send({
                            "type": "finished",
                            "transcript": final_text,
                        })
                        stopped = True
                        break

        except WebSocketDisconnect as exc:
            print(f"[live {client}] Client disconnected (code={exc.code})")
        except Exception as exc:
            print(f"[live {client}] Error: {type(exc).__name__}: {exc}")
            try:
                await send({
                    "type": "error",
                    "message": "Something went wrong with the live session. The audio received so far is being saved.",
                })
            except Exception:
                pass
        finally:
            _live_sessions[user_key] -= 1
            if not _live_sessions[user_key]:
                del _live_sessions[user_key]
            if not stopped and note_id is not None and worker_task is not None:
                # The page was closed (or the network dropped) mid-session: finish
                # the queued audio and the final summary anyway, and save them.
                client_gone = True
                print(f"[live {client}] Page closed mid-session; finishing note {note_id} in the background")
                task = asyncio.create_task(finish_session())
                _background_sessions.add(task)
                task.add_done_callback(_background_sessions.discard)
            else:
                for task in (worker_task, summary_task):
                    if task is not None and not task.done():
                        task.cancel()
            if not stopped:
                try:
                    await websocket.close()
                except Exception:
                    pass

    return router
