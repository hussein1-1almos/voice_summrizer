
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from live_router import build_live_router
import accounts
import security


# ============================================================
# Configuration
# ============================================================

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")

# This is a very large model. Override with another compatible
# Hugging Face model ID if your server cannot fit it in RAM.
SUMMARY_MODEL_NAME = os.getenv(
    "SUMMARY_MODEL_NAME",
    "Qwen/Qwen3-32B",
)

CPU_THREADS = max(1, (os.cpu_count() or 2) - 1)
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg",
    ".webm", ".mp4", ".mpeg", ".mpga",
}

SUPPORTED_LANGUAGES = {"ar", "en", "auto"}

# Limits the number of transcript tokens passed to the LLM
# in each summarization request.
SUMMARY_CHUNK_TOKENS = 4000
SUMMARY_MAX_NEW_TOKENS = 500

whisper_model = None
summary_tokenizer = None
summary_model = None

# Protect CPU inference if multiple requests arrive together.
whisper_lock = threading.Lock()
summary_lock = threading.Lock()

# Uploads are processed one at a time on this worker thread, detached from the
# HTTP request, so a closed browser tab does not cancel them.
upload_jobs = ThreadPoolExecutor(max_workers=1, thread_name_prefix="upload-job")


# ============================================================
# Model initialization
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model, summary_tokenizer, summary_model

    print(f"Loading Whisper {WHISPER_MODEL_SIZE} on CPU...")

    whisper_model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        cpu_threads=CPU_THREADS,
        num_workers=1,
    )

    print("Whisper loaded.")

    print(f"Loading summary model: {SUMMARY_MODEL_NAME}")
    print("This may take a long time and require substantial RAM.")

    summary_tokenizer = AutoTokenizer.from_pretrained(
        SUMMARY_MODEL_NAME,
        trust_remote_code=False,
    )

    summary_model = AutoModelForCausalLM.from_pretrained(
        SUMMARY_MODEL_NAME,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    summary_model.to("cpu")
    summary_model.eval()

    print("Summary model loaded.")
    print("API startup complete.")

    try:
        yield
    finally:
        whisper_model = None
        summary_model = None
        summary_tokenizer = None
        print("Models released.")


app = FastAPI(
    title="Whisper Transcription and Iraqi Arabic Summary API",
    description=(
        "Transcribes uploaded audio and generates a cautious "
        "Iraqi Arabic summary. The summary may contain uncertainty "
        "because speech recognition can be inaccurate."
    ),
    version="2.0.0",
    lifespan=lifespan,
    # The interactive API docs describe every endpoint to anyone; off unless asked for.
    docs_url="/docs" if os.getenv("ENABLE_API_DOCS") == "1" else None,
    redoc_url=None,
    openapi_url="/openapi.json" if os.getenv("ENABLE_API_DOCS") == "1" else None,
)
security.install(app)

# Uploads waiting or running per user; more are refused so one account
# cannot fill the server's queue.
MAX_PENDING_UPLOADS_PER_USER = 3


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "whisper_loaded": whisper_model is not None,
        "summary_model_loaded": summary_model is not None,
        "whisper_model": WHISPER_MODEL_SIZE,
        "summary_model": SUMMARY_MODEL_NAME,
        "device": "cpu",
    }


# ============================================================
# Summarization
# ============================================================

def summarize_chunk(transcript: str) -> str:
    """
    Summarize one transcript chunk in Iraqi Arabic.
    Treat the transcript as imperfect speech recognition,
    not as a guaranteed accurate record.
    """
    if not transcript.strip():
        return "ماكو نص واضح كافي حتى ينلخّص."

    system_prompt = """
    أنت مساعد يلخّص رسائل صوتية واجتماعات عمل. المتحدث يتكلم باللهجة العراقية ويخلط الإنجليزية بالعربية.
    النص ناتج عن تحويل الكلام إلى نص وقد تظهر فيه كلمات مشوّهة.

    القواعد:
    1. اكتب بالعربية الفصحى المبسطة، وأبقِ المصطلحات التقنية بالإنجليزية كما هي (JSON, Front-end, access...).
    2. افهم المعنى من السياق. الأجزاء المشوشة تماماً تجاهلها، ولا تذكرها ولا تفسرها.
    3. لا تضف أي معلومة غير موجودة في النص، ولا تخمّن أسماء أو أرقام أو تواريخ.
    4. انتبه للنفي (ما، مو، لا) ولا تحوّله إلى إثبات.
    5. انسب المهام والأفعال إلى "المتحدث" ما لم يُذكر اسم آخر.
    6. احذف أي قسم لا توجد له معلومات.
    7. أخرج الناتج بصيغة Markdown فقط، بدون أي مقدمة أو خاتمة وبدون ``` حول الناتج.

    الصيغة (Markdown):
    # عنوان قصير للمحادثة

    ## الملخص
    جملتان إلى ثلاث.

    ## المستجدات
    - ...

    ## القرارات
    - ...

    ## العوائق
    - ...

    ## الخطوات القادمة
    - [ ] المهمة — **المسؤول**، الموعد إن ذُكر
    """

    user_prompt = (
        "لخّص النص التالي بحذر، والتزم بالتعليمات السابقة:\n\n"
        f"النص:\n{transcript}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Qwen3 supports disabling its reasoning/thinking output
    # through enable_thinking=False in compatible Transformers versions.
    try:
        prompt = summary_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Compatibility fallback for older Transformers versions.
        prompt = summary_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = summary_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=SUMMARY_CHUNK_TOKENS + 1500,
    )

    inputs = {key: value.to("cpu") for key, value in inputs.items()}

    with summary_lock, torch.inference_mode():
        output = summary_model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True, temperature=0.3, top_p=0.8, top_k=20,
            pad_token_id=summary_tokenizer.eos_token_id,
            eos_token_id=summary_tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = output[0][input_length:]

    result = summary_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    if not result:
        raise RuntimeError("The summary model returned an empty result.")

    return result


SUMMARY_SECTION_NAMES = (
    "العنوان", "الملخص", "المستجدات", "القرارات", "العوائق",
    "الخطوات القادمة", "الخطوات التالية", "المهام",
)
ACTION_SECTION_NAMES = ("الخطوات القادمة", "الخطوات التالية", "المهام")


def normalize_summary_markdown(text: str) -> str:
    """
    Make the model output consistent Markdown, even when it falls back
    to the older "Label: value" style or wraps the answer in a code fence.
    Bullets under the next-steps section become "- [ ]" task items.
    """
    text = text.strip()
    fence = re.fullmatch(r"```[a-zA-Z]*\s*\n(.*?)\n?```", text, re.S)
    if fence:
        text = fence.group(1).strip()

    lines = []
    in_actions = False
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        heading = None

        md_heading = re.match(r"^(#{1,4})\s+(.+?)\s*:?\s*$", line)
        bold_heading = re.match(r"^\*\*(.+?)\*\*\s*:?\s*(.*)$", line)
        label = re.match(r"^(" + "|".join(SUMMARY_SECTION_NAMES) + r")\s*:\s*(.*)$", line)

        if md_heading:
            heading, rest = md_heading.group(2), ""
            level = len(md_heading.group(1))
        elif label or (bold_heading and bold_heading.group(1).rstrip(":") in SUMMARY_SECTION_NAMES):
            match = label or bold_heading
            heading, rest = match.group(1).rstrip(":"), match.group(2).strip()
            level = 2
            if heading == "العنوان":
                # "العنوان: X" becomes the document title "# X".
                heading, rest, level = (rest or heading), "", 1

        if heading is not None:
            in_actions = any(name in heading for name in ACTION_SECTION_NAMES)
            lines.append(f"{'#' * level} {heading}")
            if rest:
                lines.append(rest)
            continue

        if in_actions:
            bullet = re.match(r"^(?:[-*•]|\d+[.)])\s+(?!\[[ xX]\])(.+)$", line)
            if bullet:
                line = f"- [ ] {bullet.group(1)}"
        lines.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def split_transcript(transcript: str) -> list[str]:
    """
    Split long transcripts by tokenizer tokens, rather than
    character count, to avoid exceeding the model input limit.
    """
    token_ids = summary_tokenizer.encode(
        transcript,
        add_special_tokens=False,
    )

    if not token_ids:
        return []

    chunks = []
    for start in range(0, len(token_ids), SUMMARY_CHUNK_TOKENS):
        chunk_ids = token_ids[start:start + SUMMARY_CHUNK_TOKENS]
        chunk_text = summary_tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
        ).strip()

        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def summarize_transcript(transcript: str) -> str:
    """
    Summarize the whole transcript. Long recordings are summarized
    in chunks, then the chunk summaries are combined.
    """
    if not transcript.strip():
        return "ماكو نص كافي حتى ينلخّص."

    chunks = split_transcript(transcript)

    if not chunks:
        return "ماكو نص كافي حتى ينلخّص."

    if len(chunks) == 1:
        return normalize_summary_markdown(summarize_chunk(chunks[0]))

    print(f"Summarizing {len(chunks)} transcript chunks...")

    partial_summaries = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"Summarizing chunk {index}/{len(chunks)}...")
        partial_summaries.append(summarize_chunk(chunk))

    combined = "\n\n".join(
        f"مقطع {index}:\n{summary}"
        for index, summary in enumerate(partial_summaries, start=1)
    )

    # If the partial summaries are still too long, combine them in
    # smaller groups until the final input fits the token limit.
    while len(summary_tokenizer.encode(
        combined, add_special_tokens=False
    )) > SUMMARY_CHUNK_TOKENS:
        summary_ids = summary_tokenizer.encode(
            combined,
            add_special_tokens=False,
        )
        shortened = []

        for start in range(0, len(summary_ids), SUMMARY_CHUNK_TOKENS):
            piece_ids = summary_ids[start:start + SUMMARY_CHUNK_TOKENS]
            piece = summary_tokenizer.decode(
                piece_ids,
                skip_special_tokens=True,
            )
            shortened.append(summarize_chunk(piece))

        combined = "\n\n".join(shortened)

        # Safety guard against a model repeatedly producing summaries
        # that do not shrink.
        if len(shortened) >= len(partial_summaries):
            break
        partial_summaries = shortened

    return normalize_summary_markdown(summarize_chunk(combined))


# ============================================================
# Transcription endpoint
# ============================================================
app.include_router(accounts.build_accounts_router())
app.include_router(
    build_live_router(
        whisper_getter=lambda: whisper_model,
        summarize_fn=summarize_transcript,
        whisper_lock=whisper_lock,
        authenticate=accounts.websocket_user,
        create_note=accounts.create_note,
        save_note=accounts.save_note,
    )
)
app.mount("/web", StaticFiles(directory="web"), name="web")

@app.get("/", include_in_schema=False)
def web_home():
    return FileResponse("web/index.html")

@app.post("/transcribe")
def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(default="ar"),
    include_summary: str = Form(default="true"),
    user: dict = Depends(accounts.require_user),
):
    # Parsed by hand: a bool Form field came through as False when the
    # client omitted it, silently skipping the summary.
    include_summary = include_summary.strip().lower() not in ("false", "0", "no", "off")
    if whisper_model is None:
        raise HTTPException(
            status_code=503,
            detail="Whisper model is not loaded.",
        )

    if include_summary and (
        summary_model is None or summary_tokenizer is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Summary model is not loaded.",
        )

    filename = Path(file.filename or "audio").name
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. Supported formats: "
                + ", ".join(sorted(ALLOWED_EXTENSIONS))
            ),
        )

    language = language.strip().lower()
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail="language must be 'ar', 'en', or 'auto'.",
        )

    temp_path = None

    try:
        # Write the uploaded audio directly. FFmpeg/PyAV decoding
        # is handled on the server by faster-whisper.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:
            temp_path = temp_file.name
            total_bytes = 0

            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Audio file exceeds the 100 MB limit.",
                    )

                temp_file.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty.",
            )

        if accounts.count_notes(user["id"], status="processing") >= MAX_PENDING_UPLOADS_PER_USER:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"You already have {MAX_PENDING_UPLOADS_PER_USER} recordings processing. "
                    "Wait for one to finish, then upload again."
                ),
            )

        note_id = accounts.create_note(
            user["id"],
            title=Path(filename).stem[:200],
            source="upload",
            language=language,
            status="processing",
        )
        upload_jobs.submit(
            process_upload_job, note_id, temp_path, filename, total_bytes, language, include_summary,
        )
        print(f"Queued {filename} ({total_bytes} bytes) as note {note_id}.")
        # The job owns the temporary file from here on and deletes it when done.
        temp_path = None

        return {
            "note_id": note_id,
            "status": "processing",
            "message": (
                "Processing on the server. You can close this page; the note "
                "is saved to your account when it finishes."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Upload failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Upload failed. Please try again.",
        ) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

        try:
            file.file.close()
        except Exception:
            pass


def process_upload_job(note_id, temp_path, filename, total_bytes, language, include_summary):
    """
    Runs on the upload_jobs worker thread, independent of the HTTP request,
    so closing the browser does not stop it. Results go straight to the note.
    """
    try:
        whisper_language = None if language == "auto" else language

        print(f"[note {note_id}] Transcribing {filename} ({total_bytes} bytes)...")

        with whisper_lock:
            segments, info = whisper_model.transcribe(
                temp_path,
                language=whisper_language,
                beam_size=5,
                temperature=0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )

            # faster-whisper returns a generator; consume it while
            # still inside the lock.
            text_parts = [
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            ]

        transcript = " ".join(text_parts).strip()

        if not transcript:
            accounts.save_note(
                note_id,
                status="ready",
                error=(
                    "Whisper did not detect clear speech. "
                    "This does not prove the audio contains no speech."
                ),
            )
            print(f"[note {note_id}] No speech detected.")
            return

        # Save the transcript first so it is visible while the summary runs.
        accounts.save_note(note_id, transcript=transcript)

        summary = None
        summary_error = ""
        if include_summary:
            print(f"[note {note_id}] Generating summary...")
            try:
                summary = summarize_transcript(transcript)
            except Exception as exc:
                # Keep the transcript even if summarization fails.
                print(f"[note {note_id}] Summarization failed: {type(exc).__name__}: {exc}")
                summary_error = "The summary could not be generated. The transcript is saved."

        accounts.save_note(note_id, summary=summary, status="ready", error=summary_error)
        print(f"[note {note_id}] Done ({info.duration:.0f}s of audio).")

    except Exception as exc:
        print(f"[note {note_id}] Processing failed: {type(exc).__name__}: {exc}")
        accounts.save_note(
            note_id,
            status="failed",
            error="This recording could not be processed. Check that the file is a valid audio file and upload it again.",
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
