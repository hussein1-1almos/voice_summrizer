
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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
)


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

    الصيغة:
    العنوان: ...
    الملخص: (جملتان إلى ثلاث)
    المستجدات:
    - ...
    العوائق:
    - ...
    الخطوات القادمة:
    - المهمة (المسؤول، الموعد إن ذُكر)
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
        return summarize_chunk(chunks[0])

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

    return summarize_chunk(combined)


# ============================================================
# Transcription endpoint
# ============================================================

@app.post("/transcribe")
def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(default="ar"),
    include_summary: bool = Form(default=True),
):
    print(f"DEBUG: Running file: {__file__}")
    print(f"DEBUG: include_summary={include_summary}")
    print(f"DEBUG: summary_model_loaded={summary_model is not None}")
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

        whisper_language = None if language == "auto" else language

        print(f"Transcribing {filename} ({total_bytes} bytes)...")

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
            segment_results = []
            text_parts = []

            for segment in segments:
                segment_text = segment.text.strip()
                if segment_text:
                    text_parts.append(segment_text)

                segment_results.append({
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment_text,
                })

        transcript = " ".join(text_parts).strip()

        if not transcript:
            return {
                "text": "",
                "summary": None,
                "summary_error": None,
                "warning": (
                    "Whisper did not detect clear speech. "
                    "This does not prove the audio contains no speech."
                ),
                "language": info.language,
                "language_probability": round(
                    info.language_probability, 4
                ),
                "duration_seconds": round(info.duration, 2),
                "model": WHISPER_MODEL_SIZE,
                "summary_model": SUMMARY_MODEL_NAME,
                "segments": segment_results,
            }

        summary = None
        summary_error = None

        if include_summary:
            print("Generating summary...")
            try:
                summary = summarize_transcript(transcript)
            except Exception as exc:
                # Return the transcript even if summarization fails.
                summary_error = (
                    f"Summarization failed: {type(exc).__name__}: {exc}"
                )
                print(summary_error)

        return {
            "text": transcript,
            "summary": summary,
            "summary_error": summary_error,
            "warning": (
                "The transcript and summary may contain recognition "
                "errors. Verify important details against the original "
                "audio. The summary cannot reliably recover speech "
                "that Whisper missed."
            ),
            "language": info.language,
            "language_probability": round(
                info.language_probability, 4
            ),
            "duration_seconds": round(info.duration, 2),
            "model": WHISPER_MODEL_SIZE,
            "summary_model": SUMMARY_MODEL_NAME if include_summary else None,
            "segments": segment_results,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Transcription request failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {type(exc).__name__}: {exc}",
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
