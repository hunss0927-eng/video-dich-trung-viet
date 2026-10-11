"""
🎬 Tool Dịch & Lồng Tiếng Video đa ngôn ngữ (Gemini + Edge-TTS + FFmpeg)

Cài đặt:
    pip install -U streamlit google-genai edge-tts numpy pandas gTTS
    (cần FFmpeg: có cả `ffmpeg` và `ffprobe` trong PATH)
    Trên Streamlit Cloud: tạo file packages.txt chứa dòng `ffmpeg`.

Chạy:
    streamlit run app.py
"""

import asyncio
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
import wave

import edge_tts
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

try:  # giọng dự phòng khi Edge-TTS bị chặn / lỗi mạng
    from gtts import gTTS
except Exception:  # noqa: BLE001
    gTTS = None

# ----------------------------------------------------------------------------
# Cấu hình
# ----------------------------------------------------------------------------
SAMPLE_RATE = 24000      # Edge-TTS xuất mp3 24 kHz mono
CHUNK_SECONDS = 360      # video dài được cắt thành các phần ~6 phút để dịch chính xác hơn
SESSION_KEYS = ("segments", "result", "video_path", "total", "used_models", "segment_editor")

# code -> (tên tiếng Việt, tên tiếng Anh dùng trong prompt)
LANGUAGES = {
    "vi": ("Tiếng Việt", "Vietnamese"),
    "zh": ("Tiếng Trung", "Chinese"),
    "en": ("Tiếng Anh", "English"),
    "ja": ("Tiếng Nhật", "Japanese"),
    "ko": ("Tiếng Hàn", "Korean"),
    "th": ("Tiếng Thái", "Thai"),
    "id": ("Tiếng Indonesia", "Indonesian"),
    "ms": ("Tiếng Mã Lai", "Malay"),
    "fil": ("Tiếng Philippines", "Filipino"),
    "km": ("Tiếng Khmer", "Khmer"),
    "lo": ("Tiếng Lào", "Lao"),
    "my": ("Tiếng Myanmar", "Burmese"),
    "hi": ("Tiếng Hindi", "Hindi"),
    "bn": ("Tiếng Bengal", "Bengali"),
    "ta": ("Tiếng Tamil", "Tamil"),
    "ur": ("Tiếng Urdu", "Urdu"),
    "ar": ("Tiếng Ả Rập", "Arabic"),
    "fa": ("Tiếng Ba Tư", "Persian"),
    "he": ("Tiếng Do Thái", "Hebrew"),
    "tr": ("Tiếng Thổ Nhĩ Kỳ", "Turkish"),
    "ru": ("Tiếng Nga", "Russian"),
    "uk": ("Tiếng Ukraina", "Ukrainian"),
    "pl": ("Tiếng Ba Lan", "Polish"),
    "cs": ("Tiếng Séc", "Czech"),
    "de": ("Tiếng Đức", "German"),
    "fr": ("Tiếng Pháp", "French"),
    "es": ("Tiếng Tây Ban Nha", "Spanish"),
    "pt": ("Tiếng Bồ Đào Nha", "Portuguese"),
    "it": ("Tiếng Ý", "Italian"),
    "nl": ("Tiếng Hà Lan", "Dutch"),
    "sv": ("Tiếng Thụy Điển", "Swedish"),
    "el": ("Tiếng Hy Lạp", "Greek"),
}

# Danh sách giọng dự phòng, chỉ dùng khi không tải được danh sách đầy đủ từ Microsoft
FALLBACK_VOICES = {
    "vi": [("vi-VN-NamMinhNeural", "Nam"), ("vi-VN-HoaiMyNeural", "Nữ")],
    "zh": [("zh-CN-YunxiNeural", "Nam"), ("zh-CN-XiaoxiaoNeural", "Nữ")],
    "en": [("en-US-GuyNeural", "Nam"), ("en-US-AriaNeural", "Nữ")],
    "ja": [("ja-JP-KeitaNeural", "Nam"), ("ja-JP-NanamiNeural", "Nữ")],
    "ko": [("ko-KR-InJoonNeural", "Nam"), ("ko-KR-SunHiNeural", "Nữ")],
    "th": [("th-TH-NiwatNeural", "Nam"), ("th-TH-PremwadeeNeural", "Nữ")],
    "id": [("id-ID-ArdiNeural", "Nam"), ("id-ID-GadisNeural", "Nữ")],
    "ms": [("ms-MY-OsmanNeural", "Nam"), ("ms-MY-YasminNeural", "Nữ")],
    "hi": [("hi-IN-MadhurNeural", "Nam"), ("hi-IN-SwaraNeural", "Nữ")],
    "ar": [("ar-SA-HamedNeural", "Nam"), ("ar-SA-ZariyahNeural", "Nữ")],
    "tr": [("tr-TR-AhmetNeural", "Nam"), ("tr-TR-EmelNeural", "Nữ")],
    "ru": [("ru-RU-DmitryNeural", "Nam"), ("ru-RU-SvetlanaNeural", "Nữ")],
    "de": [("de-DE-ConradNeural", "Nam"), ("de-DE-KatjaNeural", "Nữ")],
    "fr": [("fr-FR-HenriNeural", "Nam"), ("fr-FR-DeniseNeural", "Nữ")],
    "es": [("es-ES-AlvaroNeural", "Nam"), ("es-ES-ElviraNeural", "Nữ")],
    "pt": [("pt-BR-AntonioNeural", "Nam"), ("pt-BR-FranciscaNeural", "Nữ")],
    "it": [("it-IT-DiegoNeural", "Nam"), ("it-IT-ElsaNeural", "Nữ")],
}

SAMPLE_TEXT = {
    "vi": "Xin chào, đây là giọng đọc lồng tiếng cho video của bạn.",
    "zh": "你好，这是为你的视频配音的声音。",
    "en": "Hello, this is the voice that will dub your video.",
    "ja": "こんにちは、これはあなたの動画に吹き替えをする声です。",
    "ko": "안녕하세요, 이것은 당신의 영상을 더빙할 목소리입니다.",
    "th": "สวัสดีครับ นี่คือเสียงพากย์สำหรับวิดีโอของคุณ",
    "id": "Halo, ini adalah suara pengisi suara untuk video Anda.",
    "fr": "Bonjour, voici la voix qui doublera votre vidéo.",
    "de": "Hallo, das ist die Stimme, die Ihr Video synchronisiert.",
    "es": "Hola, esta es la voz que doblará tu video.",
    "pt": "Olá, esta é a voz que vai dublar o seu vídeo.",
    "ru": "Здравствуйте, это голос, который озвучит ваше видео.",
}

GTTS_LANG = {"zh": "zh-CN", "fil": "tl", "he": "iw"}  # còn lại dùng nguyên mã

# ---- Model Gemini: không gắn cứng tên (Google đổi model rất nhanh) ----
AUTO_MODEL = "🤖 Tự động (Flash ổn định mới nhất)"
MANUAL_MODEL = "✏️ Nhập tên model khác..."
PRESET_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"]
_MODEL_RE = re.compile(r"^gemini-(\d+(?:\.\d+)?)-(flash-lite|flash|pro)(?:-(.+))?$")
_MODEL_EXCLUDE = (
    "tts", "image", "live", "embedding", "transcribe", "native-audio", "robotics",
    "computer-use", "omni", "customtools", "nano-banana",
)
_TIER_RANK = {"flash": 0, "flash-lite": 1, "pro": 2}
RETRY_CODES = {429, 500, 502, 503, 504}
SWITCH_CODES = {403, 404, 429, 500, 502, 503, 504}

PROMPT = """You are a professional audio-visual translator and dubbing script writer.
Listen to the audio. {src_rule} Translate ALL speech into natural spoken {dst}.

Rules:
1. Split the speech into dialogue segments that follow the real timing of the audio. Each segment is one sentence
   or one complete thought, roughly 2-12 seconds long.
2. "start" and "end" are seconds from the beginning of THIS audio file, as plain numbers (for example 12.5).
3. Translate meaning, tone and register naturally, not word by word. Keep proper names consistent.
4. The translation will be spoken aloud in the same time slot, so keep it concise: the {dst} text should take about
   as long to say as the original speech.
5. Skip music, sound effects and silence. Never invent speech. If there is no speech at all, return [].
6. "speaker" is the apparent gender of the speaker: "male" or "female".
{glossary}{context}
Return ONLY a JSON array, with no commentary, in exactly this form:
[{{"start": 0.0, "end": 3.2, "speaker": "male", "text": "..."}}]
"""


# ----------------------------------------------------------------------------
# FFmpeg / FFprobe
# ----------------------------------------------------------------------------
def run_cmd(cmd):
    """Chạy lệnh và báo rõ nguyên nhân khi lỗi (không nuốt stderr)."""
    if cmd[0] == "ffmpeg":
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *cmd[1:]]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="ignore").strip()[-1500:]
        raise RuntimeError(f"Lệnh {cmd[0]} bị lỗi:\n{detail}")
    return proc.stdout


def get_duration(path):
    out = run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=nw=1:nk=1", path])
    return float(out.decode().strip())


def has_audio_stream(path):
    out = run_cmd(["ffprobe", "-v", "error", "-select_streams", "a",
                   "-show_entries", "stream=index", "-of", "csv=p=0", path])
    return bool(out.strip())


def get_video_codec(path):
    out = run_cmd(["ffprobe", "-v", "error", "-select_streams", "v:0",
                   "-show_entries", "stream=codec_name", "-of", "csv=p=0", path])
    return out.decode().strip()


def extract_audio(video_path, audio_path):
    """Tách âm thanh mono 16 kHz: nhẹ nên upload và xử lý nhanh."""
    if not has_audio_stream(video_path):
        raise RuntimeError("Video này không có âm thanh để dịch.")
    run_cmd(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "libmp3lame", "-b:a", "64k", audio_path])


def cut_audio(src, start, duration, out):
    run_cmd(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", src,
             "-c:a", "libmp3lame", "-b:a", "64k", "-ac", "1", "-ar", "16000", out])


def silence_midpoints(audio_path):
    """Các điểm im lặng (giữa 2 câu) để cắt video dài mà không cắt ngang lời thoại."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", audio_path,
         "-af", "silencedetect=noise=-35dB:d=0.35", "-f", "null", "-"],
        capture_output=True,
    )
    log = proc.stderr.decode("utf-8", errors="ignore")
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?[\d.]+)", log)]
    return [(s + e) / 2 for s, e in zip(starts, ends)]


def plan_chunks(total, mids, chunk=CHUNK_SECONDS):
    """Chia thời lượng thành các đoạn ~chunk giây, ưu tiên cắt ở chỗ im lặng."""
    if total <= chunk * 1.4:
        return [(0.0, total)]
    cuts = []
    k = 1
    while k * chunk < total - chunk * 0.4:
        target = k * chunk
        floor = (cuts[-1] if cuts else 0.0) + 30
        near = [m for m in mids if abs(m - target) <= 45 and m > floor]
        cuts.append(min(near, key=lambda m: abs(m - target)) if near else float(target))
        k += 1
    bounds = [0.0, *cuts, total]
    return list(zip(bounds[:-1], bounds[1:]))


def decode_pcm(path, speed=1.0):
    """Giải mã mp3 -> PCM int16 mono, có thể tăng tốc mà không đổi cao độ."""
    cmd = ["ffmpeg", "-i", path]
    if abs(speed - 1.0) > 0.01:
        cmd += ["-filter:a", f"atempo={speed:.3f}"]
    cmd += ["-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    return np.frombuffer(run_cmd(cmd), dtype=np.int16)


# ----------------------------------------------------------------------------
# Giải thích lỗi bằng tiếng Việt
# ----------------------------------------------------------------------------
def explain_error(e):
    code = getattr(e, "code", None)
    text = str(e)
    low = text.lower()
    if code == 400 and ("api key" in low or "api_key" in low):
        return "API key không hợp lệ. Hãy kiểm tra lại key (lấy tại aistudio.google.com/apikey)."
    if code in (401, 403):
        return ("API key bị từ chối hoặc không có quyền dùng model này. Kiểm tra key, "
                "khu vực được hỗ trợ và gói thanh toán, hoặc chọn model khác.")
    if code == 404:
        return "Model không tồn tại hoặc Google đã ngừng. Hãy chọn model khác ở mục 'Model Gemini'."
    if code == 429:
        return ("Đã hết hạn mức (quota) hoặc gửi quá nhanh. Đợi 1-2 phút rồi thử lại, chọn model nhẹ hơn "
                "(Flash-Lite), hoặc bật thanh toán cho API key.")
    if code in (500, 502, 503, 504):
        return "Máy chủ Gemini đang quá tải. Hãy thử lại sau ít phút."
    if code == 400:
        return f"Gemini từ chối yêu cầu: {text[:400]}"
    return f"{type(e).__name__}: {text[:600]}"


def show_error(e):
    st.error(f"❌ {explain_error(e)}")
    with st.expander("Chi tiết kỹ thuật (gửi cho người hỗ trợ nếu cần)"):
        st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))


# ----------------------------------------------------------------------------
# Chọn model Gemini
# ----------------------------------------------------------------------------
def rank_models(names):
    """Lọc model dịch được (flash/pro) và trả về [(tên, phiên bản, tier, ổn định?)]."""
    found = {}
    for raw in names:
        name = raw.split("/")[-1]
        if any(x in name for x in _MODEL_EXCLUDE):
            continue
        m = _MODEL_RE.match(name)
        if not m:
            continue
        suffix = m.group(3) or ""
        unstable = 1 if ("preview" in suffix or "exp" in suffix) else 0
        found[name] = (name, float(m.group(1)), m.group(2), unstable)
    return list(found.values())


def display_models(items):
    """Thứ tự hiển thị: bản mới nhất trước."""
    return [i[0] for i in sorted(items, key=lambda i: (-i[1], _TIER_RANK[i[2]], i[3]))]


def auto_pool(items):
    """Thứ tự ưu tiên khi tự chọn: Flash ổn định mới nhất -> Flash-Lite -> bản preview -> Pro."""
    ordered = sorted(items, key=lambda i: (i[3], _TIER_RANK[i[2]], -i[1]))
    return [i[0] for i in ordered]


def model_chain(choice, items):
    """Danh sách model sẽ thử lần lượt (model chính rồi tới các model dự phòng)."""
    pool = auto_pool(items) or list(PRESET_MODELS)
    if choice in (None, AUTO_MODEL):
        return pool[:4]
    return [choice, *[m for m in pool if m != choice][:3]]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_model_names(api_key):
    client = genai.Client(api_key=api_key)
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None)
        if actions and "generateContent" not in actions:
            continue
        names.append(m.name)
    return names


# ----------------------------------------------------------------------------
# Giọng đọc (Edge-TTS)
# ----------------------------------------------------------------------------
def build_voice_db(raw_voices):
    db = {}
    for v in raw_voices:
        short = v.get("ShortName") or ""
        lang = short.split("-")[0].lower()
        if lang not in LANGUAGES:
            continue
        gender = "Nam" if str(v.get("Gender", "")).lower() == "male" else "Nữ"
        locale = v.get("Locale") or "-".join(short.split("-")[:2])
        db.setdefault(lang, []).append({"name": short, "gender": gender, "locale": locale})
    for lang in db:
        db[lang].sort(key=lambda x: (x["locale"], x["gender"], x["name"]))
    return db


def fallback_voice_db():
    return {
        lang: [{"name": n, "gender": g, "locale": "-".join(n.split("-")[:2])} for n, g in voices]
        for lang, voices in FALLBACK_VOICES.items()
    }


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_edge_voices():
    return asyncio.run(edge_tts.list_voices())


def load_voice_db():
    try:
        db = build_voice_db(fetch_edge_voices())
        if db:
            return db, True
    except Exception:  # noqa: BLE001
        pass
    return fallback_voice_db(), False


def voice_label(v):
    return f"{v['name'].replace('Neural', '')} · {v['gender']} · {v['locale']}"


def assign_voices(segments, by_gender, default_voice, male_voice, female_voice):
    voices, last = [], "Nam"
    for s in segments:
        g = s.get("speaker") or last
        last = g
        voices.append((male_voice if g == "Nam" else female_voice) if by_gender else default_voice)
    return voices


async def _tts_one(text, voice, rate, pitch, path, sem, gtts_lang, stats, on_done):
    async with sem:
        err = None
        try:
            for attempt in range(3):
                try:
                    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(path)
                    if os.path.getsize(path) > 0:
                        return True
                    raise RuntimeError("File âm thanh rỗng")
                except Exception as e:  # noqa: BLE001
                    err = e
                    await asyncio.sleep(1.5 * (attempt + 1))
            if gTTS is not None and gtts_lang:  # dự phòng: giọng Google Translate
                try:
                    await asyncio.to_thread(lambda: gTTS(text=text, lang=gtts_lang).save(path))
                    stats["fallback"] += 1
                    return True
                except Exception as e:  # noqa: BLE001
                    err = e
            stats["errors"].append(f"{type(err).__name__}: {err}")
            return False
        finally:
            if on_done:
                on_done()


async def _tts_all(segments, voices, rate, pitch, workdir, gtts_lang, stats, on_done):
    sem = asyncio.Semaphore(4)
    paths = [os.path.join(workdir, f"seg_{i:04d}.mp3") for i in range(len(segments))]
    results = await asyncio.gather(*(
        _tts_one(s["text"], v, rate, pitch, p, sem, gtts_lang, stats, on_done)
        for s, v, p in zip(segments, voices, paths)
    ))
    return [p if ok else None for p, ok in zip(paths, results)]


def synthesize_segments(segments, voices, rate_pct, pitch_hz, workdir, lang, on_done=None):
    stats = {"fallback": 0, "errors": []}
    rate, pitch = f"{int(rate_pct):+d}%", f"{int(pitch_hz):+d}Hz"
    paths = asyncio.run(_tts_all(segments, voices, rate, pitch, workdir,
                                 GTTS_LANG.get(lang, lang), stats, on_done))
    if not any(paths):
        raise RuntimeError("Không tạo được giọng đọc nào. Lỗi đầu tiên: "
                           + (stats["errors"][0] if stats["errors"] else "không rõ"))
    stats["failed"] = sum(1 for p in paths if p is None)
    return paths, stats


def preview_voice(voice, text, rate_pct, pitch_hz):
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        comm = edge_tts.Communicate(text, voice, rate=f"{int(rate_pct):+d}%", pitch=f"{int(pitch_hz):+d}Hz")
        asyncio.run(comm.save(path))
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)


# ----------------------------------------------------------------------------
# Gemini: nghe + dịch có mốc thời gian
# ----------------------------------------------------------------------------
def parse_time(value):
    """Nhận 12.5, "12.5", "01:23", "01:23.5", "1:02:03" -> số giây."""
    if isinstance(value, (int, float)):
        return float(value)
    seconds = 0.0
    for part in str(value).strip().replace(",", ".").split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def _norm_speaker(v):
    if not isinstance(v, str):
        return None
    s = v.strip().lower()
    if s in ("male", "m", "man", "nam"):
        return "Nam"
    if s in ("female", "f", "woman", "nữ", "nu"):
        return "Nữ"
    return None


def normalize_segments(items, total):
    """Làm sạch danh sách đoạn thoại (từ Gemini hoặc từ bảng người dùng sửa)."""
    segments = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not re.search(r"\w", text):
            continue  # bỏ đoạn trống / chỉ có dấu câu (Edge-TTS sẽ báo lỗi)
        try:
            start = parse_time(item["start"])
            end = parse_time(item.get("end", start + 3))
        except (KeyError, TypeError, ValueError):
            continue
        if math.isnan(start) or math.isnan(end) or start >= total:
            continue
        start = max(start, 0.0)
        end = min(max(end, start + 0.5), total)
        segments.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "speaker": _norm_speaker(item.get("speaker")),
            "text": text.strip(),
        })
    segments.sort(key=lambda s: s["start"])
    return segments


def parse_segments(raw, total):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # JSON bị cắt giữa chừng: vớt lại các đối tượng còn nguyên vẹn
        data = []
        for blob in re.findall(r"\{[^{}]*\}", raw):
            try:
                data.append(json.loads(blob))
            except json.JSONDecodeError:
                pass
        if not data and not re.fullmatch(r"\[\s*\]", raw):
            raise ValueError("Gemini trả về dữ liệu không đọc được (không phải JSON).")
    if isinstance(data, dict):  # phòng khi bọc trong {"segments": [...]}
        lists = [v for v in data.values() if isinstance(v, list)]
        data = lists[0] if lists else [data]
    return normalize_segments(data, total)  # có thể rỗng: đoạn không có lời thoại


def build_prompt(src_code, dst_code, glossary, previous):
    if src_code == "auto":
        src_rule = "The spoken language is unknown: detect it automatically (it may be several languages)."
    else:
        src_rule = f"The spoken language is {LANGUAGES[src_code][1]}."
    gl = ""
    if glossary.strip():
        gl = "\nUse this glossary for names and terms (source = translation):\n" + glossary.strip() + "\n"
    ctx = ""
    if previous:
        ctx = ("\nFor continuity, the previous part of the video ended with these translated lines "
               "(do NOT repeat them):\n" + "\n".join(f"- {t}" for t in previous) + "\n")
    return PROMPT.format(src_rule=src_rule, dst=LANGUAGES[dst_code][1], glossary=gl, context=ctx)


def _file_state(f):
    return getattr(getattr(f, "state", None), "name", None)


def upload_and_wait(client, path):
    err = None
    for attempt in range(3):
        try:
            f = client.files.upload(file=path)
            break
        except Exception as e:  # noqa: BLE001
            err = e
            if getattr(e, "code", None) not in RETRY_CODES and getattr(e, "code", None) is not None:
                raise
            time.sleep(3 * (attempt + 1))
    else:
        raise err
    waited = 0
    while _file_state(f) == "PROCESSING" and waited < 180:  # chờ file ACTIVE mới dùng được
        time.sleep(2)
        waited += 2
        f = client.files.get(name=f.name)
    if _file_state(f) == "FAILED":
        raise RuntimeError("Gemini không xử lý được file âm thanh.")
    return f


def make_config(thinking):
    """Không truyền temperature/top_p/top_k: Google đã bỏ các tham số này ở dòng Gemini 3.x."""
    kwargs = {
        "response_mime_type": "application/json",
        "safety_settings": [
            types.SafetySetting(category=c, threshold="BLOCK_NONE")
            for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                      "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
        ],
    }
    if thinking:  # dịch thoại không cần suy luận sâu: mức "low" nhanh và rẻ hơn
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
    return types.GenerateContentConfig(**kwargs)


def _is_gemini3(model):
    m = re.match(r"gemini-(\d+)", model)
    return bool(m) and int(m.group(1)) >= 3


def translate_chunk(client, uploaded, chain, prompt, chunk_dur):
    """Thử lần lượt từng model; tự thử lại khi quá tải / hết quota / trả lời hỏng."""
    plans = []
    for m in chain:
        if _is_gemini3(m):
            plans.append((m, True))
        plans.append((m, False))
    last_err, skipped = None, set()
    for model, thinking in plans:
        if model in skipped:
            continue
        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model=model, contents=[uploaded, prompt], config=make_config(thinking))
                if not resp.text:
                    raise RuntimeError("Gemini không trả về nội dung (có thể bị bộ lọc an toàn chặn).")
                return parse_segments(resp.text, chunk_dur), model
            except Exception as e:  # noqa: BLE001
                last_err = e
                code = getattr(e, "code", None)
                if code == 401 or (code == 400 and "api key" in str(e).lower()):
                    raise
                if code in RETRY_CODES and attempt == 0:
                    time.sleep(6)
                    continue
                if code is None and attempt == 0:  # JSON hỏng / rỗng: thử lại một lần
                    continue
                if code in SWITCH_CODES:
                    skipped.add(model)  # model này hết quota / không có -> bỏ qua
                break
    raise last_err


def translate_audio(audio_path, total, api_key, chain, src, dst, glossary, workdir, progress):
    client = genai.Client(api_key=api_key)
    mids = silence_midpoints(audio_path) if total > CHUNK_SECONDS * 1.4 else []
    chunks = plan_chunks(total, mids)
    all_segs, used, previous = [], [], []
    for i, (cs, ce) in enumerate(chunks):
        progress(0.08 + 0.9 * i / len(chunks),
                 f"Gemini đang nghe & dịch phần {i + 1}/{len(chunks)}...")
        part = audio_path
        if len(chunks) > 1:
            part = os.path.join(workdir, f"chunk_{i:03d}.mp3")
            cut_audio(audio_path, cs, ce - cs, part)
        uploaded = upload_and_wait(client, part)
        try:
            prompt = build_prompt(src, dst, glossary, previous)
            segs, model = translate_chunk(client, uploaded, chain, prompt, ce - cs)
        finally:
            try:  # luôn xóa file trên server Gemini
                client.files.delete(name=uploaded.name)
            except Exception:  # noqa: BLE001
                pass
        chain = [model, *[m for m in chain if m != model]]  # lần sau dùng ngay model đã chạy được
        if model not in used:
            used.append(model)
        for s in segs:
            s["start"] = round(s["start"] + cs, 2)
            s["end"] = round(s["end"] + cs, 2)
        all_segs += segs
        previous = [s["text"] for s in segs[-4:]]
    if not all_segs:
        raise ValueError("Không nhận diện được lời thoại nào trong video "
                         "(video có thể không có lời nói, hoặc chọn sai ngôn ngữ gốc).")
    all_segs.sort(key=lambda s: s["start"])
    progress(1.0, "Đã dịch xong")
    return all_segs, used


# ----------------------------------------------------------------------------
# Ghép giọng đọc vào đúng mốc thời gian
# ----------------------------------------------------------------------------
def build_dub_track(segments, seg_paths, total, max_speed, out_wav):
    """
    Đặt từng đoạn giọng đọc vào đúng thời điểm trên một track dài bằng video.
    Đoạn đọc dài hơn khoảng trống cho phép sẽ được tăng tốc (tối đa max_speed);
    nếu vẫn dài thì đẩy lùi đoạn kế tiếp một chút để không chồng tiếng.
    """
    total_samples = int(total * SAMPLE_RATE)
    track = np.zeros(total_samples, dtype=np.int32)
    gap = int(0.08 * SAMPLE_RATE)
    cursor, sped_up, max_delay = 0, 0, 0.0

    for i, (seg, path) in enumerate(zip(segments, seg_paths)):
        if path is None:
            continue
        pcm = decode_pcm(path)
        if len(pcm) == 0:
            continue
        wanted = int(seg["start"] * SAMPLE_RATE)
        start = max(wanted, cursor)
        next_start = (int(segments[i + 1]["start"] * SAMPLE_RATE)
                      if i + 1 < len(segments) else total_samples)
        slot = next_start - start - gap
        if len(pcm) > slot:
            speed = max_speed if slot < 1 else min(len(pcm) / slot, max_speed)
            if speed > 1.02:
                pcm = decode_pcm(path, speed)
                sped_up += 1
        end = min(start + len(pcm), total_samples)
        if end > start:
            track[start:end] += pcm[: end - start]
        cursor = end + gap
        max_delay = max(max_delay, (start - wanted) / SAMPLE_RATE)

    track = np.clip(track, -32768, 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(track.tobytes())
    return {"sped_up": sped_up, "max_delay": max_delay}


def merge_audio_video(video_path, dub_wav, out_path, bg_volume=0.0):
    """Ghép giọng lồng tiếng vào video. bg_volume > 0 giữ lại tiếng gốc ở mức nhỏ."""
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", dub_wav]
    if bg_volume > 0 and has_audio_stream(video_path):
        fmt = "aformat=sample_rates=44100:channel_layouts=stereo"
        graph = (f"[0:a]volume={bg_volume:.2f},{fmt}[bg];[1:a]{fmt}[vo];"
                 "[vo][bg]amix=inputs=2:duration=first:dropout_transition=0,volume=2[a]")
        cmd += ["-filter_complex", graph, "-map", "0:v:0", "-map", "[a]"]
    else:
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    if get_video_codec(video_path) == "h264":  # copy nếu đã là H.264, nếu không mã hóa lại cho chạy được trên web
        cmd += ["-c:v", "copy"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]
    run_cmd(cmd)


# ----------------------------------------------------------------------------
# Phụ đề SRT
# ----------------------------------------------------------------------------
def _srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def make_srt(segments):
    lines = []
    for i, s in enumerate(segments, 1):
        lines += [str(i), f"{_srt_time(s['start'])} --> {_srt_time(s['end'])}", s["text"], ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Giao diện Streamlit
# ----------------------------------------------------------------------------
def reset_workspace(video_key):
    old = st.session_state.get("workdir")
    if old:
        shutil.rmtree(old, ignore_errors=True)
    for k in SESSION_KEYS:
        st.session_state.pop(k, None)
    st.session_state.workdir = tempfile.mkdtemp(prefix="dub_")
    st.session_state.video_key = video_key


def main():
    st.set_page_config(page_title="Dịch & Lồng Tiếng Video bằng Gemini AI", page_icon="🎬", layout="centered")
    st.title("🎬 Dịch & Lồng Tiếng Video đa ngôn ngữ")
    st.subheader("Gemini AI + giọng đọc AI nhiều ngôn ngữ")

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        st.error("Chưa cài FFmpeg (cần cả ffmpeg và ffprobe trong PATH). "
                 "Trên Streamlit Cloud hãy thêm file packages.txt chứa dòng: ffmpeg")
        st.stop()

    voice_db, voices_live = load_voice_db()

    # ---- Thanh bên ----
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        api_key = st.text_input("Google Gemini API Key:", type="password").strip()

        items, list_error = [], None
        if api_key:
            try:
                items = rank_models(fetch_model_names(api_key))
            except Exception as e:  # noqa: BLE001
                list_error = explain_error(e)
        if list_error:
            st.warning(f"Không lấy được danh sách model: {list_error}")
        names = display_models(items) or PRESET_MODELS
        choice = st.selectbox("Model Gemini:", [AUTO_MODEL, *names, MANUAL_MODEL],
                              help="Tự động = model Flash ổn định mới nhất trong tài khoản của bạn, "
                                   "tự chuyển sang model khác nếu gặp lỗi hoặc hết quota. "
                                   "Muốn chính xác hơn có thể chọn bản Pro.")
        if choice == MANUAL_MODEL:
            choice = st.text_input("Tên model:", value=PRESET_MODELS[0]).strip() or AUTO_MODEL

        st.subheader("🌐 Ngôn ngữ")
        src = st.selectbox("Ngôn ngữ trong video:", ["auto", *LANGUAGES],
                           format_func=lambda c: "🔎 Tự nhận diện" if c == "auto" else LANGUAGES[c][0])
        targets = [c for c in LANGUAGES if c in voice_db]
        dst = st.selectbox("Dịch & lồng tiếng sang:", targets,
                           index=targets.index("vi") if "vi" in targets else 0,
                           format_func=lambda c: LANGUAGES[c][0])

        st.subheader("🎙️ Giọng đọc")
        if not voices_live:
            st.caption("⚠️ Không tải được danh sách giọng đầy đủ, đang dùng danh sách rút gọn.")
        voices = voice_db[dst]
        label_of = {v["name"]: voice_label(v) for v in voices}
        by_gender = st.radio("Chế độ giọng:", ["Một giọng cho cả video", "Nam/Nữ theo người nói (tự nhận diện)"]) \
            .startswith("Nam/Nữ")
        if by_gender:
            males = [v["name"] for v in voices if v["gender"] == "Nam"] or [v["name"] for v in voices]
            females = [v["name"] for v in voices if v["gender"] == "Nữ"] or [v["name"] for v in voices]
            male_voice = st.selectbox("Giọng nam:", males, format_func=label_of.get)
            female_voice = st.selectbox("Giọng nữ:", females, format_func=label_of.get)
            voice = male_voice
        else:
            voice = st.selectbox("Giọng đọc:", [v["name"] for v in voices], format_func=label_of.get)
            male_voice = female_voice = voice

        rate_pct = st.slider("Tốc độ đọc cơ bản (%)", -30, 50, 0, step=5)
        pitch_hz = st.slider("Cao độ giọng (Hz)", -30, 30, 0, step=5)
        max_speed = st.slider("Tăng tốc tối đa để khớp thời lượng", 1.0, 1.8, 1.4, step=0.1,
                              help="Nếu bản dịch đọc dài hơn lời thoại gốc, giọng đọc sẽ tăng tốc tới mức này.")
        bg_pct = st.slider("Giữ tiếng gốc làm nền (%)", 0, 30, 0, step=5,
                           help="0 = thay hoàn toàn âm thanh gốc. Video có nhạc nền có thể để 10-20%.")

        if st.button("🔊 Nghe thử giọng này"):
            try:
                sample = SAMPLE_TEXT.get(dst, "Hello, this is a voice sample.")
                st.audio(preview_voice(male_voice if by_gender else voice, sample, rate_pct, pitch_hz),
                         format="audio/mp3")
                if by_gender:
                    st.audio(preview_voice(female_voice, sample, rate_pct, pitch_hz), format="audio/mp3")
            except Exception as e:  # noqa: BLE001
                st.warning(f"Không nghe thử được: {explain_error(e)}")

        glossary = st.text_area("Từ điển tên riêng (tuỳ chọn)", height=90,
                                placeholder="张伟 = Trương Vĩ\nOne Piece = Đảo Hải Tặc",
                                help="Mỗi dòng một cặp 'gốc = bản dịch' để tên riêng được dịch nhất quán.")

    uploaded = st.file_uploader("Tải lên video cần dịch (MP4, MOV, AVI, MKV)", type=["mp4", "mov", "avi", "mkv"])
    if not api_key:
        st.warning("⚠️ Vui lòng nhập Google Gemini API Key ở thanh bên trái để bắt đầu.")
        st.stop()
    if not uploaded:
        st.stop()

    # ---- Lưu video vào thư mục tạm riêng của phiên làm việc ----
    video_key = (uploaded.name, uploaded.size)
    if st.session_state.get("video_key") != video_key:
        reset_workspace(video_key)
        ext = os.path.splitext(uploaded.name)[1].lower() or ".mp4"
        video_path = os.path.join(st.session_state.workdir, f"input{ext}")
        with open(video_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.video_path = video_path
        try:
            st.session_state.total = get_duration(video_path)
        except Exception as e:  # noqa: BLE001
            show_error(e)
            st.stop()

    workdir, video_path, total = st.session_state.workdir, st.session_state.video_path, st.session_state.total
    st.video(video_path)
    st.caption(f"Thời lượng: {total:.1f} giây")

    # ---- Bước 1: nghe & dịch ----
    if st.button("1️⃣ Nghe & dịch bằng Gemini", type="primary"):
        for k in ("result", "segments", "segment_editor"):
            st.session_state.pop(k, None)
        bar = st.progress(0.0, text="Đang bắt đầu...")

        def progress(frac, msg):
            bar.progress(min(max(frac, 0.0), 1.0), text=msg)

        try:
            progress(0.02, "Đang tách âm thanh từ video...")
            audio_path = os.path.join(workdir, "source_audio.mp3")
            extract_audio(video_path, audio_path)
            chain = model_chain(choice, items)
            segs, used = translate_audio(audio_path, total, api_key, chain, src, dst,
                                         glossary, workdir, progress)
            st.session_state.segments = segs
            st.session_state.used_models = used
            bar.empty()
        except Exception as e:  # noqa: BLE001
            bar.empty()
            show_error(e)

    # ---- Xem / sửa bản dịch ----
    if st.session_state.get("segments"):
        used = ", ".join(st.session_state.get("used_models", []))
        st.success(f"Đã dịch {len(st.session_state.segments)} đoạn thoại bằng {used}. "
                   "Kiểm tra và sửa bản dịch bên dưới (bấm vào ô để sửa, thêm/xóa dòng nếu cần).")
        edited = st.data_editor(
            pd.DataFrame(st.session_state.segments),
            num_rows="dynamic",
            hide_index=True,
            column_order=["start", "end", "speaker", "text"],
            column_config={
                "start": st.column_config.NumberColumn("Bắt đầu (giây)", format="%.2f", min_value=0.0),
                "end": st.column_config.NumberColumn("Kết thúc (giây)", format="%.2f", min_value=0.0),
                "speaker": st.column_config.SelectboxColumn("Người nói", options=["Nam", "Nữ"]),
                "text": st.column_config.TextColumn("Bản dịch", width="large"),
            },
            key="segment_editor",
        )

        # ---- Bước 2: tạo giọng + ghép video ----
        if st.button("2️⃣ Tạo giọng lồng tiếng & ghép vào video", type="primary"):
            st.session_state.pop("result", None)
            segments = normalize_segments(edited.to_dict("records"), total)
            if not segments:
                st.error("Bảng bản dịch đang trống.")
            else:
                bar = st.progress(0.0, text="Đang tạo giọng đọc...")
                done = {"n": 0}

                def on_done():
                    done["n"] += 1
                    bar.progress(min(0.6 * done["n"] / len(segments), 0.6),
                                 text=f"Đang tạo giọng đọc... {done['n']}/{len(segments)}")

                try:
                    voice_list = assign_voices(segments, by_gender, voice, male_voice, female_voice)
                    seg_paths, tts_stats = synthesize_segments(
                        segments, voice_list, rate_pct, pitch_hz, workdir, dst, on_done)

                    bar.progress(0.65, text="Đang căn khớp giọng đọc với thời gian của video...")
                    dub_wav = os.path.join(workdir, "dub_track.wav")
                    stats = build_dub_track(segments, seg_paths, total, max_speed, dub_wav)
                    stats.update(tts_stats)

                    bar.progress(0.8, text="Đang ghép âm thanh lồng tiếng vào video...")
                    out_path = os.path.join(workdir, "output_video.mp4")
                    merge_audio_video(video_path, dub_wav, out_path, bg_pct / 100)

                    st.session_state.result = {"path": out_path, "srt": make_srt(segments), "stats": stats}
                    bar.empty()
                except Exception as e:  # noqa: BLE001
                    bar.empty()
                    show_error(e)

    # ---- Kết quả (lưu trong session_state nên không mất khi bấm tải xuống) ----
    result = st.session_state.get("result")
    if result:
        st.subheader("🎉 Video sau khi lồng tiếng:")
        st.video(result["path"])
        stats = result["stats"]
        if stats.get("failed"):
            st.warning(f"{stats['failed']} đoạn không tạo được giọng nên bị bỏ trống. "
                       "Bấm bước 2 lại để thử lần nữa.")
        if stats.get("fallback"):
            st.info(f"{stats['fallback']} đoạn dùng giọng dự phòng (gTTS) vì Edge-TTS bị lỗi.")
        if stats["max_delay"] > 1.5:
            st.info(f"Một số đoạn dịch đọc dài hơn lời thoại gốc nên bị trễ tối đa {stats['max_delay']:.1f}s. "
                    "Hãy rút gọn các câu dài trong bảng dịch rồi bấm bước 2 lại.")
        col1, col2 = st.columns(2)
        with col1:
            with open(result["path"], "rb") as f:
                st.download_button("📥 Tải video đã lồng tiếng", f,
                                   file_name="video_long_tieng.mp4", mime="video/mp4")
        with col2:
            st.download_button("📄 Tải phụ đề (.srt)", result["srt"].encode("utf-8"),
                               file_name="phu_de.srt", mime="text/plain")


if __name__ == "__main__":
    main()
