"""

🎬 Tool Dịch & Lồng Tiếng Video Trung -> Việt (Gemini + Edge-TTS + FFmpeg)



Cài đặt:

    pip install -U streamlit google-genai edge-tts numpy pandas

    (cần cài FFmpeg, có cả `ffmpeg` và `ffprobe` trong PATH)



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

import wave



import edge_tts

import numpy as np

import pandas as pd

import streamlit as st

from google import genai

from google.genai import types



# ----------------------------------------------------------------------------

# Cấu hình

# ----------------------------------------------------------------------------

# Model Gemini có thể đổi ngay trên thanh bên. gemini-1.5-flash (model cũ trong

# code gốc) đã bị Google tắt nên gọi sẽ báo lỗi 404.

DEFAULT_MODEL = "gemini-2.5-flash"

SAMPLE_RATE = 24000  # Edge-TTS xuất mp3 24 kHz mono

VOICES = {

    "Nam Minh (Nam)": "vi-VN-NamMinhNeural",

    "Hoài My (Nữ)": "vi-VN-HoaiMyNeural",

}



PROMPT = """Bạn là biên dịch viên phim và video chuyên nghiệp.

Hãy nghe đoạn âm thanh tiếng Trung và dịch sang tiếng Việt, chia thành các đoạn thoại theo đúng thời điểm nói.



Yêu cầu:

1. Dịch tự nhiên, thoát ý, đúng văn phong nói tiếng Việt. Tên riêng dịch nhất quán trong toàn bộ video.

2. Mỗi đoạn là một câu hoặc một ý trọn vẹn, dài khoảng 3-12 giây.

3. Bản dịch mỗi đoạn phải đủ ngắn gọn để đọc kịp trong khoảng thời gian của đoạn đó

   (tiếng Việt thường dài hơn tiếng Trung nên hãy diễn đạt súc tích).

4. Bỏ qua nhạc nền, tiếng động và những chỗ không có lời thoại.

5. Chỉ trả về JSON đúng định dạng sau, không kèm giải thích:

[{"start": <giây bắt đầu, số thực>, "end": <giây kết thúc, số thực>, "text": "<bản dịch tiếng Việt>"}]

"""





# ----------------------------------------------------------------------------

# Tiện ích FFmpeg / FFprobe

# ----------------------------------------------------------------------------

def run_cmd(cmd):

    """Chạy lệnh, nếu lỗi thì báo rõ nguyên nhân (code cũ nuốt hết lỗi của ffmpeg)."""

    if cmd[0] == "ffmpeg":

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *cmd[1:]]

    proc = subprocess.run(cmd, capture_output=True)

    if proc.returncode != 0:

        detail = proc.stderr.decode("utf-8", errors="ignore").strip()[-1500:]

        raise RuntimeError(f"Lệnh {cmd[0]} bị lỗi:\n{detail}")

    return proc.stdout





def get_duration(path):

    out = run_cmd(

        ["ffprobe", "-v", "error", "-show_entries", "format=duration",

         "-of", "default=nw=1:nk=1", path]

    )

    return float(out.decode().strip())





def has_audio_stream(path):

    out = run_cmd(

        ["ffprobe", "-v", "error", "-select_streams", "a",

         "-show_entries", "stream=index", "-of", "csv=p=0", path]

    )

    return bool(out.strip())





def get_video_codec(path):

    out = run_cmd(

        ["ffprobe", "-v", "error", "-select_streams", "v:0",

         "-show_entries", "stream=codec_name", "-of", "csv=p=0", path]

    )

    return out.decode().strip()





def extract_audio(video_path, audio_path):

    """Tách âm thanh (mono, 16 kHz) - nhẹ hơn nhiều nên upload & xử lý nhanh hơn."""

    if not has_audio_stream(video_path):

        raise RuntimeError("Video này không có âm thanh để dịch.")

    run_cmd(

        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",

         "-c:a", "libmp3lame", "-b:a", "64k", audio_path]

    )





def decode_pcm(path, speed=1.0):

    """Giải mã mp3 -> mảng PCM int16 mono, có thể tăng tốc (giữ nguyên cao độ)."""

    cmd = ["ffmpeg", "-i", path]

    if abs(speed - 1.0) > 0.01:

        cmd += ["-filter:a", f"atempo={speed:.3f}"]

    cmd += ["-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]

    return np.frombuffer(run_cmd(cmd), dtype=np.int16)





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





def normalize_segments(items, total):

    """Làm sạch danh sách đoạn thoại (từ Gemini hoặc từ bảng người dùng sửa)."""

    segments = []

    for item in items:

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

        segments.append({"start": round(start, 2), "end": round(end, 2), "text": text.strip()})

    segments.sort(key=lambda s: s["start"])

    return segments





def parse_segments(raw, total):

    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    data = json.loads(raw)

    if isinstance(data, dict):  # phòng khi Gemini bọc trong {"segments": [...]}

        data = next((v for v in data.values() if isinstance(v, list)), [])

    segments = normalize_segments(data, total)

    if not segments:

        raise ValueError("Gemini không trả về đoạn thoại nào (video có thể không có lời nói).")

    return segments





def _file_state(f):

    return getattr(getattr(f, "state", None), "name", None)





def gemini_translate(audio_path, api_key, model):

    client = genai.Client(api_key=api_key)

    uploaded = client.files.upload(file=audio_path)

    try:

        # File âm thanh cần thời gian xử lý trên server, phải chờ ACTIVE mới dùng được

        waited = 0

        while _file_state(uploaded) == "PROCESSING" and waited < 120:

            time.sleep(2)

            waited += 2

            uploaded = client.files.get(name=uploaded.name)

        if _file_state(uploaded) == "FAILED":

            raise RuntimeError("Gemini không xử lý được file âm thanh.")



        response = client.models.generate_content(

            model=model,

            contents=[uploaded, PROMPT],

            config=types.GenerateContentConfig(

                temperature=0.3,

                response_mime_type="application/json",

            ),

        )

        if not response.text:

            raise RuntimeError("Gemini không trả về nội dung (có thể bị bộ lọc an toàn chặn).")

        return response.text

    finally:

        try:  # luôn xóa file trên server Gemini, kể cả khi lỗi

            client.files.delete(name=uploaded.name)

        except Exception:

            pass





# ----------------------------------------------------------------------------

# Edge-TTS: đọc từng đoạn (song song, có thử lại khi lỗi mạng)

# ----------------------------------------------------------------------------

async def _tts_one(text, voice, rate, path, sem, retries=3):

    async with sem:

        for attempt in range(retries):

            try:

                await edge_tts.Communicate(text, voice, rate=rate).save(path)

                return

            except Exception:

                if attempt == retries - 1:

                    raise

                await asyncio.sleep(1.5 * (attempt + 1))





async def _tts_all(segments, voice, rate, workdir):

    sem = asyncio.Semaphore(4)

    paths = [os.path.join(workdir, f"seg_{i:04d}.mp3") for i in range(len(segments))]

    await asyncio.gather(

        *(_tts_one(s["text"], voice, rate, p, sem) for s, p in zip(segments, paths))

    )

    return paths





def synthesize_segments(segments, voice, rate_pct, workdir):

    return asyncio.run(_tts_all(segments, voice, f"{rate_pct:+d}%", workdir))





# ----------------------------------------------------------------------------

# Ghép các đoạn thoại vào đúng mốc thời gian

# ----------------------------------------------------------------------------

def build_dub_track(segments, seg_paths, total, max_speed, out_wav):

    """

    Đặt từng đoạn giọng đọc vào đúng thời điểm trên một track dài bằng video.

    Đoạn nào đọc dài hơn khoảng trống cho phép sẽ được tăng tốc (tối đa max_speed);

    nếu vẫn dài thì đẩy lùi đoạn kế tiếp một chút để không chồng tiếng.

    """

    total_samples = int(total * SAMPLE_RATE)

    track = np.zeros(total_samples, dtype=np.int32)

    gap = int(0.08 * SAMPLE_RATE)

    cursor = 0

    sped_up = 0

    max_delay = 0.0



    for i, (seg, path) in enumerate(zip(segments, seg_paths)):

        pcm = decode_pcm(path)

        if len(pcm) == 0:

            continue

        wanted = int(seg["start"] * SAMPLE_RATE)

        start = max(wanted, cursor)

        next_start = (

            int(segments[i + 1]["start"] * SAMPLE_RATE)

            if i + 1 < len(segments)

            else total_samples

        )

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

    """Ghép giọng lồng tiếng vào video. bg_volume > 0 sẽ giữ lại tiếng gốc ở mức nhỏ."""

    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", dub_wav]

    if bg_volume > 0 and has_audio_stream(video_path):

        fmt = "aformat=sample_rates=44100:channel_layouts=stereo"

        graph = (

            f"[0:a]volume={bg_volume:.2f},{fmt}[bg];"

            f"[1:a]{fmt}[vo];"

            "[vo][bg]amix=inputs=2:duration=first:dropout_transition=0,volume=2[a]"

        )

        cmd += ["-filter_complex", graph, "-map", "0:v:0", "-map", "[a]"]

    else:

        cmd += ["-map", "0:v:0", "-map", "1:a:0"]



    # Copy video nếu đã là H.264, ngược lại mã hóa lại để file mp4 chạy được trên trình duyệt

    if get_video_codec(video_path) == "h264":

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

    st.session_state.clear()

    st.session_state.workdir = tempfile.mkdtemp(prefix="dub_")

    st.session_state.video_key = video_key





def main():

    st.set_page_config(

        page_title="Lồng Tiếng Video Trung - Việt bằng Gemini AI",

        page_icon="🎬",

        layout="centered",

    )

    st.title("🎬 Dịch & Lồng Tiếng Video Trung -> Việt")

    st.subheader("Gemini AI + Giọng đọc AI tiếng Việt")



    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):

        st.error("Chưa cài FFmpeg (cần cả ffmpeg và ffprobe trong PATH). Hãy cài rồi chạy lại.")

        st.stop()



    # ---- Thanh bên ----

    with st.sidebar:

        api_key = st.text_input(

            "Google Gemini API Key:",

            type="password",

            value=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""),

        )

        model = st.text_input("Model Gemini:", value=DEFAULT_MODEL)

        voice_label = st.selectbox("Giọng đọc lồng tiếng:", list(VOICES))

        rate_pct = st.slider("Tốc độ đọc cơ bản (%)", -30, 50, 0, step=5)

        max_speed = st.slider(

            "Tăng tốc tối đa để khớp thời lượng", 1.0, 1.8, 1.4, step=0.1,

            help="Nếu bản dịch đọc dài hơn lời thoại gốc, giọng đọc sẽ được tăng tốc tới mức này.",

        )

        bg_pct = st.slider(

            "Giữ tiếng gốc làm nền (%)", 0, 30, 0, step=5,

            help="0 = thay hoàn toàn âm thanh gốc. Nếu video có nhạc nền, có thể để 10-20%.",

        )

    voice = VOICES[voice_label]



    uploaded = st.file_uploader(

        "Tải lên video tiếng Trung (MP4, MOV, AVI, MKV)", type=["mp4", "mov", "avi", "mkv"]

    )

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

        except Exception as e:

            st.error(f"Không đọc được video: {e}")

            st.stop()



    workdir = st.session_state.workdir

    video_path = st.session_state.video_path

    total = st.session_state.total

    st.video(video_path)

    st.caption(f"Thời lượng: {total:.1f} giây")



    # ---- Bước 1: nghe & dịch ----

    if st.button("1️⃣ Nghe & dịch bằng Gemini", type="primary"):

        st.session_state.pop("result", None)

        st.session_state.pop("segments", None)

        with st.status("Đang xử lý...", expanded=True) as status:

            try:

                st.write("⏳ Đang tách âm thanh từ video...")

                audio_path = os.path.join(workdir, "chinese_audio.mp3")

                extract_audio(video_path, audio_path)



                st.write("⏳ Gemini đang nghe và dịch sang tiếng Việt...")

                raw = gemini_translate(audio_path, api_key, model)

                st.session_state.segments = parse_segments(raw, total)

                status.update(label="Đã dịch xong! Kiểm tra bản dịch bên dưới.", state="complete")

            except Exception as e:

                status.update(label="Có lỗi xảy ra", state="error")

                st.error(f"Đã xảy ra lỗi: {e}")



    # ---- Xem / sửa bản dịch ----

    if st.session_state.get("segments"):

        st.markdown("### 📝 Bản dịch (bấm vào ô để sửa, thêm/xóa dòng nếu cần)")

        edited = st.data_editor(

            pd.DataFrame(st.session_state.segments),

            num_rows="dynamic",

            hide_index=True,

            column_config={

                "start": st.column_config.NumberColumn("Bắt đầu (giây)", format="%.2f", min_value=0.0),

                "end": st.column_config.NumberColumn("Kết thúc (giây)", format="%.2f", min_value=0.0),

                "text": st.column_config.TextColumn("Bản dịch tiếng Việt", width="large"),

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

                with st.status("Đang xử lý...", expanded=True) as status:

                    try:

                        st.write(f"⏳ Đang tạo giọng đọc cho {len(segments)} đoạn thoại...")

                        seg_paths = synthesize_segments(segments, voice, rate_pct, workdir)



                        st.write("⏳ Đang căn khớp giọng đọc với thời gian của video...")

                        dub_wav = os.path.join(workdir, "dub_track.wav")

                        stats = build_dub_track(segments, seg_paths, total, max_speed, dub_wav)



                        st.write("⏳ Đang ghép âm thanh lồng tiếng vào video...")

                        out_path = os.path.join(workdir, "output_vietnamese_video.mp4")

                        merge_audio_video(video_path, dub_wav, out_path, bg_pct / 100)



                        st.session_state.result = {

                            "path": out_path,

                            "srt": make_srt(segments),

                            "stats": stats,

                        }

                        status.update(label="🎉 Hoàn tất lồng tiếng!", state="complete")

                    except Exception as e:

                        status.update(label="Có lỗi xảy ra", state="error")

                        st.error(f"Đã xảy ra lỗi: {e}")



    # ---- Kết quả (lưu trong session_state nên không mất khi bấm tải xuống) ----

    result = st.session_state.get("result")

    if result:

        st.subheader("Video sau khi lồng tiếng Việt:")

        st.video(result["path"])



        stats = result["stats"]

        if stats["max_delay"] > 1.5:

            st.info(

                f"Một số đoạn dịch đọc dài hơn lời thoại gốc nên bị trễ tối đa {stats['max_delay']:.1f}s. "

                "Hãy rút gọn các câu dài trong bảng dịch rồi bấm bước 2 lại."

            )



        col1, col2 = st.columns(2)

        with col1:

            with open(result["path"], "rb") as f:

                st.download_button(

                    "📥 Tải video đã lồng tiếng", f, file_name="video_tieng_viet.mp4", mime="video/mp4"

                )

        with col2:

            st.download_button(

                "📄 Tải phụ đề (.srt)", result["srt"].encode("utf-8"),

                file_name="phu_de_tieng_viet.srt", mime="text/plain",

            )





if __name__ == "__main__":

    main()