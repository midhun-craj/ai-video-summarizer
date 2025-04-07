from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
from moviepy.video.fx.all import fadein, fadeout
from pydub import AudioSegment
import tempfile
import whisper
import torch
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from gtts import gTTS
import nltk
nltk.download('punkt')

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)
summarizer = pipeline("summarization")

def extract_audio_from_video(video_bytes: bytes) -> tuple[AudioSegment, float]:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        temp_video.flush()
        video = VideoFileClip(temp_video.name)
        duration = video.duration
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            video.audio.write_audiofile(temp_audio.name)
            audio = AudioSegment.from_wav(temp_audio.name)
    return audio, duration

def transcribe_audio(audio: AudioSegment) -> tuple[str, list]:
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
        audio.export(temp_audio.name, format="wav")
        result = whisper_model.transcribe(temp_audio.name, word_timestamps=True)
    return result['text'], result['segments']

def summarize_text(text: str) -> str:
    sentences = sent_tokenize(text)
    chunks = [" ".join(sentences[i:i+5]) for i in range(0, len(sentences), 5)]
    summarized_text = ""
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summarized_text += summary[0]['summary_text'] + " "
    return summarized_text.strip()

def extract_key_phrases(text: str, top_k: int = 10) -> list:
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=0)).ravel()
    indices = np.argsort(scores)[-top_k:]
    features = vectorizer.get_feature_names_out()
    return [features[i] for i in indices]

def find_timestamps_for_keywords(segments, keywords: list[str]) -> list:
    timestamps = []
    for seg in segments:
        text = seg['text'].lower()
        for keyword in keywords:
            if keyword.lower() in text:
                timestamps.append((seg['start'], seg['end']))
                break
    return merge_close_segments(timestamps)

def merge_close_segments(segments: list[tuple[float, float]], threshold: float = 3.0) -> list:
    if not segments:
        return []
    segments.sort()
    merged = [list(segments[0])]
    for curr in segments[1:]:
        if curr[0] - merged[-1][1] <= threshold:
            merged[-1][1] = max(merged[-1][1], curr[1])
        else:
            merged.append(list(curr))
    return [tuple(seg) for seg in merged]

def extract_video_clips_with_narration(video_bytes: bytes, timestamps: list[tuple[float, float]], narration_chunks: list[str]) -> VideoFileClip:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        temp_video.flush()
        video = VideoFileClip(temp_video.name)
        clips = []
        audio_clips = []

        for i, (start, end) in enumerate(timestamps):
            clip = video.subclip(start, end)
            if end - start > 1.5:
                clip = fadein(clip, 0.5).fx(fadeout, 0.5)
                clips.append(clip)

                # Generate narration for this chunk
                if i < len(narration_chunks):
                    tts = gTTS(narration_chunks[i])
                    with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_audio:
                        tts.save(temp_audio.name)
                        audio_clip = AudioFileClip(temp_audio.name).set_start(0)
                        audio_clips.append(audio_clip)

        if not clips:
            return None

        final_video = concatenate_videoclips(clips, method="compose")
        if audio_clips:
            combined_audio = CompositeAudioClip(audio_clips)
            final_video = final_video.set_audio(combined_audio)

    return final_video

def split_summary_text(summary_text: str, num_chunks: int) -> list:
    sentences = sent_tokenize(summary_text)
    avg = max(1, len(sentences) // num_chunks)
    return [" ".join(sentences[i:i+avg]) for i in range(0, len(sentences), avg)]

def combine_audio_video(video_clip: VideoFileClip) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as output:
        video_clip.write_videofile(output.name, codec="libx264", audio_codec="aac")
        return output.read()

def process_video(video_bytes: bytes) -> bytes:
    audio, duration = extract_audio_from_video(video_bytes)
    transcript_text, segments = transcribe_audio(audio)
    summarized_text = summarize_text(transcript_text)
    keywords = extract_key_phrases(summarized_text)
    timestamps = find_timestamps_for_keywords(segments, keywords)
    summary_chunks = split_summary_text(summarized_text, len(timestamps))
    summarized_video_clip = extract_video_clips_with_narration(video_bytes, timestamps, summary_chunks)
    if summarized_video_clip is None:
        raise ValueError("No valid video segments found for summary.")
    final_output_video = combine_audio_video(summarized_video_clip)
    return final_output_video
