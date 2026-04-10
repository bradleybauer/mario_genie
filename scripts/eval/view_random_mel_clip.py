#!/usr/bin/env python3
"""Visualize the mel-spectrogram for a random 16-frame clip from data/raw.

This script picks a random AVI from data/raw, samples a random clip spanning a
fixed number of video frames, extracts only that audio segment with ffmpeg, and
plots a representative video frame, the clip waveform, and the clip
mel-spectrogram.

Usage:
    python scripts/view_random_mel_clip.py
    python scripts/view_random_mel_clip.py --seed 7
    python scripts/view_random_mel_clip.py --output /tmp/random_mel.png
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import numpy as np
from scipy import signal
from scipy.io import wavfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class ClipSelection:
    avi_path: Path
    total_frames: int
    fps: float
    start_frame: int
    clip_frames: int
    sample_rate: int
    audio_samples: np.ndarray
    preview_frame: np.ndarray

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.clip_frames - 1

    @property
    def start_time_s(self) -> float:
        return self.start_frame / self.fps

    @property
    def duration_s(self) -> float:
        return self.clip_frames / self.fps

    @property
    def end_time_s(self) -> float:
        return self.start_time_s + self.duration_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory containing raw AVI recordings (default: data/raw)",
    )
    parser.add_argument(
        "--clip-frames",
        type=int,
        default=16,
        help="Number of video frames to include in the sampled clip",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible clip selection",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Audio sample rate (default: 24000 Hz, chosen to retain 98.5%% of dataset energy)",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=400,
        help="FFT window size (default: 400, ~1 video frame at 24 kHz / 60.1 fps)",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=100,
        help="Hop size (default: 100, ~4 mel steps per video frame at 24 kHz)",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=64,
        help="Number of mel bands to visualize",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=40.0,
        help="Lowest mel frequency in Hz (default: 40, only 2.5%% of dataset energy is below this)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=8000.0,
        help="Highest mel frequency in Hz (default: 8000, captures 97.6%% of dataset energy)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the figure instead of opening a window",
    )
    return parser.parse_args()


def probe_audio_sample_rate(avi_path: Path) -> int:
    if shutil.which("ffprobe"):
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(avi_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            sample_rate_text = result.stdout.strip()
            if sample_rate_text:
                return int(sample_rate_text)

    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(avi_path)],
        capture_output=True,
        text=True,
    )
    media_info = result.stderr.splitlines()
    for line in media_info:
        if "Audio:" not in line:
            continue
        for part in (chunk.strip() for chunk in line.split(",")):
            if part.endswith("Hz"):
                return int(part[:-2].strip())

    raise RuntimeError(f"Unable to determine audio sample rate for {avi_path.name}")


def load_video_metadata(avi_path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {avi_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS reported for {avi_path.name}: {fps}")
    return fps, total_frames


def read_frame(avi_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {avi_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to read frame {frame_index} from {avi_path.name}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_audio_clip_to_wav(
    avi_path: Path,
    start_time_s: float,
    duration_s: float,
    sample_rate: int,
) -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to extract audio from AVI files")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(avi_path),
        "-ss",
        f"{start_time_s:.6f}",
        "-t",
        f"{duration_s:.6f}",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        Path(tmp.name).unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed on {avi_path.name}: {result.stderr.strip()}")
    return Path(tmp.name)


def load_audio_clip(
    avi_path: Path,
    start_time_s: float,
    duration_s: float,
    sample_rate: int,
) -> tuple[int, np.ndarray]:
    wav_path = extract_audio_clip_to_wav(
        avi_path=avi_path,
        start_time_s=start_time_s,
        duration_s=duration_s,
        sample_rate=sample_rate,
    )
    try:
        clip_sample_rate, samples = wavfile.read(wav_path)
    finally:
        wav_path.unlink(missing_ok=True)

    if samples.size == 0:
        raise RuntimeError(f"Audio clip was empty for {avi_path.name}")

    samples = np.asarray(samples, dtype=np.float32)
    max_int16 = float(np.iinfo(np.int16).max)
    if np.max(np.abs(samples)) > 0:
        samples /= max_int16
    return clip_sample_rate, samples


def hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def mel_to_hz(freq_mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (freq_mel / 2595.0) - 1.0)


def build_mel_filter_bank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    max_freq = float(sample_rate) / 2.0 if fmax is None else float(fmax)
    if fmin < 0 or max_freq <= fmin:
        raise ValueError(f"Invalid mel frequency range: fmin={fmin}, fmax={max_freq}")

    mel_edges = np.linspace(
        hz_to_mel(np.array([fmin], dtype=np.float64))[0],
        hz_to_mel(np.array([max_freq], dtype=np.float64))[0],
        n_mels + 2,
    )
    hz_edges = mel_to_hz(mel_edges)
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    filter_bank = np.zeros((n_mels, fft_freqs.shape[0]), dtype=np.float32)

    for mel_idx in range(n_mels):
        left_hz = hz_edges[mel_idx]
        center_hz = hz_edges[mel_idx + 1]
        right_hz = hz_edges[mel_idx + 2]

        left_mask = (fft_freqs >= left_hz) & (fft_freqs <= center_hz)
        right_mask = (fft_freqs >= center_hz) & (fft_freqs <= right_hz)

        if center_hz > left_hz:
            filter_bank[mel_idx, left_mask] = (
                (fft_freqs[left_mask] - left_hz) / (center_hz - left_hz)
            )
        if right_hz > center_hz:
            filter_bank[mel_idx, right_mask] = (
                (right_hz - fft_freqs[right_mask]) / (right_hz - center_hz)
            )

    return filter_bank, hz_edges[1:-1]


def compute_log_mel_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    fmax: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if samples.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {samples.shape}")
    if len(samples) < 32:
        raise ValueError("Audio clip is too short to analyze")

    effective_n_fft = min(n_fft, len(samples))
    if effective_n_fft % 2 == 1:
        effective_n_fft -= 1
    effective_n_fft = max(effective_n_fft, 32)
    effective_hop = min(hop_length, max(1, effective_n_fft // 4))
    noverlap = effective_n_fft - effective_hop

    freqs, times, stft = signal.stft(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=effective_n_fft,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    power_spec = np.abs(stft) ** 2
    filter_bank, mel_freqs = build_mel_filter_bank(
        sample_rate=sample_rate,
        n_fft=effective_n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel_spec = filter_bank @ power_spec
    log_mel_spec = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
    return log_mel_spec, times, mel_freqs


def choose_random_clip(
    raw_dir: Path,
    clip_frames: int,
    rng: random.Random,
    sample_rate_override: int | None,
) -> ClipSelection:
    avi_paths = sorted(raw_dir.glob("*.avi"))
    if not avi_paths:
        raise FileNotFoundError(f"No AVI files found in {raw_dir}")

    rng.shuffle(avi_paths)
    failures: list[str] = []

    for avi_path in avi_paths:
        try:
            fps, total_frames = load_video_metadata(avi_path)
            if total_frames < clip_frames:
                failures.append(
                    f"{avi_path.name}: only {total_frames} frames available"
                )
                continue

            start_frame = rng.randint(0, total_frames - clip_frames)
            start_time_s = start_frame / fps
            duration_s = clip_frames / fps
            sample_rate = sample_rate_override or probe_audio_sample_rate(avi_path)
            clip_sample_rate, samples = load_audio_clip(
                avi_path=avi_path,
                start_time_s=start_time_s,
                duration_s=duration_s,
                sample_rate=sample_rate,
            )
            preview_frame = read_frame(avi_path, start_frame + clip_frames // 2)
            return ClipSelection(
                avi_path=avi_path,
                total_frames=total_frames,
                fps=fps,
                start_frame=start_frame,
                clip_frames=clip_frames,
                sample_rate=clip_sample_rate,
                audio_samples=samples,
                preview_frame=preview_frame,
            )
        except Exception as exc:  # pragma: no cover - best effort selection loop
            failures.append(f"{avi_path.name}: {exc}")

    failure_summary = "\n  - ".join(failures[:10])
    raise RuntimeError(f"Unable to sample a valid AVI clip from {raw_dir}:\n  - {failure_summary}")


def plot_clip(
    clip: ClipSelection,
    log_mel_spec: np.ndarray,
    stft_times: np.ndarray,
    mel_freqs: np.ndarray,
    output_path: Path | None,
) -> None:
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.5, 1.5, 3.0]},
    )

    axes[0].imshow(clip.preview_frame)
    axes[0].set_title(
        f"{clip.avi_path.name} | frames {clip.start_frame}-{clip.end_frame} "
        f"of {clip.total_frames - 1}"
    )
    axes[0].axis("off")

    waveform_times = np.arange(len(clip.audio_samples), dtype=np.float64) / clip.sample_rate
    waveform_times += clip.start_time_s
    axes[1].plot(waveform_times, clip.audio_samples, linewidth=0.8, color="#33658a")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlim(clip.start_time_s, clip.end_time_s)
    axes[1].set_title(
        f"Waveform | start {clip.start_time_s:.3f}s | duration {clip.duration_s:.3f}s"
    )

    mel_extent = [
        clip.start_time_s + float(stft_times[0]) if stft_times.size else clip.start_time_s,
        clip.start_time_s + float(stft_times[-1]) if stft_times.size else clip.end_time_s,
        0,
        log_mel_spec.shape[0],
    ]
    image = axes[2].imshow(
        log_mel_spec,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=mel_extent,
    )
    tick_positions = np.linspace(0, log_mel_spec.shape[0] - 1, num=min(6, log_mel_spec.shape[0]))
    tick_labels = [f"{mel_freqs[int(round(idx))]:.0f}" for idx in tick_positions]
    axes[2].set_yticks(tick_positions)
    axes[2].set_yticklabels(tick_labels)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Mel frequency (Hz)")
    axes[2].set_title("Log mel-spectrogram (dB)")
    colorbar = figure.colorbar(image, ax=axes[2], pad=0.01)
    colorbar.set_label("Power (dB)")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
        print(f"Saved figure to {output_path}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    clip = choose_random_clip(
        raw_dir=args.raw_dir,
        clip_frames=args.clip_frames,
        rng=rng,
        sample_rate_override=args.sample_rate,
    )
    log_mel_spec, stft_times, mel_freqs = compute_log_mel_spectrogram(
        samples=clip.audio_samples,
        sample_rate=clip.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    print(f"Selected AVI: {clip.avi_path}")
    print(f"Clip frames: {clip.start_frame}-{clip.end_frame} / {clip.total_frames - 1}")
    print(
        f"Clip time: {clip.start_time_s:.3f}s-{clip.end_time_s:.3f}s at {clip.fps:.3f} FPS"
    )
    print(f"Audio samples: {len(clip.audio_samples)} at {clip.sample_rate} Hz")
    plot_clip(
        clip=clip,
        log_mel_spec=log_mel_spec,
        stft_times=stft_times,
        mel_freqs=mel_freqs,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()