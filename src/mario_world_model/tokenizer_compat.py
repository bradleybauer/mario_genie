from __future__ import annotations


def resolve_video_contains_first_frame(model, seq_len: int) -> bool:
    time_downsample_factor = getattr(model, "time_downsample_factor", 1)

    if time_downsample_factor <= 1:
        return True

    if (seq_len - 1) % time_downsample_factor == 0:
        return True

    if seq_len % time_downsample_factor == 0:
        return False

    raise ValueError(
        f"sequence_length={seq_len} is incompatible with time_downsample_factor="
        f"{time_downsample_factor}: neither seq_len-1 nor seq_len is divisible by it"
    )