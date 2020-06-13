import numpy as np


def Buffer(buffer, signal, frame_size, overlap):
    slices = np.arange(0, len(signal), frame_size - overlap, dtype=np.int)
    for start, end in zip(slices[:-1], slices[1:]):
        start_audio = start
        end_audio = end + overlap
        buffer.append(list(signal[start_audio:end_audio]))
    return buffer
