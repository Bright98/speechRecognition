import numpy as np
from functions.Buffer import Buffer

#  audio preproccessing
def Frame(signal, frameSize, overlap):
    buffer = []

    Buffer(buffer, signal, frameSize, overlap)
    frame = np.array(np.transpose(buffer))

    ham = np.hamming(frameSize)
    ham = np.reshape(ham, (200, -1))

    win = np.multiply(ham, frame)
    return win
