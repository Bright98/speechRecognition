import sounddevice as sd
from scipy.io.wavfile import write


def Record(file_name):
    fs = 11025
    duration = 3  # second

    signal = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    # print("recording...")
    sd.wait()
    write("data/records/" + file_name + ".wav", fs, signal)
    # print("recorded !")
