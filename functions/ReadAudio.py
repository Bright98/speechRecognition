# from scipy.io import wavfile
import librosa

# read audio data
def ReadAudio(folder, fileName):

    signal, fs = librosa.load("data/" + folder + "/" + str(fileName) + ".wav", sr=11025)
    return fs, signal
