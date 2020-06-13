from functions.ReadAudio import ReadAudio
from functions.Framing import Frame
from functions.FeatureDetection import FeatureDetection


def MakeFeature(folder, fileName, label, frameSize, overlap):
    # read audio
    fs, signal = ReadAudio(folder, fileName)

    # frame signal and set hamming window
    window = Frame(signal, frameSize, overlap)

    # feature detection
    feature = FeatureDetection(signal, fs, window)
    feature.extend([label])

    return feature
