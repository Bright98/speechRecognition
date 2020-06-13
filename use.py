from functions.RecordAudio import Record
from functions.ReadAudio import ReadAudio
from functions.Framing import Frame
from functions.FeatureDetection import FeatureDetection
from joblib import load
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
import sys
import os


class SpeechApp(QWidget):

    frameSize = 200
    overlap = 2
    fs = 11025
    duration = 3  # second

    # set file name for records
    record_filename = 1
    with os.scandir("data/records") as entries:
        for entry in entries:
            record_filename += 1

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        # title
        self.title = QLabel("speech recognition", self)
        self.title.move(400, 100)
        self.title.setStyleSheet(
            "color: #85929E ; font-size: 22px; font-family: Comic Sans MS; font-weight: 600;"
        )

        # recognition label
        self.label = QLabel("", self)
        self.label.move(412, 410)
        self.label.setStyleSheet(
            "color: #1BB0BA; font-size: 19px; font-family: Comic Sans MS; font-weight:600"
        )

        # save file label
        self.save = QLabel("", self)
        self.save.move(412, 445)
        self.save.setStyleSheet(
            "color: #1BB0BA; font-size: 18px; font-family: Comic Sans MS; font-weight:500"
        )

        # record button
        self.button = QPushButton("Record", self)
        self.button.setStyleSheet(
            "background-color: #1BB0BA ; font-weight: 600; color: #17202A; font-family: Comic Sans MS; border-radius: 10px;"
        )
        self.button.resize(150, 70)
        self.button.move(425, 500)
        self.button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))

        # audio icon
        image = QLabel(self)
        pixmap = QPixmap("icon/record.svg")
        pixmap = pixmap.scaled(160, 160)
        image.setPixmap(pixmap)
        image.move(415, 200)

        self.button.clicked.connect(self.on_click)

        self.setGeometry(500, 200, 1000, 700)
        self.setFixedSize(1000, 700)
        self.setStyleSheet("background-color: #17202A;")
        self.setWindowTitle("speech recognition")
        self.setWindowIcon(QIcon("icon/record.svg"))
        self.show()

    def on_click(self):

        # record audio
        Record(str(self.record_filename))

        self.button.setText("recorded")

        # read audio
        fs, signal = ReadAudio("records", self.record_filename)

        # Frame
        window = Frame(signal, self.frameSize, self.overlap)

        # feature detection
        recordFeature = FeatureDetection(signal, fs, window)

        # load model
        svm_model = load("model/svm_model.joblib")

        # predict new record
        recordFeature = np.array([recordFeature])
        recordFeature.reshape(1, -1)
        recordPredict = svm_model.predict(recordFeature)

        text = "you said : "
        if recordPredict.round() == 1:
            text += " sholugh "
        elif recordPredict.round() == 2:
            text += " khalvat "
        else:
            text = "error!"

        self.label.setText(text)
        self.label.adjustSize()

        self.save.setText("( file save as: " + str(self.record_filename) + ".wav )")
        self.save.adjustSize()


def main():
    app = QApplication(sys.argv)
    ex = SpeechApp()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
