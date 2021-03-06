import sys
import os
import pandas as pd
import numpy as np
from networks import Network
import tensorflow as tf
from tensorflow.python.keras import backend as K
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtCore
from main_window import Ui_Form
from keras.models import load_model
import joblib

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def fit():
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 6})
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
    my_dataset = pd.read_excel('input.xlsx', sheet_name=0, na_values=np.nan)
    mynetwork = Network(my_dataset)
    mynetwork.first_network(copy=True)
    mynetwork.second_network(copy=True)


class Window(QWidget, Ui_Form):
    std_model_1 = None
    std_model_2 = None
    robust_model = None
    network_1 = None
    network_2 = None

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.load_models()
        self.pushButton.clicked.connect(self.predict)

    def load_models(self):
        self.std_model_1 = joblib.load('Models/std_scaler_first.pkl')
        self.std_model_2 = joblib.load('Models/std_scaler_sec.pkl')
        self.robust_model = joblib.load('Models/rob_scaler_sec.pkl')
        self.network_1 = load_model('Models/model_first.h5')
        self.network_2 = load_model('Models/model_sec.h5')

    @QtCore.pyqtSlot()
    def predict(self):
        try:
            p = float(self.lineEdit.text())
            t = float(self.lineEdit_2.text())
        except ValueError:
            return
        if p < 5:
            features = self.std_model_2.transform(np.array([[p, t]]))
            print(features)
            raw_result = self.network_2.predict(features)
            result = self.robust_model.inverse_transform(raw_result)
            self.result.setText(str(result[0][0]))
        else:
            features = self.std_model_1.transform(np.array([[p, t]]))
            print(features)
            result = self.network_1.predict(features)
            self.result.setText(str(result[0][0]))


if __name__ == '__main__':
    qapp = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(qapp.exec())
