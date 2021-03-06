import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras import models, layers


class Network:
    dataset = None

    def __init__(self, dataset):
        self.dataset = dataset

    @staticmethod
    def graph_loss(epoch_count, training_loss, test_loss):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(epoch_count, training_loss, 'r--')
        ax.plot(epoch_count, test_loss, 'b-')
        plt.legend(['Потеря на тренировке', 'Потеря на тестировании'])
        plt.xlabel('Эпоха')
        plt.ylabel('Потеря')
        plt.show()

    @staticmethod
    def graph_accuracy(epoch_count, training_accuracy, test_acccuracy):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(epoch_count, training_accuracy, 'r--')
        ax.plot(epoch_count, test_acccuracy, 'b-')
        plt.legend(['Точность на тренировке', 'Точность на тестировании'])
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.show()

    def second_network(self, copy=False):
        if copy:
            dataset = self.dataset.copy()
        else:
            dataset = self.dataset
        std_scaler = StandardScaler()
        robust_scaler = preprocessing.RobustScaler()
        np.random.seed(0)
        dataset = dataset.dropna()
        dataset = dataset[dataset['p'] < 5]
        target = np.array(dataset['v'])
        dataset = dataset.drop('v', axis=1)
        target = robust_scaler.fit_transform(target.reshape(-1, 1))
        features_std = std_scaler.fit_transform(dataset)
        joblib.dump(robust_scaler, 'Models/rob_scaler_sec.pkl')
        joblib.dump(std_scaler, 'Models/std_scaler_sec.pkl')
        features_train, features_test, target_train, target_test = \
            train_test_split(features_std, target, random_state=1, train_size=0.8)
        checkpoint = [ModelCheckpoint(filepath='Models/model_sec.h5', save_best_only=True, monitor='val_loss')]
        mynetwork = models.Sequential()
        mynetwork.add(layers.Dense(
            units=64,
            activation='relu',
            input_shape=(features_train.shape[1],)))
        mynetwork.add(layers.Dense(
            units=32,
            activation='relu',
            input_shape=(features_train.shape[1],)))
        mynetwork.add(layers.Dense(units=1))
        mynetwork.compile(loss='mse',
                          optimizer='RMSprop',
                          metrics=['mse'])
        history = mynetwork.fit(
            features_train,
            target_train,
            epochs=5000,
            callbacks=checkpoint,
            verbose=0,
            batch_size=500,
            validation_data=(features_test, target_test)
        )
        training_loss = history.history['loss']
        test_loss = history.history['val_loss']
        epoch_count = range(1, len(training_loss) + 1)
        self.graph_loss(epoch_count, training_loss, test_loss)

    def first_network(self, copy=False):
        if copy:
            dataset = self.dataset.copy()
        else:
            dataset = self.dataset
        std_scaler = StandardScaler()
        np.random.seed(0)
        dataset = dataset.dropna()
        dataset = dataset[dataset['v'] < 200]
        target = np.array(dataset['v'])
        dataset = dataset.drop('v', axis=1)
        features_std = std_scaler.fit_transform(dataset)
        joblib.dump(std_scaler, 'Models/std_scaler_first.pkl')
        features_train, features_test, target_train, target_test = \
            train_test_split(features_std, target, random_state=1, train_size=0.8)
        checkpoint = [ModelCheckpoint(filepath='Models/model_first.h5', save_best_only=True, monitor='val_loss')]
        network = models.Sequential()
        network.add(layers.Dense(
            units=64,
            activation='relu',
            input_shape=(features_train.shape[1],)))
        network.add(layers.Dense(
            units=32,
            activation='relu',
            input_shape=(features_train.shape[1],)))
        network.add(layers.Dense(units=1))
        network.compile(loss='mse',
                        optimizer='RMSprop',
                        metrics=['mse'])
        history = network.fit(
            features_train,
            target_train,
            callbacks=checkpoint,
            epochs=10000,
            verbose=0,
            batch_size=200,
            validation_data=(features_test, target_test)
        )
        training_loss = history.history['loss']
        test_loss = history.history['val_loss']
        epoch_count = range(1, len(training_loss) + 1)
        self.graph_loss(epoch_count, training_loss, test_loss)
