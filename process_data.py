import numpy as np
import pandas as pd
import AbstractModel as abst
import keras
from keras.models import Sequential, load_model
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

class SymbolModel(abst.AbsModel):
    def load_train(self, train, train_labels):
        numbers = train_labels[train_labels['label'].isin(list(range(10, 13)))]
        self.train = train[train['index'].isin(numbers['index'].values)].drop(['index'], axis=1)
        self.train = self.train.as_matrix().reshape(numbers['index'].count(), 24, 24, 1)
        self.train_labels = numbers.drop(['index'], axis=1).as_matrix()
        one_hot = np.zeros([self.train_labels.shape[0], 3])
        for i, label in enumerate(self.train_labels):
            one_hot[i][label - 10] = 1
        self.train_labels = one_hot
    
    def load_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3, activation='softmax'))

        self.model.compile(loss='mean_squared_error', optimizer='adam')
    
    def load_model_from_file(self, filename):
        self.model = load_model(filename)

    def fit(self):
        self.model.fit(self.train, self.train_labels, batch_size=32, epochs=10)
        self.model.save('symbol_model.h5')

    def score(self):
        x_train, x_test, y_train, y_test = train_test_split(self.train, self.train_labels)
        self.model.fit(x_train, y_train, batch_size=32, epochs=10)
        score = self.model.evaluate(x_test, y_test, batch_size=32)
        print(score)

    def load_test(self, test):
        test_no_index = test.drop(['index'], axis=1)
        test_matrices = test_no_index.as_matrix().reshape(20000, 24, 120, 1)
        test_ims = np.array([test_matrices[:, :, 24 * i:24 * (i + 1), :] for i in [1, 3]])
        self.test = test_ims

    def predict(self):
        symbarr = ['+', '-', '=']
        symbol1 = np.array([symbarr[symb] for symb in np.argmax(self.model.predict(self.test[0, :, :, :]), axis=1)])
        symbol2 = np.array([symbarr[symb] for symb in np.argmax(self.model.predict(self.test[1, :, :, :]), axis=1)])
        
        newdf = pd.DataFrame(np.array([symbol1, symbol2]).T, columns=["symbol1", "symbol2"])
        newdf.to_csv("symbol.csv")

class NumberModel(abst.AbsModel):
    def load_train(self, train, train_labels):
        numbers = train_labels[train_labels['label'].isin(list(range(10)))]
        self.train = train[train['index'].isin(numbers['index'].values)].drop(['index'], axis=1)
        self.train = self.train.as_matrix().reshape(numbers['index'].count(), 24, 24, 1)
        self.train_labels = numbers.drop(['index'], axis=1).as_matrix()
        one_hot = np.zeros([self.train_labels.shape[0], 10])
        for i, label in enumerate(self.train_labels):
            one_hot[i][label] = 1
        self.train_labels = one_hot
    
    def load_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(24, 24, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(32, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(500, use_bias = False))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    def load_model_from_file(self, filename):
        self.model = load_model(filename)

    def fit(self):
        self.log(str(self.model.to_json()))
        self.model.fit(self.train, self.train_labels, batch_size=128, epochs=20)
        self.model.save('number_model.h5')

    def score(self):
        x_train, x_test, y_train, y_test = train_test_split(self.train, self.train_labels)
        self.model.fit(x_train, y_train, batch_size=32, epochs=10)
        score = self.model.evaluate(x_test, y_test, batch_size=32)
        print(score)

    def load_test(self, test):
        test_no_index = test.drop(['index'], axis=1)
        test_matrices = test_no_index.as_matrix().reshape(20000, 24, 120, 1)
        test_ims = np.array([test_matrices[:, :, 24 * i:24 * (i + 1), :] for i in range(0, 5, 2)])
        self.test = test_ims

    def predict(self):
        num1scores = self.model.predict(self.test[0, :, :, :])
        num2scores = self.model.predict(self.test[1, :, :, :])
        num3scores = self.model.predict(self.test[2, :, :, :])

        pd.DataFrame(num1scores).to_csv("num1scores.csv")
        pd.DataFrame(num2scores).to_csv("num2scores.csv")
        pd.DataFrame(num3scores).to_csv("num3scores.csv")

symbol_model = SymbolModel()
symbol_model.load_train(
    pd.read_csv("data/train.csv"),
    pd.read_csv("data/train_labels.csv")
    )
symbol_model.load_model()
symbol_model.fit()
symbol_model.load_test(
        pd.read_csv("data/test.csv")
        )
symbol_model.load_model_from_file('symbol_model.h5')
symbol_model.predict()

number_model = NumberModel()
number_model.load_train(
    pd.read_csv("data/train.csv"),
    pd.read_csv("data/train_labels.csv")
    )
number_model.load_model()
number_model.fit()
number_model.load_test(
        pd.read_csv("data/test.csv")
        )
number_model.load_model_from_file('number_model.h5')
number_model.predict()
