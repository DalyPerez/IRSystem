from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, Dropout
import matplotlib.pyplot as plt
import numpy as np
from preproc import data2train

def GetDataAsyncContext(X, Y):
    while True:
        for i in range(len(X)):
            a,b = X[i]
            # print ("->", a.shape, b.shape)
            xx = [np.array([a]), np.array([b])]
            c = [0, 0]
            c[Y[i]] = 1
            yield xx, np.array([c])

class lstmModel:
    def __init__(self, laten_space, emmb_size):
        hidden = laten_space // 2
        inp_s1 = Input(shape = (None, emmb_size), name="document")
        inp_s2 = Input(shape = (None, emmb_size), name="query")
        
        decoder_s1 = LSTM(laten_space, dropout=0.3, recurrent_dropout=0.3)(inp_s1)
        decoder_s2 = LSTM(laten_space, dropout=0.3, recurrent_dropout=0.3)(inp_s2)
        
        merge = Concatenate()([decoder_s1, decoder_s2])
        clasif = Dense(2, activation = 'softmax')(merge)
        self.model = Model([inp_s1, inp_s2], clasif)
        self.model.summary()

        self.model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics = ["acc"])

    def train(self, X, Y, VX, VY, n_epoch):
        history = self.model.fit(GetDataAsyncContext(X, Y), steps_per_epoch =len(X), validation_data = GetDataAsyncContext(VX, VY) , validation_steps = len(VX), epochs=n_epoch, verbose=1)
        # self.model.save("lstmmodel.h5")
        plt.plot(history.history['acc'], "b")
        plt.plot(history.history['val_acc'], "g:")
        plt.title('metrics')
        plt.xlabel('epoch')
        plt.legend(['acc', 'val_acc'], loc='upper left')
        plt.savefig('lstmmodel.png')
        plt.show()



def TrainSimilarity(docsdict, querysdict, relpairs, w2v_dict):
    X, Y, VX, VY = data2train(docsdict, querysdict, relpairs, w2v_dict)
    print(np.shape(X[0]), np.shape(X[1]), np.shape(VX), np.shape(Y))
    print("creating model")
    model = lstmModel(64, 50)
    print("ready to train")
    model.train(X, Y, VX, VY, 5)