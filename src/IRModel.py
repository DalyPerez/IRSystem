from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, Dropout
import matplotlib.pyplot as plt
import numpy as np

class lstmModel:
    def __init__(self, laten_space, emmb_size):
        hidden = laten_space // 2
        inp_s1 = Input(shape = (None, emmb_size))
        inp_s2 = Input(shape = (None, emmb_size))
        
        encoder = LSTM(laten_space, dropout=0.2, recurrent_dropout=0.2)

        decoder_s1 = encoder(inp_s1)
        decoder_s2 = encoder(inp_s2)
        
        merge = Concatenate()([decoder_s1, decoder_s2])
        clasif = Dense(2, activation = 'softmax')(merge)
        self.model = Model([inp_s1, inp_s2], clasif)
        self.model.summary()

        self.model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics = ["acc"])

    def train(self, X, Y, VX, VY, n_epoch):

        history = self.model.fit_generator(GetDataAsyncContext(X, Y), steps_per_epoch=len(X), validation_data = GetDataAsyncContext(VX, VY), validation_steps = len(VX), epochs=n_epoch, verbose=1)
        self.model.save("context_model4.bin")
        plt.plot(history.history['acc'], "b")
        plt.plot(history.history['val_acc'], "g:")
        plt.title('metrics')
        plt.xlabel('epoch')
        plt.legend(['acc', 'val_acc'], loc='upper left')
        plt.savefig('context4.png')
        plt.show()