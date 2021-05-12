from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, BatchNormalization, Activation
from keras.layers.core import Dense, Reshape, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import dot
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import regularizers

class MatchPyramidModel:
    def __init__(self, query_len, doc_len):
        num_conv2d_layers=1
        filters_2d=[16,32]
        kernel_size_2d=[[3,3], [3,3]]
        mpool_size_2d=[[2,2], [2,2]]
        dropout_rate=0.5

        query=Input(shape=(query_len,), name='query')
        doc=Input(shape=(doc_len,), name='doc')

        layer1_dot=dot([query, doc], axes=-1)
        layer1_dot=Reshape((query_len, doc_len, -1))(layer1_dot)
            
        layer1_conv=Conv2D(filters=8, kernel_size=5, padding='same')(layer1_dot)
        layer1_activation=Activation('relu')(layer1_conv)
        z=MaxPooling2D(pool_size=(2,2))(layer1_activation)
            
        for i in range(num_conv2d_layers):
            z=Conv2D(filters=filters_2d[i], kernel_size=kernel_size_2d[i], padding='same')(z)
            z=Activation('relu')(z)
            z=MaxPooling2D(pool_size=(mpool_size_2d[i][0], mpool_size_2d[i][1]))(z)
                
        pool1_flat=Flatten()(z)
        pool1_flat_drop=Dropout(rate=dropout_rate)(pool1_flat)
        mlp1=Dense(32)(pool1_flat_drop)
        mlp1=Activation('relu')(mlp1)
        out=Dense(2, activation='softmax')(mlp1)
            
        model=Model(inputs=[query, doc], outputs=out)
        model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['acc'])
        model.summary()