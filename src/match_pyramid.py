import matplotlib.pyplot as plt
import keras
from keras.layers import Input, Embedding, BatchNormalization, Activation
from keras.layers.core import Dense, Reshape, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import dot
from keras.models import Model, load_model
from keras import regularizers

from keras.callbacks import EarlyStopping, ModelCheckpoint

from preproc import data2train_mp, doc2vector_mp
from evaluator import IREvaluator
from load_data import *
import time, random

r = random.Random()

r.seed(99)
class MatchPyramidModel:
    def __init__(self, query_len, doc_len, emb_dim):
        num_conv2d_layers=1
        filters_2d=[64]
        kernel_size_2d=[[1,3]]
        mpool_size_2d=[[3,10]]
        dropout_rate=0.5

        query=Input(shape=(query_len,emb_dim), name='query')
        doc=Input(shape=(doc_len,emb_dim), name='doc')

        layer1_dot=dot([query, doc], axes=-1)
        layer1_dot=Reshape((query_len, doc_len, -1))(layer1_dot)
            
        # layer1_conv=Conv2D(filters=8, kernel_size=5, padding='same')(layer1_dot)
        # layer1_activation=Activation('relu')(layer1_conv)
        # z=MaxPooling2D(pool_size=(2,2))(layer1_activation)
            
        # for i in range(num_conv2d_layers):
        z=Conv2D(filters=filters_2d[0], kernel_size=kernel_size_2d[0], padding='same', activation='relu')(layer1_dot)
        z=MaxPooling2D(pool_size=(mpool_size_2d[0][0], mpool_size_2d[0][1]))(z)
                
        pool1_flat=Flatten()(z)
        pool1_flat_drop=Dropout(rate=dropout_rate)(pool1_flat)
        # mlp1=Dense(32)(pool1_flat_drop)
        # mlp1=Activation('relu')(mlp1)
        out=Dense(2, activation='softmax')(pool1_flat_drop)
            
        self.model = Model(inputs=[query, doc], outputs=out)

        adam = keras.optimizers.Adam(learning_rate=0.0001)

        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
        self.model.summary()

    def train(self, X_query, X_doc, Y, XV_query, XV_doc, YV, n_epoch):
        early_stopping = EarlyStopping(monitor='val_acc', patience=2, mode = 'max')
        model_checkpoint = ModelCheckpoint('mp.{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.h5', monitor = 'val_loss', verbose=1,  mode = 'min')
        callbacks_list = [
                    model_checkpoint
                ]
        
        history = self.model.fit([X_query, X_doc], Y, validation_data=([XV_query, XV_doc], YV), epochs=n_epoch, verbose=1, callbacks = callbacks_list, batch_size= 32)
        self.model.save("mp_cisi.h5")
        plt.plot(history.history['acc'], "b")
        plt.plot(history.history['val_acc'], "g:")
        # plt.plot(history.history['mse'], "y")
        # plt.plot(history.history['val_mse'], "m:")

        plt.title('metrics')
        plt.xlabel('epoch')
        plt.legend(['acc', 'val_acc', 'mse', 'val_mse'], loc='upper left')
        plt.savefig('mpmodel.png')
        plt.show()

def predict(query_len, doc_len, emb_dim):
    model = load_model("./trained/mp.15-0.83-0.38.h5")
    
    # pdocs, docs_dict = dataset_dict('../dataset/yolanda/corpus/MED.ALL')
    # pqueries, queries_dict = dataset_dict('../dataset/yolanda/queries/MED.QRY')
    # relevances = read_relevances('../dataset/yolanda/relevance/MED.REL')

    docs_dict, pdocs = read_all('../dataset/jsons/CISI.ALL.json')
    queries_dict, pqueries = read_qry('../dataset/jsons/CISI.QRY.json')
    relevances = read_rel('../dataset/jsons/CISI.REL.json', len(pqueries))
    
    #Load the words vectors dict
    fd = open('./word2vect/w2vect50.bin')
    w2v_dict = js.load(fd)

    #Processing query
    query_id = 10
    query = pqueries[query_id -1]
    print("Processed query ", query_id, ": ", query)
    vquery = doc2vector_mp(query, w2v_dict,query_len, emb_dim)

    print(relevances[query_id])

    #Processing relevants docs
    vdocs = {}
    for d_id in range(1, len(docs_dict) + 1):
        pdoc = docs_dict[d_id]
        vdoc = doc2vector_mp(pdoc, w2v_dict, doc_len, emb_dim)
        vdocs[d_id] = vdoc
        pair = [np.array([vquery]), np.array([vdoc])]
        solve = model.predict(pair)[0]
        if solve[1] > 0.5:
            print("doc id ", d_id, " -> ", solve)

if __name__ == "__main__":
    query_len = 50
    doc_len = 200
    emb_dim = 50

    docs_dict, pdocs = read_all('../dataset/jsons/CISI.ALL.json')
    queries_dict, pqueries = read_qry('../dataset/jsons/CISI.QRY.json')
    relevances = read_rel('../dataset/jsons/CISI.REL.json', len(pqueries))

    true, false, relpairs = conforms_pairs(relevances, len(docs_dict))
    print(len(true), len(false), len(relpairs))
    
    r.shuffle(false)

    m = len(false) // 10
    false = false[:m]
    relpairs = true + false
    r.shuffle(relpairs)

    print( len(true), len(false), len(relpairs))
    
    fd = open('./word2vect/w2vect50.bin')
    w2v_dict = js.load(fd)

    X_query, X_doc, Y, XV_query, XV_doc, YV = data2train_mp(docs_dict, queries_dict, relpairs, w2v_dict, query_len, doc_len, emb_dim)

    print("creating model")
    model = MatchPyramidModel(query_len, doc_len, emb_dim)
    print("ready to train")
    start = time.time()
    model.train(X_query, X_doc, Y, XV_query, XV_doc, YV, 50)
    end = time.time()

    print(end - start)

    # predict(query_len, doc_len, 50)