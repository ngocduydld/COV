import os
import tensorflow as tf
from keras import metrics
from keras.layers import Bidirectional, Input, Dense, Dropout, BatchNormalization, LSTM
import datetime
import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from m_data_aspect import load_data

os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print(datetime.datetime.now())
t1 = datetime.datetime.now()

model_json_file = "BiLSTM_aspect.json"

EMBEDDING_DIM = 300
filter_sizes = [3, 4, 5]
num_filters = 298
drop = 0.2
dropout = 0.5
NUM_WORDS = 50000
epoch = 700
batch_size = 512
max_length = 300
pad = 'post'
activation_func = "relu"
train_len = 0
np.random.seed(0)

X_train, y_train, X_test, y_test, X_val, y_val, vocabulary_size, embedding_layer, dense_num = load_data()

mirrored_strategy = tf.distribute.MirroredStrategy() # multi-GPU

with mirrored_strategy.scope():
    inputs = Input((max_length,))
    emb = embedding_layer(inputs)
    lstm1 = Bidirectional(LSTM(300, return_sequences=True))(emb)
    drop1 = Dropout(dropout)(lstm1)
    lstm2 = Bidirectional(LSTM(300, return_sequences=True))(drop1)
    drop2 = Dropout(dropout)(lstm2)
    lstm2 = Bidirectional(LSTM(300, return_sequences=False))(drop2)

    batch1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros',
                                gamma_initializer='ones', moving_mean_initializer='zeros',
                                moving_variance_initializer='ones', beta_regularizer=None,
                                gamma_regularizer=None, beta_constraint=None,
                                gamma_constraint=None)(lstm2)
    dense3 = Dense(128, activation='relu')(batch1)
    drop4 = Dropout(0.5)(dense3)
    dense4 = Dense(128, activation='relu')(drop4)
    drop5 = Dropout(0.5)(dense4)
    dense7 = Dense(128, activation='relu')(drop5)
    out = Dense(2, activation='softmax')(dense7)
    model = Model(inputs, out)

    from tensorflow.python.keras.optimizer_v1 import Adam

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy", metrics.Precision(), metrics.Recall()])

    checkpoint_filepath = 'BiLSTM_aspect-{epoch:03d}-{val_loss:.4f}-{val_accuracy:.4f}.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='auto',
        save_freq='epoch')

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch, verbose=1,
              validation_data=(X_val, y_val),
              callbacks=[model_checkpoint_callback])

model_json = model.to_json()
with open(model_json_file, 'w') as json_file:
    json_file.write(model_json)
scores = model.evaluate(X_test, y_test)
print("Loss:", (scores[0]))
print("Accuracy:", (scores[1]*100))
print("Precision:", (scores[2]))
print("Recall:", (scores[3]*100))
