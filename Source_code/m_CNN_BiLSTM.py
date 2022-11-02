import numpy as np
import os
from keras import metrics
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dropout, Bidirectional
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from m_data_aspect import load_data

model_json_full = "CNN_BiLSTM_aspect.json"

EMBEDDING_DIM = 300
filter_sizes = [3, 4, 5]
num_filters = 298
drop = 0.2
NUM_WORDS = 50000
epoch = 1000
batch_size = 512
max_length = 300
pad = 'post'
activation_func = "relu"
train_len = 0
pool_size = 2
np.random.seed(0)

input_shape = (300,300,1)
kernel_size = 3

X_train, y_train, X_test, y_test, X_val, y_val, vocabulary_size, embedding_layer, dense_num = load_data()
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.2))
model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.3))
model.add(Conv1D(128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.4))
model.add(Conv1D(256, kernel_size=4, padding='same', activation='sigmoid'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.3))
model.add(Conv1D(298, kernel_size=5, padding='same', activation='sigmoid'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.4))
lstm1 = Bidirectional(LSTM(300, return_sequences=False))
model.add(lstm1)
model.add(Dropout(0.35))
model.add(Dropout(0.4))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy', metrics.Precision(), metrics.Recall()])

checkpoint_filepath = 'CNN_BiLSTM-{epoch:03d}-{val_loss:.4f}-{val_accuracy:.4f}.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='auto',
    save_freq='epoch')

model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch,
          validation_data=(X_val, y_val),
          callbacks=[model_checkpoint_callback])

model_json = model.to_json()
with open(model_json_full, 'w') as json_file:
    json_file.write(model_json)

scores = model.evaluate(X_test, y_test)
print("Loss:", (scores[0]))
print("Accuracy:", (scores[1]*100))
print("Precision:", (scores[2]))
print("Recall:", (scores[3]*100))
