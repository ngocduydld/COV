from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
import numpy as np
from keras import metrics
from m_data_aspect import load_data_forlstm

model_json_full = "LSTM_aspect.json"

EMBEDDING_DIM = 300
num_filters = 298
drop = 0.2
NUM_WORDS = 50000
epoch = 1000
batch_size = 512
max_length = 300
pad = 'post'
activation_func = "sigmoid"
train_len = 0
np.random.seed(0)
pool_size = 2
np.random.seed(0)

X_train, y_train, X_test, y_test, X_val, y_val, vocabulary_size, embedding_layer, dense_num = load_data_forlstm()

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(units=300))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', metrics.Precision(), metrics.Recall()])

checkpoint_filepath = 'LSTM_aspect-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='auto',
    save_freq='epoch')

model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1,
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
