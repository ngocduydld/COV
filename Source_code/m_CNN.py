from keras import metrics
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout,concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from m_data_aspect import load_data

model_json_file = "CNN_aspect.json"
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
L2 = 0.0004
np.random.seed(0)

X_train, y_train, X_test, y_test, X_val, y_val, vocabulary_size, embedding_layer, dense_num = load_data()

sequence_length = X_train.shape[1]
inputs = Input(shape=(sequence_length,))
embedding = embedding_layer(inputs)
reshape = Reshape((sequence_length,EMBEDDING_DIM,1))(embedding)

conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',
                kernel_regularizer=regularizers.l2(L2))(reshape)
conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',
                kernel_regularizer=regularizers.l2(L2))(reshape)
conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',
                kernel_regularizer=regularizers.l2(L2))(reshape)
conv_3 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='sigmoid',
                kernel_regularizer=regularizers.l2(L2))(reshape)
conv_4 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='sigmoid',
                kernel_regularizer=regularizers.l2(L2))(reshape)
maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)
maxpool_3 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_3)
maxpool_4 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_4)

merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4], axis=1)
flatten = Flatten()(merged_tensor)
reshape = Reshape((5*num_filters,))(flatten)
dropout = Dropout(drop)(flatten)
output = Dense(units=2, activation='softmax',kernel_regularizer=regularizers.l2(L2))(dropout)

model = Model(inputs, output)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
model.compile(loss=loss_fn,
              optimizer='adam',
              metrics=['acc', metrics.Precision(), metrics.Recall()])
callbacks = [EarlyStopping(monitor='val_loss')]

checkpoint_filepath = 'CNN_aspect-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
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
