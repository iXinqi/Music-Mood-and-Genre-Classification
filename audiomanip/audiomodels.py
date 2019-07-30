import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Bidirectional, CuDNNGRU
from keras.layers import LSTM
from keras.layers.recurrent import GRU
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from keras.layers.recurrent import LSTM
from keras.optimizers import Adam

# @Class: ModelZoo
# @Description: Set of models to use to solve the classification problem.
class ModelZoo(object):

  @staticmethod
  def cnn_melspect_1D(input_shape, class_num):
    
    kernel_size = 3
    #activation_func = LeakyReLU()
    activation_func = Activation('relu')
    inputs = Input(input_shape)

    # Convolutional block_1
    conv1 = Conv1D(64, kernel_size)(inputs) #64
    act1 = activation_func(conv1)
    bn1 = BatchNormalization()(act1)
    pool1 = MaxPooling1D(pool_size=2, strides=2)(bn1)

    # Convolutional block_2
    conv2 = Conv1D(128, kernel_size)(pool1) #128
    act2 = activation_func(conv2)
    bn2 = BatchNormalization()(act2)
    pool2 = MaxPooling1D(pool_size=2, strides=2)(bn2)

    # Convolutional block_3
    conv3 = Conv1D(256, kernel_size)(pool2) #256
    act3 = activation_func(conv3)
    bn3 = BatchNormalization()(act3)
    pool3 = MaxPooling1D(pool_size=2, strides=2)(bn3) 
    
    # Convolutional block_4
    conv4 = Conv1D(256, kernel_size)(pool3)
    act4 = activation_func(conv4)
    bn4 = BatchNormalization()(act4)
    pool4 = MaxPooling1D(pool_size=4, strides=4)(bn4)
    
    # LSTM
#    drop_1 = Dropout(0.2)(pool4)
#    l_lstm = LSTM(32, return_sequences = True, go_backwards= False)(drop_1)
#    r_lstm = LSTM(32, return_sequences = True, go_backwards= True)(drop_1)
#    lstm_merged = keras.layers.merge([l_lstm, r_lstm], mode='sum')
#    lstm_merged = Dropout(0.5)(lstm_merged)
    
#     RNN
#    drop_1 = Dropout(0.2)(pool4)
#    l_rnn = GRU(64, return_sequences = True, go_backwards= False)(drop_1)
#    r_rnn = GRU(64, return_sequences = True, go_backwards= True)(drop_1)
#    rnn_merged = keras.layers.merge([l_rnn, r_rnn], mode='sum')
#    pool4 = Dropout(0.5)(rnn_merged)
    
    # Global Layers
    gmaxpl = GlobalMaxPooling1D()(pool4)
    gmeanpl = GlobalAveragePooling1D()(pool4)
    mergedlayer = concatenate([gmaxpl, gmeanpl], axis=1)
    
    # Regular MLP
    dense1 = Dense(512,
        kernel_initializer='glorot_normal',
        bias_initializer='glorot_normal')(mergedlayer)
    actmlp = activation_func(dense1)
    reg = Dropout(0.5)(actmlp)
    
    dense2 = Dense(512,
        kernel_initializer='glorot_normal',
        bias_initializer='glorot_normal')(reg)
    actmlp = activation_func(dense2)
    reg = Dropout(0.5)(actmlp)
    
    dense2 = Dense(class_num, activation='softmax')(reg)
    
    model = Model(inputs=[inputs], outputs=[dense2])
  
    return model


  def lstm(input_shape, class_num):
    cnn = Sequential()
    cnn.add(LSTM(units=64, dropout=0.5, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
    cnn.add(LSTM(units=64, dropout=0.5, recurrent_dropout=0.35, return_sequences=False))
    cnn.add(Dense(class_num, activation='softmax'))
    return cnn


  
  def get_model(input_shape, class_num):
    input_layer = Input(input_shape)
    # embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
    #                             weights=[embedding_matrix], trainable=False)(input_layer)
    
    x = Bidirectional(GRU(128, kernel_initializer='random_uniform', return_sequences=True, dropout=0.0, recurrent_dropout=0.5))(input_layer)
#    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(128, kernel_initializer='random_uniform', return_sequences=False, dropout=0.0, recurrent_dropout=0.5))(x)
    x = Dense(64, activation="relu")(x)
    output_layer = Dense(class_num, activation="relu")(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
  