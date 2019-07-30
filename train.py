import gc
import os
import ast
import sys
import configparser

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras import backend as K

from audiomanip.audiostruct import AudioStruct
from audiomanip.audiomodels import ModelZoo
from audiomanip.audioutils import AudioUtils
from audiomanip.audioutils import MusicDataGenerator

# load mp3 file
from pydub import AudioSegment
AudioSegment.converter = r"D:\\ffmpeg\\bin\\ffmpeg.exe"

# Disable TF warnings about speed up
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main():
  # Parse config file
  config = configparser.ConfigParser()
  config.read('train_params.ini')

  # Constants
  ## Configuration
  GTZAN_FOLDER = config['FILE_READ']['GTZAN_FOLDER']
  MODEL_PATH = config['FILE_READ']['SAVE_MODEL']
  SAVE_NPY = ast.literal_eval(config['FILE_READ']['SAVE_NPY'])
  EXEC_TIMES = int(config['PARAMETERS_MODEL']['EXEC_TIMES'])
  CNN_TYPE = '1D'

  ## CNN hyperparameters
  batch_size = int(config['PARAMETERS_MODEL']['BATCH_SIZE'])
  epochs = int(config['PARAMETERS_MODEL']['EPOCHS'])
  genres_class = eval(config['PARAMETERS_MODEL']['CLASS_DICT'])
  class_num = int(config['PARAMETERS_MODEL']['CLASS_NUM'])

  # Read data
  data_type = config['FILE_READ']['TYPE']
  
  # use melspectrogram
  # input_shape = (128, 128)
  
  # use mfcc + melspectrogram
  input_shape = (129, 148)
  print("data_type: %s" % data_type)
  
#  # use GTZAN
#  Nfold = 1
#  numset = [1]
  
  # use big data
  Nfold = 10
  numset = [1,2,3,4,5,6,7,8,9,10]
  
  val_acc = []
  test_history = []
  test_acc = []
  test_acc_mvs = []
  for x in range(EXEC_TIMES):
    for num in numset:
    # Read the audio files
      if data_type == 'AUDIO_FILES':
        song_rep = AudioStruct(GTZAN_FOLDER)
        #mood_29classes = {'dramatic': 7, 'nostalgic': 18, 'driving-exciting-exhilarating': 9, 'beds-underscore': 2, 'bright-optimistic': 3, 'peaceful-pastoral': 19, 'happy-sprightly-jolly': 12, 'warm-uplifting': 28, 'beautiful': 1, 'drama-general': 6, 'jaunty-whimsical': 15, 'quirky-strange': 21, 'anger-aggression': 0, 'hypnotic': 13, 'light-tension': 17, 'tension': 25, 'ghostly-eerie-spooky': 11, 'funny-comedy': 10, 'dark': 5, 'dream-heavenly-flight': 8, 'simple-sparse-minimal': 24, 'laid-back': 16, 'dance-club': 4, 'inspiring-stirring': 14, 'thoughtful-reflective': 26, 'power-energy': 20, 'violence': 27, 'sad': 23, 'romantic': 22}
        #mood_5classes = {'anger-violence-tension': 1, 'beautiful-peaceful': 0, 'happy-funny': 4, 'inspiring-stirring': 2, 'sad': 3}
        song_rep.set_genres(genres_class)
        filename, songs, genres = song_rep.getdata(fold=Nfold,num = num)

    # Save the audio files as npy files to read faster next time
        if SAVE_NPY:
          np.save(GTZAN_FOLDER + 'songs'+'_total'+str(Nfold)+'_part'+str(num)+'.npy', songs)
          np.save(GTZAN_FOLDER + 'genres'+'_total'+str(Nfold)+'_part'+str(num)+'.npy', genres)

    # Read from npy file
      elif data_type == 'NPY':
        songs = np.load(GTZAN_FOLDER + 'songs'+'_total'+str(Nfold)+'_part'+str(num)+'.npy')
        genres = np.load(GTZAN_FOLDER + 'genres'+'_total'+str(Nfold)+'_part'+str(num)+'.npy')
  
    # Not valid datatype
      else:
        raise ValueError('Argument Invalid: The options are AUDIO_FILES or NPY for data_type')

      print("Original songs array shape: {0}".format(songs.shape))
      print("Original genre array shape: {0}".format(genres.shape))
    
    
    # Split the dataset into training and test
      X_train, X_test, y_train, y_test = train_test_split(
      songs, genres, test_size=0.1, stratify=genres, random_state = 0)
    
    # Split training set into training and validation
      X_train, X_Val, y_train, y_val = train_test_split(
      X_train, y_train, test_size=1/6, stratify=y_train, random_state = 0)
        
    # Split the train, test and validation data in size 128x128
      X_Val, y_val = AudioUtils().splitsongs_melspect(X_Val, y_val, CNN_TYPE)
      X_test, y_test = AudioUtils().splitsongs_melspect(X_test, y_test, CNN_TYPE)
      X_train, y_train = AudioUtils().splitsongs_melspect(X_train, y_train, CNN_TYPE)
      
#      print(input_shape)
      
    # Construct the model
      #cnn = ModelZoo.cnn_melspect_1D(input_shape, class_num)
      
      
    #LSTM
      cnn = ModelZoo.get_model(input_shape, class_num)
        
      print("\nTrain shape: {0}".format(X_train.shape))
      print("Validation shape: {0}".format(X_Val.shape))
      print("Test shape: {0}\n".format(X_test.shape))
      print("Size of the CNN: %s\n" % cnn.count_params())

    # Optimizers
      #cnn
      #sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-5, nesterov=True)
      
      #lstm
      sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-5, nesterov=True)
      adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
    
    # Compiler for the model
      cnn.compile(loss=keras.losses.categorical_crossentropy,
      optimizer=sgd,
      metrics=['accuracy'])

    # Early stop
      earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',
      min_delta=0,
      patience=3,
      verbose=0,
      mode='auto')
    
    # Load Weights 
#      try:
#          cnn.load_weights(MODEL_PATH)
#      except:
#          print('no model to use')
    
    # Fit the model
      history = cnn.fit(X_train, y_train,
      batch_size=batch_size,
      epochs=epochs,
      verbose=1,
      validation_data=(X_Val, y_val),
      callbacks = [earlystop])

      score = cnn.evaluate(X_test, y_test, verbose=0)
      score_val = cnn.evaluate(X_Val, y_val, verbose=0)
        
    # Majority Voting System (Top 2 Err)
      pred_values = np.argsort(cnn.predict(X_test), axis=1)[:,-2:]
      print('pred_values:')
      print(pred_values.shape) 
      
      mvs_truth, mvs_res = AudioUtils().voting(np.argmax(y_test, axis = 1), pred_values)

      if len(mvs_truth) != len(mvs_res):
        print('mismatch')
      
      total = len(mvs_truth) 
      right = 0
      for i in range(len(mvs_truth)):
          if mvs_truth[i] in mvs_res[i]:
            right += 1
      acc_mvs = right/total

    # Save metrics
      val_acc.append(score_val[1])
      test_acc.append(score[1])
      test_history.append(history)
      test_acc_mvs.append(acc_mvs)

    # Print metrics
      print('Test accuracy:', score[1])
      print('Test accuracy for Majority Voting System:', acc_mvs)

    # Save the model
      cnn.save(MODEL_PATH)
    
    # Free memory  
      del songs
      del genres
      gc.collect()
  
  # Print the statistics
  print("Validation accuracy - mean: %s, std: %s" % (np.mean(val_acc), np.std(val_acc)))
  print("Test accuracy - mean: %s, std: %s" % (np.mean(test_acc), np.std(test_acc)))
  print("Test accuracy MVS - mean: %s, std: %s" % (np.mean(test_acc_mvs), np.std(test_acc_mvs)))
  

if __name__ == '__main__':
  main()
  