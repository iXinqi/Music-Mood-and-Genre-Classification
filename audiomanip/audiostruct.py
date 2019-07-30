import os
import keras
import librosa
import numpy as np
from sklearn import preprocessing
from keras.layers import concatenate
# @Class: MelSpectrogram
# @Description: 
#  Class to read audio files and export the songs as MelSpectrograms
class AudioStruct(object):
  def __init__(self, file_path):
    # Constants
    self.song_samples = 660000
    self.file_path = file_path
    self.n_fft = 2048
    self.n_mels = 128
    self.hop_length = 512
    self.n_mfcc = 20
    self.genres = {'chill-out': 1, 'electronica': 9, 'house': 14, 'christmas': 2, 'rock': 21, 'funk': 11, 'dance-club-general': 6, 'folk-nu-folk': 10, 'grooves': 12, 'new-age': 17, 'classical': 3, 'comedy': 4, 'easy-listening': 8, 'hip-hop': 13, 'children': 0, 'urban': 22, 'jazz': 15, 'country': 5, 'retro': 20, 'latin': 16, 'period': 18, 'dance-traditional': 7, 'pop': 19, 'world-music-crossover': 23}


  # @Method: getdata
  # @Description:
  #  Retrieve data from audio files and return then as numpy arrays
  def get_genres(self):
      return dict(zip(self.genres.values(), self.genres.keys()))
  
  def set_genres(self, classdict):
    self.genres = classdict
    return   
  
  def getdata(self,fold = 1, num = 1):
    # Structure for the array of songs
    song_data = []
    genre_data = []
    filename_data = []    
    # Read files from the folders
    for x,_ in self.genres.items():
      for root, subdirs, files in os.walk(self.file_path + x):
        n = int(len(files)/fold)
        for file in files[n*(num-1):(n*num)]:
          # Read the audio file
            file_name = self.file_path + x + "/" + file
            
            print(int(os.path.getsize(file_name)/1024))
            
            if int(os.path.getsize(file_name)/1024) != 703:
              print('*******  skip not 704kb  *************')
              print(file_name)              
              print('*******  skip not 704kb  *************')             
              continue
#            
            try:
              signal, sr = librosa.load(file_name)            	        	
              print(file_name)

              # Calculate the melspectrogram of the audio and use log scale
              melspec = librosa.feature.melspectrogram(signal[:self.song_samples], sr = sr, n_fft = self.n_fft, n_mels=self.n_mels, hop_length = self.hop_length).T[:2000,]
             
              # Try calculate the mfcc of the aduio and use log scale
              v_mfcc = librosa.feature.mfcc(signal[:self.song_samples], sr=sr, n_mfcc=self.n_mfcc).T[:2000,]  

              # calculate the stft
#              v_stft = librosa.stft(signal[:self.song_samples], n_fft = 512).T[:1280,]
             
#              print(type(v_mfcc))
#              print(type(melspec))
#              print(v_mfcc.shape)
#              print(melspec.shape)              
#              print(v_stft.shape)
              
              result = np.hstack((v_mfcc, melspec))
              print(result.shape)
              
              #result = preprocessing.scale(result)
              
              # Append the result to the data structure
              filename_data.append(file_name)
#              song_data.append(melspec)
#              song_data.append(v_mfcc)
              song_data.append(result)
              genre_data.append(self.genres[x])
            except:
              print(file_name,'*** load failed ***')

    return filename_data, np.array(song_data), keras.utils.to_categorical(genre_data, len(self.genres))
 
    return filename_data, np.array(song_data)