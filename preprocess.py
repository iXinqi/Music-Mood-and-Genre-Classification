import pandas as pd
from pydub import AudioSegment
import os, re, random, librosa


bpm_list = []
file_list = []
g = os.walk('../dataset/genre_15_45s')
for path,d,filelist in g:
    for filename in filelist:

        full_filename = os.path.join(path, filename)
        
        #BPM : Estimate the global tempo for display purposes
#        y, sr = librosa.load(full_filename)
#        hop_length = 512
#        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
#        bpm = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)[0]
#        bpm_list.append(bpm)
        file_list.append(full_filename)
        print(full_filename + '***bpm finish')
        
#        # Cut mp3 file into 30 second       
#        try:
#            mp3 = AudioSegment.from_mp3(full_filename)
#            mp3[:30*1000].export(full_filename[:-4]+'_000_030'+full_filename[-4:], format="mp3") # 切割前30秒并覆盖保存
#            mp3[30*1000: 60*1000].export(full_filename[:-4]+'_030_060'+full_filename[-4:], format="mp3") # 切割30-60秒并覆盖保存
#            mp3[60*1000: 90*1000].export(full_filename[:-4]+'_060_090'+full_filename[-4:], format="mp3")# 切割前60-90秒并覆盖保存
#            mp3[90*1000: 120*1000].export(full_filename[:-4]+'_090_120'+full_filename[-4:], format="mp3") # 切割前90-120秒并覆盖保存
        
                # Cut mp3 file into 30 second       
        try:
            mp3 = AudioSegment.from_mp3(full_filename)
            mp3[:45*1000].export(full_filename[:-4]+'_000_045'+full_filename[-4:], format="mp3") # 切割前45秒并覆盖保存
            mp3[45*1000: 90*1000].export(full_filename[:-4]+'_045_090'+full_filename[-4:], format="mp3") # 45-90
            mp3[90*1000: 135*1000].export(full_filename[:-4]+'_090_135'+full_filename[-4:], format="mp3")# 90-135
        except:
            print(full_filename)
        os.remove(full_filename)
        print(full_filename + '***cut finish')

        
#dataframe = pd.DataFrame({'BPM':bpm_list, 'File_name':file_list})
#dataframe.to_csv('BPM.csv',index=False,sep = ',') 
        
dataframe = pd.DataFrame({'File_name':file_list})
dataframe.to_csv('BPM_genre.csv',index=False) 