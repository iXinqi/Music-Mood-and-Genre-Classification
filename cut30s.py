# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:59:46 2017

@author: lixinqi
"""

from pydub import AudioSegment
import os, re, random
AudioSegment.converter = r"D:\\ffmpeg\\bin\\ffmpeg.exe"

# 循环目录下所有文件
g = os.walk('.\\tocut\\sad')
for path,d,filelist in g:
    for filename in filelist:

        full_filename = os.path.join(path, filename)
         # 打开mp3文件       
        try:
            mp3 = AudioSegment.from_mp3(full_filename)
            mp3[:30*1000].export(full_filename[:-4]+'_000_030'+full_filename[-4:], format="mp3") # 切割前30秒并覆盖保存
            mp3[30*1000: 60*1000].export(full_filename[:-4]+'_030_060'+full_filename[-4:], format="mp3") # 切割30-60秒并覆盖保存
            mp3[60*1000: 90*1000].export(full_filename[:-4]+'_060_090'+full_filename[-4:], format="mp3")# 切割前60-90秒并覆盖保存
            mp3[90*1000: 120*1000].export(full_filename[:-4]+'_090_120'+full_filename[-4:], format="mp3") # 切割前90-120秒并覆盖保存
        except:
            print(full_filename)
        os.remove(full_filename)