# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 16:29:11 2017

@author: lixinqi
"""
import os, random
import shutil
count = 0
original_fold = os.walk('.\\full_music_111G')
subset = '.\\subset'
for path,d,filelist in original_fold:
    for filename in filelist:
        rnum = random.random()
        if 0.1<rnum<0.6:
            continue
        
        label = path.split('\\')[2]
        data_fold = os.path.join(subset, label)
        if not os.path.exists(data_fold):
            os.mkdir(data_fold)
        try:
            shutil.copy(os.path.join(path, filename), os.path.join(data_fold, filename))
        except:
            print(os.path.join(path, filename))
        
        count += 1
        if count%100 == 0:
            print(count)