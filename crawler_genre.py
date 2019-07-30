# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:24:04 2017

@author: zhangjinfeng
"""



import urllib
import re, os
from bs4 import BeautifulSoup
from distutils.filelist import findall
import pandas as pd

def add_zero(cur_str):
    if len(cur_str) == 1:
        return '0' + cur_str
    else:
        return cur_str
    
# 歌曲下载函数  
def download_song(song_data, path='./aaaa/'):
    if not os.path.exists(path):
        os.mkdir(path)        
    song_data.reset_index(drop=True, inplace=True)
    prefix_str = 'http://content2.audionetwork.com/Preview/tracks/mp3/v5res/ANW'
    song_data['download_url'] = prefix_str + song_data.first_num + '/' + song_data.last_num + '.mp3'
    for cur_inx in song_data.index:
        cur_download_url = song_data.ix[cur_inx, 'download_url']
        # 保存路径
        output_name = path + ''.join(song_data.ix[cur_inx, ['first_num', 'last_num']].values) + '.mp3'
        try:
            urllib.request.urlretrieve(cur_download_url, output_name)
        except:
            print(song_data.ix[cur_inx, 'song_name'])  
   
  

# 无子类歌曲信息获取
def get_song_info(url):
    page = urllib.request.urlopen(url)
    contents = page.read()
    soup = BeautifulSoup(contents, "html.parser")
    try:
        n_page = int(soup.find('li', class_='anw-track-results__controls__pagination__page--info').get_text().strip().split()[-1])
    except:
        n_page = 1
    song_name = list()
    first_num = list()
    last_num = list()
    for cur_page in range(1, n_page+1):
        cur_page_song_name = list()
        cur_page_first_num = list()
        cur_page_last_num = list()
        while len(cur_page_song_name) == 0:
            print('working on: {}/{}'.format(cur_page, n_page))
            cur_url = url + '?page={}&size=1&sort=2'.format(cur_page)
            page = urllib.request.urlopen(cur_url)
            contents = page.read()
            soup = BeautifulSoup(contents, "html.parser")  
            for tag in soup.find_all('div', class_="col-xs-12 col-sm-8"):
                cur_page_song_name.append(tag.find('strong', class_='js-title').get_text().strip())
                for cur_a in tag.find_all('a'):
                    cur_str = cur_a.get_text().strip()
                    cur_str = re.search('\(\d+/\d+\)', cur_str)
                    if cur_str:
                        cur_list = cur_str.group().strip('()').split('/')
                        cur_page_first_num.append(cur_list[0])
                        cur_page_last_num.append(cur_list[1])
        song_name.extend(cur_page_song_name)
        first_num.extend(cur_page_first_num)
        last_num.extend(cur_page_last_num)
    last_num = list(map(add_zero, last_num))
    cur_song_data = {}
    cur_song_data['song_name'] = song_name
    cur_song_data['first_num'] = first_num
    cur_song_data['last_num'] = last_num
    cur_song_data = pd.DataFrame(cur_song_data)
    return cur_song_data


# 有子类歌曲信息获取
def get_class_song_info(url):
    web_site = 'https://www.audionetwork.com'
    page = urllib.request.urlopen(url)
    contents = page.read()
    soup = BeautifulSoup(contents, "html.parser")  
    song_data = pd.DataFrame()
    for tag in soup.find_all('li', class_='anw-cats__item col-xs-12 col-sm-6 col-md-4'):
        print(tag.find('a').get_text().strip())
        sub_class_url = web_site + tag.find('a')['href']
        cur_song_data = get_song_info(sub_class_url)
        song_data = pd.concat([song_data, cur_song_data])
    song_data.reset_index(drop=True, inplace=True)
    return song_data




'''

# 有子类歌曲下载，以 style 为 national-anthem 类为例
url = 'https://www.audionetwork.com/browse/m/musical-styles/national-anthems'
song_data = get_class_song_info(url)
path = './national/' # 保存在当前路径 children 文件夹下
download_song(song_data, path)


# 无子类歌曲下载，以 style 为 punk 类为例
url = 'https://www.audionetwork.com/browse/m/musical-styles/musical-styles/punk/results'
song_data = get_song_info(url)
path = './punk/' # 保存在当前路径 punk 文件夹下
download_song(song_data, path)
'''

if __name__ == '__main__':
    #全部情感类别
    mood_list = ["children", "chill-out", "christmas", "classical", "comedy",\
                "dance-traditional", "dance-club-general", "electronica", "folk-nu-folk",\
                "grooves", "jazz", "latin", "new-age","period", "pop", "rock", "urban", \
                "country", "easy-listening", "funk", "house", "hip-hop", "retro", "world-music-crossover"]
    #有子类的类别
    with_subclass = ["children", "christmas", "classical", "comedy", "dance-traditional", "jazz", \
                     "latin","period", "pop", "rock"]
    
    #本程序爬取的类别
    my_list = ["chill-out"]
        
    for label in my_list:
        print(label)
        
        if label in with_subclass:   #有子类下载
            url = 'https://www.audionetwork.com/browse/m/musical-styles/' +label
            song_data = get_class_song_info(url)
            path = './'+label+'/' 
            download_song(song_data, path)
        else:   #无子类下载
            url = 'https://www.audionetwork.com/browse/m/musical-styles/musical-styles/' + label +'/results'
            song_data = get_song_info(url)
            path = './'+label+'/' # 保存在当前路径 punk 文件夹下
            download_song(song_data, path)

'''
import thread
import time

try:
   thread.start_new_thread( crawler )
   thread.start_new_thread( crawler )
except:
   print ("Error: unable to start thread")
 
while 1:
   pass
'''







## 数每个大类下，小类包含的音乐数
#import urllib
#import re
#from bs4 import BeautifulSoup
#from distutils.filelist import findall
#import pandas as pd
#
#url = 'https://www.audionetwork.com/browse/m/mood-emotion/romantic'
#page = urllib.request.urlopen(url)
#contents = page.read()
#soup = BeautifulSoup(contents, "html.parser")  
#
#dic = {}
#for tag in soup.find_all('li', class_='anw-cats__item col-xs-12 col-sm-6 col-md-4'):
#    cur_label = tag.find('a').get_text().strip()
#    if 'Norteno / Norteño' == cur_label:
#        continue
#    print(cur_label)
#    cur_url = url + '/' + cur_label.lower().replace("'", '').replace("-", '').replace("(", '').replace(")", '').replace('&', 'and').replace(' ', '-').replace('/', '-').replace('`', '%60').replace('---', '-') + '/results'
#    cur_page = urllib.request.urlopen(cur_url)
#    cur_contents = cur_page.read()
#    cur_soup = BeautifulSoup(cur_contents, "html.parser")
#    cur_n_page = re.search('\d+', cur_soup.find('div', class_='anw-track-results__controls__refine search-track-refine').get_text().strip()).group()
#    print(cur_n_page)
#    dic[cur_label] = cur_n_page
#    
#
#pd.Series(dic).to_csv('ccc.csv')