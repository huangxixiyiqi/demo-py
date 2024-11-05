# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 20:58:41 2022
@author: Administrator
"""
 
import requests
 
def client_post(url, data):
    rep = requests.post(url, files=data)
    return rep.text 
 
if __name__ == '__main__':
    files = {
            "imagename":"videdo server demo",
            "image":("6520763290014977287.mp4",open('/media/disk3/yjt/videoret_demo/6520763290014977287.mp4','rb'),"application/octet-stream"),
            "imagetype":"jpg",
            "key1":"key1 content",
            "video_id":"6520763290014977287"
            }
    url = 'http://127.0.0.1:8080/upload/search/'
    res = client_post(url, files)
    print('127.0.0.1:8080/upload/search(返回结果):', res)