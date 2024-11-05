
from datetime import datetime
import math
import os
from flask import Flask, flash, jsonify, redirect, request, url_for, session
from flask import render_template
# from flask_sqlalchemy import SQLAlchemy
import requests
import h5py
import yaml
# from video_process.frame_extraction import FrameExtractor
# from video_process.deep_feature import FeatureExtractor
import json
# from video_model.knn import SearchNearestModel, search_image_nearest
# from video_model.network import  VideoRetrievalNetwork
# from image_model.network import Image_Retrieval_Network
import h5py
# import pymysql
import traceback
import joblib
import numpy as np
import re
# import VideoRetrieval
# import ImageRetrieval
# from common.md5_operate import get_md5
# from common.redis_operate import redis_db
import re, time
# from sql import db,Users
# from common.mysql_operate import db
from werkzeug.security import generate_password_hash, check_password_hash
app = Flask(__name__)
with open('sql_config/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
app.config["JSON_AS_ASCII"] = False
# app.register_blueprint(VideoRetrieval.blueprint)
# app.register_blueprint(ImageRetrieval.blueprint)

app.config['SQLALCHEMY_DATABASE_URI'] = config['database']['uri']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = config['database'].get('track_modifications', False)
app.secret_key = config['secret_key']
db.init_app(app)


config = json.load(open('config.json'))
query_video_path = config['query_video_path']
query_frames_path = config['query_frames_path']
frame_batch_size = config["frame_batch_size"]
pretrained = config['pretrained']
cfg_path = config['cfg']
args_path = config['args']
candidate_feature_path = config['candidate_feature']
max_hits = config['max_hits']

upload_image_path = config['query_image_path']
upload_video_path = config['query_video_path']



# 在应用程序启动前进行预加载
@app.before_first_request
def preload():
    app.frame_extraction = FrameExtractor(query_video_path, query_frames_path)
    app.deep_feature = FeatureExtractor(frame_batch_size, pretrained)

    app.video_feature_mapping = VideoRetrievalNetwork(cfg_path, args_path)
    # video_retrieval_network = VideoRetrievalNetwork(cfg_path, args_path)
    app.searcher = SearchNearestModel(candidate_feature_path)
 

    app.history_record = []
@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login')
def login():
    return redirect(url_for('login_page'))

@app.route('/register')
def register_page():
    return render_template('register.html')


@app.route('/registuser', methods=['POST','GET'])
def registuser():
    if request.method == "POST":
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # 基本验证
        if not all([username, email, password]):
            return jsonify({'message': 'Missing fields'}), 400
        
        # 检查用户名和邮箱是否已被使用
        existing_user = Users.query.filter((Users.username == username) | (Users.email == email)).first()
        if existing_user:
            flash('用户名或邮箱已被使用，请尝试其他的用户名或邮箱。')
            return redirect(url_for('register_page'))
    
        # 密码加密
        hashed_password = generate_password_hash(password, method='sha256')
        new_user = Users(username=username, email=email, password=hashed_password)
        # 插入数据库
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('注册成功，请登录。')
            return redirect(url_for('login_page'))
        except Exception as e:
            db.session.rollback()
            flash('注册失败，请稍后再试。错误信息：{}'.format(str(e)))
            return redirect(url_for('register_page'))
    
    # 如果是GET请求，或者注册失败/用户名已被使用，就重新渲染注册页面
    return render_template('register.html')

@app.route('/loginuser', methods=['POST',"GET"])
def loginuser():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = Users.query.filter_by(username=username).first()
        # 检查用户是否存在以及密码是否正确
        if user and check_password_hash(user.password, password):
            # 这里的 user_id 应该是用户的唯一标识，比如用户表的主键
            session['user_id'] = user.user_id
            session['username'] = username
            session['logged_in'] = True
            flash('登录成功！')
            # return redirect('/home')
            return redirect(url_for('show_video_library'))
        else:
            flash('用户名或密码错误，请重试。')
            return redirect(url_for('login_page'))
    # 如果是 GET 请求或者登录失败，就重新渲染登录页面
    return redirect(url_for('login_page'))

@app.route('/show_video_library')
def show_video_library():
    resp = {
        'hangout': {},
        'page_info': {}
    }
    
    idlist = []
    try:
        with open('./static/upload/candidate_videos50715.txt', 'r') as f:
            for line in f:
                video_id = line.strip().split('.')[0]
                idlist.append(video_id)
    except IOError:
        resp['status'] = 1  # Set an error status if file can't be read
        resp['ret']['error_msg'] = 'Unable to read video list file'
        return render_template('error.html', resp=resp)  # Render an error template if needed

    idlist.sort()

    if 'username' in session:
        samples_per_page = 20
        total_videos = len(idlist)
        total_pages = math.ceil(total_videos / samples_per_page)
        
        # Get 'page' from query parameters, default to 1 if not provided
        current_page = int(request.args.get('page', 1))
        print(current_page)

        resp['page_info'] = {
            'total_pages': total_pages,
            'cur_page': current_page
        }
        
        # Calculate left and right bounds for pagination links
        left = max(1, current_page - 2)
        right = min(total_pages, left + 4)
        if right - left < 4:
            left = max(1, right - 4)

        resp['page_info']['pages'] = list(range(left, right + 1))
        
        # Calculate start and end index for slicing the list
        start_index = (current_page - 1) * samples_per_page
        end_index = start_index + samples_per_page
        random_ids = idlist[start_index:end_index]

        resp['hangout']['random_queries'] = random_ids
        
        return render_template('system_index.html', resp=resp)  # Removed the extra .html extension

    # If username not in session, redirect to login or handle it appropriately
    return redirect(url_for('login_page'))  # Replace 'login' with your actual login view function



@app.route('/logout')
def logout():
    # 删除用户session
    session.clear()
    flash('您已成功退出。')
    return redirect(url_for('login_page'))
   

if __name__ == '__main__':
    
    app.run(port="4568",debug=True)