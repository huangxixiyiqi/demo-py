import os
import web
import json
import math
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from collections import OrderedDict
import urllib.request
import uuid
render = web.template.render('templates/', base='index')
config = json.load(open('/home/zz/code/zz_code/demo/config.json'))
max_hits = config['max_hits']
upload_video_path = '/home/zz/code/zz_code/demo/static/msr_images/'
urls = (
	'/search/', 'search',
	'/download/', 'download',
	'', 're_upload',
	'/', 'upload'
)

resp = {
	'status': 0,
	'hangout': dict(),
	'ret': dict(),
	'page_info': dict(),
	
}

idlist = []
with open('/home/zz/code/zz_code/demo/static/upload/unlabeled-data-id', 'r') as f:
	for l in f:
		l = l.strip().split('.')[0]
		idlist.append(l)
idlist.sort()

class re_upload:
	def GET(self):
		raise web.seeother('/')

class upload:
	samples_per_page = 20
	def GET(self):
		global resp
		resp['status'] = 0
		input_data = web.input(page=None)
		print(input_data)

		resp['page_info']['total_pages'] = math.ceil(len(idlist)/self.samples_per_page)

		if input_data.page is not None:
			resp['page_info']['cur_page'] = int(input_data.page)
		else:
			resp['page_info']['cur_page'] = 1

		left = max(1, resp['page_info']['cur_page'] - 2)
		if left == 1:
			right = min(resp['page_info']['total_pages'], left + 4)
		else:
			right = min(resp['page_info']['total_pages'], resp['page_info']['cur_page'] + 2)
			if right == resp['page_info']['total_pages']:
				left = max(1, resp['page_info']['total_pages'] - 4)
		resp['page_info']['pages'] = list(range(left, right+1))

		s = (resp['page_info']['cur_page'] - 1) * self.samples_per_page
		e = s + self.samples_per_page
		random_ids = idlist[s:e]
		resp['hangout']['random_queries'] = random_ids
		return render.upload(resp=resp)

class search:
	samples_per_page = 5
	def __init__(self):
		self.rank = []

	def GET(self):
		global resp
		input_data = web.input(page=None)
		print(input_data)
		if not resp['status']:
			raise web.seeother('/')
		# if input_data.qid:
		# 	resp['ret']['qid'] = input_data.qid
		if input_data.page:
			resp['page_info']['cur_page'] = int(input_data.page)
		s = (resp['page_info']['cur_page'] - 1) * self.samples_per_page
		e = s + self.samples_per_page
		# r = ranks.search(resp['ret']['qid'])
		resp['ret']['ranks'] = resp['ret']['overall_ranking'][s:e]
		total_pages = math.ceil(len(resp['ret']['overall_ranking'])/self.samples_per_page)
		left = max(1, resp['page_info']['cur_page'] - 2)
		if left == 1:
			right = min(total_pages, left + 4)
		else:
			right = min(total_pages, resp['page_info']['cur_page'] + 2)
			if right == total_pages:
				left = max(1, total_pages - 4)
		resp['page_info']['pages'] = list(range(left, right+1))
		return render.upload(resp=resp)

	def POST(self):
		global resp
		x = web.input(image={})
		filedir = upload_video_path
		filepath = x.image.filename.replace('\\', '/')
		filename = filepath.split('/')[-1]
		print(filename)
		save_file_path = os.path.join(filedir,filename)
		if os.path.exists(save_file_path): # 如果已经存在该文件重命名一下
			save_file_path = os.path.join(filedir,str(uuid.uuid4()) + '.' + filename.split('.')[1])
		fout = open(save_file_path, 'wb')
		fout.write(x.image.file.read())
		fout.close()
		print(f"文件保存成功,路径为{save_file_path}")
		resp['status'] = 1
		
		return render.upload(resp=resp)
		# return json.dumps(resp)


class download:
	samples_per_page = 5
	def __init__(self):
		self.rank = []
	
	def POST(self):
		global resp
		resp['status'] = 1
		data = web.input()
		print(data)
		filedir = upload_video_path
		url = web.input().url
		video_name = '123.mp4'
		urllib.request.urlretrieve(url, filedir + '/' + video_name)
		qid = video_name.split('.')[0]
		resp['ret']['qid'] = qid
		resp['page_info']['cur_page'] = 1

		qframes = web.frame_extraction.frame_extractor(qid) #qframes:[video帧路径]
		qfeature = web.deep_feature.extract_feature(qframes) # qfeature维度：m * 2048
		qfeat = web.video_feature_mapping.query_feature_mapping(qfeature) #qfeat 维度：1* 2048
		sim_ids, sim_score= web.searcher.search_near_k(qfeat , max_hits)
		search_scores = sim_score.squeeze().tolist()
		self.rank = []
		for i in range(len(sim_ids)):
			video_id_scores = {}
			video_id_scores['id'] = sim_ids[i]
			video_id_scores['score'] = search_scores[i]
			self.rank.append(video_id_scores)
		resp['ret']['overall_ranking'] = self.rank
		total_pages = math.ceil(len(resp['ret']['overall_ranking'])/self.samples_per_page)
		resp['page_info']['total_pages'] = total_pages
		resp['page_info']['pages'] = list(range(resp['page_info']['cur_page'], min(total_pages, resp['page_info']['cur_page'] + 4) + 1))

		s = (resp['page_info']['cur_page'] - 1) * self.samples_per_page
		e = s + self.samples_per_page
		resp['ret']['ranks'] = resp['ret']['overall_ranking'][s:e]
		print(resp)
		return render.upload(resp=resp)


app = web.application(urls, globals())