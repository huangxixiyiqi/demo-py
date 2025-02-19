import os
import web
import json
import math
import random
import base64
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from collections import OrderedDict
import h5py
import numpy as np
render = web.template.render('templates/', base='index',globals={'json': json})
import time
import langid
import faiss
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

urls = (
	'/searchi2t/', 'searchi2t',
	'/searchi2t_data/', 'searchi2t_data',
	'', 're_i2t',
	'/', 'i2t'
)

resp = {
	'status': 0,
	'hangout': dict(),
	'ret': dict(),
	'model_list': list(),
	'cur_model': None,
	'page_info': dict()
}


store_r = None

from init_model import *
class Ranksi2t(object):
	def __init__(self, datasetType = "msr"):
		print(f'init Ranks ')
		print("datasetType:", datasetType)
		self.datasetType = datasetType
		start_time = time.time()
		self._now = None
		self.q2i = OrderedDict()

		if self.datasetType == 'jd':
			
			# cn
			self.id2text = {}
			self.all_text = []
			# 读取 txt 文件
			with open('./static/JD_data/jd_text_name.txt', 'r') as f:
				for line in f.readlines():
					line = line.strip()
					cid = line.split(' ', 1)[0]
					cap = line.split(' ', 1)[1]
					self.id2text[cid] = cap
					self.all_text.append(cap)
			print(f'Loaded all_text: {time.time() - start_time} s')
   
			with h5py.File('./image_features/jd_all_text_name_cnclip-B16_feat_concate.hdf5', 'r') as f:
				self.cn_all_text_feat = torch.from_numpy(f['all_clip_feat_concat'][:]).to(device)
			print(f'Loaded cn_all_text_feat: {time.time() - start_time} s')
			
   
			print(f'all_text_feat:{self.cn_all_text_feat.shape}')

		#####################使用faiss加速检索###############################
		faiss_start = time.time()
		d = 512  # 向量维度
		ngpus = faiss.get_num_gpus()
		print("number of GPUs:", ngpus)
		
		
  		# cn
    	# 初始化 GPU 资源并指定设备
		res = faiss.StandardGpuResources()
		if torch.cuda.device_count()>1:   # 使用ali_data 需要两张卡，一张卡20G，一张10G才能跑起来
			print(f"可用 GPU 数量：{torch.cuda.device_count()}")
			gpu_id = 1  # 这里可以设置为你想要的 GPU ID，比如 0 或 1
		else:
			gpu_id = 0
		# 设置配置，将索引放到指定 GPU 上
		flat_config = faiss.GpuIndexFlatConfig()
		flat_config.device = gpu_id  # 指定 GPU ID
		if os.path.exists(f"faissIndex/{datasetType}_text_faissSearchCn.index"):
			cpu_indexCn = faiss.read_index(f"faissIndex/{datasetType}_text_faissSearchCn.index")
			# 将 CPU 索引转移至指定的 GPU 上
			self.faissSearchCn = faiss.index_cpu_to_gpu(res, gpu_id, cpu_indexCn)
		else:
			cpu_indexCn = faiss.IndexFlatIP(d)  # 使用内积， 建立索引
			self.faissSearchCn = faiss.GpuIndexFlatIP(res, d, flat_config)
			self.faissSearchCn.add(self.cn_all_text_feat.cpu().numpy().astype(np.float32)) # 添加所有的cn_image features到索引中
			faiss.write_index(faiss.index_gpu_to_cpu(self.faissSearchCn), f"faissIndex/{datasetType}_text_faissSearchCn.index")
		print(f'Loaded text_faissSearchCn: {time.time() - faiss_start} s')
		#####################使用faiss加速检索###############################
  
  		# 记录代码执行后的时间戳
		end_time = time.time()

		# 计算代码执行所花费的时间
		execution_time = end_time - start_time
		print("初始化ranks代码执行时间: {:.3f} 秒".format(execution_time))
		img_test = Image.open('./static/uploaded.jpg')
		self.search_faiss(img_test)

	

	def search_faiss(self, q):
		
		K = 100

		start_time = time.time()

		img_tensor = preprocess(q).unsqueeze(0).to(device)
		with torch.no_grad():
			img_features = cnclip_model.encode_image(img_tensor)

			img_features_np = img_features.cpu().numpy().astype(np.float32)

			# 使用Faiss进行搜索
			D, I = self.faissSearchCn.search(img_features_np, K)
			probs = D[0]  
		

		# 获取前 K 个最大值的索引（已经通过Faiss得出）
		topk_indices = I[0].tolist()

		# 构建返回数据
		data = []
		deep_min, deep_max = probs[len(topk_indices)-1], probs[0]
		for i, index in enumerate(topk_indices):
			if self.datasetType == 'jd':		
				cap = self.all_text[index]
				product = products[index]
				deep_model_scores = probs[i]
				deep_model_scores_norm = (deep_model_scores - deep_min) / (deep_max - deep_min)
			data.append({'id': cap,'product': product,'rank': i+1, 'score': deep_model_scores_norm, 'index': str(i)})
		
		# 记录代码执行后的时间戳
		end_time = time.time()
		execution_time = end_time - start_time
		print("search_faiss 检索执行时间: {:.3f} 秒".format(execution_time))
		return data

	def search(self, q):
		
		print(q)
		
		start_time = time.time()
		
		img_tensor = preprocess(q).unsqueeze(0).to(device)
		with torch.no_grad():
			img_features = cnclip_model.encode_image(img_tensor)
			logits_per_image = self.cn_all_text_feat @ img_features.t()
			probs = logits_per_image.detach().cpu().numpy()
			probs = np.array([value[0] for value in probs])
		
		# 获取前 K 个最大值的索引
		# print(probs.size)
		K = 100
		topk_indices = np.argsort(-probs)[:K].tolist()
		# print(topk_indices)
		data = []
		for i,index in enumerate(topk_indices):
			if self.datasetType == 'jd':		
				cap = self.all_text[index]
			data.append({'id': cap, 'rank': i+1, 'score': probs[i], 'index': str(i)})
		# 记录代码执行后的时间戳
		end_time = time.time()

		# 计算代码执行所花费的时间
		execution_time = end_time - start_time

		print("search 检索执行时间: {:.3f} 秒".format(execution_time))
		return data
		# return self.ranks[self.q2i[q]]

	def ap(self, q):
		return self.ranks[self.q2i[q]]['ap']['top-inf']
		# return self.ranks[self.q2i[q]]['ap']
	

	def detect_language(self,text):
		startTime = time.time()
		result = langid.classify(text)
		print(f"detect_language: {time.time()- startTime}s")
		return result[0]

	@property
	def map(self):
		return sum(self.aps) / len(self.aps)

	@property
	def aps(self):
		return [i['ap']['top-inf'] for i in self.ranks]

	@property
	def now(self):
		return self._now

	@property
	def queries(self):
		return [r['qid'] for r in self.ranks]

	def __len__(self):
		return 1
		# return len(self.ranks)

ranks = Ranksi2t(datasetType=datasetType)
# ranks.load_ranks(resp['cur_model'])


class re_i2t:
	def GET(self):
		raise web.seeother('/')

class i2t:
	samples_per_page = 10
	def GET(self):
		global resp, ranks
		resp['status'] = 0
		return render.i2t(resp=resp)

class searchi2t:
	samples_per_page = 10
	def GET(self):
		print('get请求')
		global resp, ranks
		input_data = web.input(page=None)
		if not resp['status']:
			raise web.seeother('/')

		if input_data.page:
			resp['page_info']['cur_page'] = int(input_data.page)

		s = (resp['page_info']['cur_page'] - 1) * self.samples_per_page
		e = s + self.samples_per_page
		# r = ranks.search(resp['ret']['qid'])
		if store_r is None:
			print('dshak')
			r = ranks.search(resp['ret']['qid'])
		else:
			r = store_r
		# resp['ret']['ranks'] = r[s:e]
		resp['ret']['positive'] = r[s:e]
		total_pages = math.ceil(len(r)/self.samples_per_page)
		left = max(1, resp['page_info']['cur_page'] - 2)
		if left == 1:
			right = min(total_pages, left + 4)
		else:
			right = min(total_pages, resp['page_info']['cur_page'] + 2)
			if right == total_pages:
				left = max(1, total_pages - 4)
		resp['page_info']['pages'] = list(range(left, right+1))

		return render.i2t(resp=resp)

	def POST(self):
		print('post请求')
		starttime  = time.time()
		global resp, ranks,store_r
		resp['status'] = 1

		 # 获取上传的图片文件
		post_data = web.input(image=None)

		if post_data.image is None:
			raise web.seeother('/')

		# 获取上传文件的内容
		image_data = post_data.image
		image = Image.open(BytesIO(image_data)).convert('RGB')
		image.save('./static/uploaded.jpg')
		resp['image_path'] = '/static/uploaded.jpg'
		resp['ret']['qid'] = image
		resp['page_info']['cur_page'] = 1


		r = ranks.search_faiss(image)
		store_r = r
		total_pages = math.ceil(len(r)/self.samples_per_page)
		resp['page_info']['total_pages'] = total_pages
		resp['page_info']['pages'] = list(range(resp['page_info']['cur_page'], min(total_pages, resp['page_info']['cur_page']+4)+1))

		s = (resp['page_info']['cur_page'] - 1) * self.samples_per_page
		e = s + self.samples_per_page
		resp['ret']['ranks'] = r[s:e]

		resp['ret']['positive'] = r[s:e]
		resp['ret']['ap'] = 0

		endtime = time.time()
		# ranks.search(search_query)
		print("post 请求响应时间为{:.3f} 秒".format(endtime-starttime))
		return render.i2t(resp=resp)

class searchi2t_data:
    samples_per_page = 10

    def GET(self):
        web.header('Content-Type', 'application/json')
        global resp, ranks
        input_data = web.input(page=None)

        if not resp['status']:
            return json.dumps({'error': 'Not initialized'})

        if input_data.page:
            resp['page_info']['cur_page'] = int(input_data.page)

        s = (resp['page_info']['cur_page'] - 1) * self.samples_per_page
        e = s + self.samples_per_page

        if store_r is None:
            r = ranks.search(resp['ret']['qid'])
        else:
            r = store_r

        resp['ret']['positive'] = r[s:e]

        # 构建JSON响应
        response_data = {
            'status': resp['status'],
            'data': resp['ret'],
            'page_info': resp['page_info']
        }

        return json.dumps(response_data)

    def POST(self):
        web.header('Content-Type', 'application/json')
        starttime = time.time()
        global resp, ranks, store_r
        resp['status'] = 1

        try:
            post_data = web.input(image=None, curpage=None)

            if post_data.image is None:
                return json.dumps({'error': 'No image uploaded'})

            image_data = post_data.image
            image = Image.open(BytesIO(image_data)).convert('RGB')
            image.save('./static/uploaded.jpg')

            resp['image_path'] = '/static/uploaded.jpg'
            resp['ret']['qid'] = image
            resp['page_info']['cur_page'] = post_data.curpage

            r = ranks.search_faiss(image)
            store_r = r

            total_pages = math.ceil(len(r) / self.samples_per_page)
            resp['page_info']['total_pages'] = total_pages
            resp['page_info']['pages'] = list(range(1, min(total_pages, 5) + 1))

            s = (int(post_data.curpage)-1) * self.samples_per_page
            e = s + self.samples_per_page
            for i in range(len(r)):
                r[i]['score'] = str(r[i]['score'])
            # 构建JSON响应
            response_data = {
                'status': 'success',
                'data': {
                    'results': r[s:e],
                    'total_pages': total_pages,
                    'current_page': resp['page_info']['cur_page'],
                }
            }

            return json.dumps(response_data)

        except Exception as e:
            return json.dumps({
                'status': 'error',
                'message': str(e)
            })


app = web.application(urls, globals())



