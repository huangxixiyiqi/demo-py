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
render = web.template.render('templates/', base='index')
import time
import langid
import faiss
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

urls = (
	'/searchi2i/', 'searchi2i',
	'', 're_i2i',
	'/', 'i2i'
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
class Ranksi2i(object):
	def __init__(self, datasetType = "msr"):
		print(f'init Ranks ')
		print("datasetType:", datasetType)
		self.datasetType = datasetType
		start_time = time.time()
		self._now = None
		self.q2i = OrderedDict()
		if self.datasetType == "msr":
			with h5py.File('image_features/msr_all_clip_feat.hdf5', 'r') as f:
				self.image_keys = list(f.keys())
			print(f'Loaded image_keys: {time.time() - start_time} s')
			
			with h5py.File('image_features/msr_all_clip_feat_concat.h5', 'r') as f:
				self.all_image_feat = torch.from_numpy(f['all_clip_feat_concat'][:]).to(device)
			print(f'Loaded all_image_feat: {time.time() - start_time} s')
			
			with h5py.File('image_features/msr_all_cnclip_feat.hdf5', 'r') as f:
				self.cn_image_keys = list(f.keys())
			print(f'Loaded cn_image_keys: {time.time() - start_time} s')
   
			with h5py.File('image_features/msr_all_cnclip_feat_concat.h5', 'r') as f:
				self.cn_all_image_feat = torch.from_numpy(f['all_clip_feat_concat'][:]).to(device)
			print(f'Loaded cn_all_image_feat: {time.time() - start_time} s')
			
			print(f'all_image_feat:{self.all_image_feat.shape}')

		if self.datasetType == "ali":
     		# 读取 JSON 文件
			with open('static/ali_data/id2image.json', 'r') as json_file:
				image_dicts = json.load(json_file)
				self.image_keys = np.array(list(image_dicts.values()))
    
			print(f'Loaded image_keys: {time.time() - start_time} s')
			
			with h5py.File('image_features/ali_all_clip-B32_feat_concate.hdf5', 'r') as f:
				self.all_image_feat = torch.from_numpy(f['all_clip_feat_concat'][:]).to(device)
			print(f'Loaded all_image_feat: {time.time() - start_time} s')
			

			# 读取 JSON 文件
			with open('static/ali_data/id2image.json', 'r') as json_file:
				image_dicts = json.load(json_file)
				self.cn_image_keys = np.array(list(image_dicts.values()))
    
			print(f'Loaded cn_image_keys: {time.time() - start_time} s')
   
			with h5py.File('image_features/ali_all_cnclip-B16_feat_concate.hdf5', 'r') as f:
				self.cn_all_image_feat = torch.from_numpy(f['all_clip_feat_concat'][:]).to(device)
			print(f'Loaded cn_all_image_feat: {time.time() - start_time} s')
			
   
			print(f'all_image_feat:{self.all_image_feat.shape}')

		if self.datasetType == 'jd':
			# 读取 JSON 文件
			with open('static/JD_data/Index2Image.json', 'r') as json_file:
				image_dicts = json.load(json_file)
				self.image_keys = np.array(list(image_dicts.values()))
    
			print(f'Loaded image_keys: {time.time() - start_time} s')
			
			with h5py.File('image_features/jd_all_clip-B32_feat_concate.hdf5', 'r') as f:
				self.all_image_feat = torch.from_numpy(f['all_clip_feat_concat'][:]).to(device)
			print(f'Loaded all_image_feat: {time.time() - start_time} s')
			

			# 读取 JSON 文件
			with open('static/JD_data/Index2Image.json', 'r') as json_file:
				image_dicts = json.load(json_file)
				self.cn_image_keys = np.array(list(image_dicts.values()))
    
			print(f'Loaded cn_image_keys: {time.time() - start_time} s')
   
			with h5py.File('image_features/jd_all_cnclip-B16_feat_concate.hdf5', 'r') as f:
				self.cn_all_image_feat = torch.from_numpy(f['all_clip_feat_concat'][:]).to(device)
			print(f'Loaded cn_all_image_feat: {time.time() - start_time} s')
			
   
			print(f'all_image_feat:{self.all_image_feat.shape}')

		#####################使用faiss加速检索###############################
		faiss_start = time.time()
		d = 512  # 向量维度
		ngpus = faiss.get_num_gpus()
		print("number of GPUs:", ngpus)
		# en
		res = faiss.StandardGpuResources()
		gpu_id = 0  # 这里可以设置为你想要的 GPU ID，比如 0 或 1

		# 设置配置，将索引放到指定 GPU 上
		flat_config = faiss.GpuIndexFlatConfig()
		flat_config.device = gpu_id  # 指定 GPU ID
		if os.path.exists(f"faissIndex/{datasetType}_faissSearchEn.index"):
			cpu_indexEn = faiss.read_index(f"faissIndex/{datasetType}_faissSearchEn.index")
			# 将 CPU 索引转移至指定的 GPU 上
			self.faissSearchEn = faiss.index_cpu_to_gpu(res, gpu_id, cpu_indexEn)
		else:
			cpu_indexEn = faiss.IndexFlatIP(d)  # 使用内积 建立索引
			self.faissSearchEn = faiss.GpuIndexFlatIP(res, d, flat_config)
			self.faissSearchEn.add(self.all_image_feat.cpu().numpy().astype(np.float32)) # 添加所有的image features到索引中
			faiss.write_index(faiss.index_gpu_to_cpu(self.faissSearchEn), f"faissIndex/{datasetType}_faissSearchEn.index")
		print(f'Loaded faissSearchEn: {time.time() - faiss_start} s')
		
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
		if os.path.exists(f"faissIndex/{datasetType}_faissSearchCn.index"):
			cpu_indexCn = faiss.read_index(f"faissIndex/{datasetType}_faissSearchCn.index")
			# 将 CPU 索引转移至指定的 GPU 上
			self.faissSearchCn = faiss.index_cpu_to_gpu(res, gpu_id, cpu_indexCn)
		else:
			cpu_indexCn = faiss.IndexFlatIP(d)  # 使用内积， 建立索引
			self.faissSearchCn = faiss.GpuIndexFlatIP(res, d, flat_config)
			self.faissSearchCn.add(self.cn_all_image_feat.cpu().numpy().astype(np.float32)) # 添加所有的cn_image features到索引中
			faiss.write_index(faiss.index_gpu_to_cpu(self.faissSearchCn), f"faissIndex/{datasetType}_faissSearchCn.index")
		print(f'Loaded faissSearchCn: {time.time() - faiss_start} s')
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
			img_features = clip_model.encode_image(img_tensor)

			img_features_np = img_features.cpu().numpy().astype(np.float32)

			# 使用Faiss进行搜索
			D, I = self.faissSearchEn.search(img_features_np, K)
			probs = D[0]  
		

		# 获取前 K 个最大值的索引（已经通过Faiss得出）
		topk_indices = I[0].tolist()

		# 构建返回数据
		data = []
		# cars = [3878 , 2515, 2987, 3910, 2628, 3034, 6443, 1890, 372, 873]
		# cars = [6836 , 4539, 469, 6811, 6775, 6622, 1977, 446, 5579, 4207]
		for i, index in enumerate(topk_indices):
			# data.append({'id': self.image_keys[index], 'rank': i+1, 'score': probs[i]})
			# random_number = random.randint(1, 10000)
			# if i < 10:
			# 	random_number = cars[i] 
			if self.datasetType == 'msr':
				image_ids = 'msr_images/'+  self.image_keys[index]
			if self.datasetType == 'ali':		
				image_ids = 'ali_data/'+  self.image_keys[index]
			if self.datasetType == 'jd':		
				image_ids = 'JD_data/Images/'+  self.image_keys[index]
			data.append({'id': image_ids, 'rank': i+1, 'score': probs[i], 'index': str(i)})
		
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
			img_features = clip_model.encode_image(img_tensor)
			logits_per_image = self.all_image_feat @ img_features.t()
			probs = logits_per_image.detach().cpu().numpy()
			probs = np.array([value[0] for value in probs])
		
		# 获取前 K 个最大值的索引
		# print(probs.size)
		K = 100
		topk_indices = np.argsort(-probs)[:K].tolist()
		# print(topk_indices)
		data = []
		for i,index in enumerate(topk_indices):
			if self.datasetType == 'msr':
				image_ids = ['msr_images/'+ value  for value in self.image_keys[index]]
			if self.datasetType == 'ali':		
				image_ids = ['ali_data/'+ value  for value in self.image_keys[index]]
			if self.datasetType == 'jd':		
				image_ids = 'JD_data/Images/'+  self.image_keys[index]
			data.append({'id':image_ids,'rank':i+1,'score':probs[index]})
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

ranks = Ranksi2i(datasetType=datasetType)
# ranks.load_ranks(resp['cur_model'])


class re_i2i:
	def GET(self):
		raise web.seeother('/')

class i2i:
	samples_per_page = 10
	def GET(self):
		global resp, ranks
		resp['status'] = 0
		return render.i2i(resp=resp)

class searchi2i:
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

		return render.i2i(resp=resp)

	def POST(self):
		print('post请求')
		starttime = time.time()
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
		# ranks.search(image)
		print("post 请求响应时间为{:.3f} 秒".format(endtime-starttime))
		return render.i2i(resp=resp)


app = web.application(urls, globals())



