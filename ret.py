import os
import web
import json
import math
import random
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from collections import OrderedDict
import h5py
import numpy as np
render = web.template.render('templates/', base='index')
import time
import langid
import faiss
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

urls = (
	'/search/', 'search',
	'', 're_ret',
	'/', 'ret'
)

resp = {
	'status': 0,
	'hangout': dict(),
	'ret': dict(),
	'model_list': list(),
	'cur_model': None,
	'page_info': dict()
}

# for file in os.listdir('static/ret'):
#     if file.split('.')[1] == 'json':
#         resp['model_list'].append(file.split('.')[0])
# resp['cur_model'] = resp['model_list'][0]
store_r = None


from CLIP.clip import clip
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
import cn_clip.clip as cn_clip
from cn_clip.clip import load_from_name
cn_model, _ = load_from_name("ViT-B-16", device=device, download_root='./')
cn_model.eval()

class Ranks(object):
	def __init__(self):
		print(f'init Ranks ')
		start_time = time.time()
		self._now = None
		self.q2i = OrderedDict()
		# self.image_clip_feat = h5py.File('/home/zz/code/zz_code/demo/msr_part_clip_feat.hdf5')
		# with h5py.File('/home/zz/code/zz_code/demo/msr_all_clip-B16_feat.hdf5') as f:
		# with h5py.File('/home/zz/code/msr_all_clip_feat.hdf5') as f:
		with h5py.File('/home/hl/code/demo/msr_all_clip_feat.hdf5', 'r') as f:
			self.image_keys = list(f.keys())
		print(f'Loaded image_keys: {time.time() - start_time} s')
		# with h5py.File('/home/zz/code/zz_code/demo/msr_all_cnclip_feat.hdf5') as f:
		with h5py.File('/home/hl/code/demo/msr_all_cnclip_feat.hdf5', 'r') as f:
			self.cn_image_keys = list(f.keys())
		print(f'Loaded cn_image_keys: {time.time() - start_time} s')
		# with h5py.File('/home/zz/code/zz_code/demo/msr_all_clip-B16_feat_concat.h5') as f:
		with h5py.File('/home/hl/code/demo/msr_all_clip_feat_concat.h5', 'r') as f:
			self.all_image_feat = torch.from_numpy(f['all_clip_feat_concat'][:]).to(device)
		print(f'Loaded all_image_feat: {time.time() - start_time} s')
		with h5py.File('/home/hl/code/demo/msr_all_cnclip_feat_concat.h5', 'r') as f:
			self.cn_all_image_feat = torch.from_numpy(f['all_clip_feat_concat'][:]).to(device)
		print(f'Loaded cn_all_image_feat: {time.time() - start_time} s')
		print(f'all_image_feat:{self.all_image_feat.shape}')
		# self.all_image_feat = self.all_image_feat / self.all_image_feat.norm(dim=1, keepdim=True)
		

		
		#####################使用faiss加速检索###############################
		faiss_start = time.time()
		d = 512  # 向量维度
		ngpus = faiss.get_num_gpus()
		print("number of GPUs:", ngpus)
		# en
		if os.path.exists("faissIndex/faissSearchEn.index"):
			cpu_indexEn = faiss.read_index("faissIndex/faissSearchEn.index")
			self.faissSearchEn = faiss.index_cpu_to_all_gpus(  # build the index 转移至GPU
						cpu_indexEn
					)
		else:
			cpu_indexEn = faiss.IndexFlatIP(d)  # 使用内积 建立索引
			self.faissSearchEn = faiss.index_cpu_to_all_gpus(  # build the index 转移至GPU
			cpu_indexEn
		)
			self.faissSearchEn.add(self.all_image_feat.cpu().numpy().astype(np.float32)) # 添加所有的image features到索引中
			faiss.write_index(faiss.index_gpu_to_cpu(self.faissSearchEn), "faissIndex/faissSearchEn.index")
		print(f'Loaded faissSearchEn: {time.time() - faiss_start} s')
		
  		# cn
		if os.path.exists("faissIndex/faissSearchCn.index"):
			cpu_indexCn = faiss.read_index("faissIndex/faissSearchCn.index")
			self.faissSearchCn = faiss.index_cpu_to_all_gpus(  # build the index 转移至GPU
						cpu_indexCn
					)
		else:
			cpu_indexCn = faiss.IndexFlatIP(d)  # 使用内积， 建立索引
			self.faissSearchCn = faiss.index_cpu_to_all_gpus(  # build the index 转移至GPU
			cpu_indexCn
		)	
			self.faissSearchCn.add(self.cn_all_image_feat.cpu().numpy().astype(np.float32)) # 添加所有的cn_image features到索引中
			faiss.write_index(faiss.index_gpu_to_cpu(self.faissSearchCn), "faissIndex/faissSearchCn.index")
		print(f'Loaded faissSearchCn: {time.time() - faiss_start} s')
		#####################使用faiss加速检索###############################
  
  		# 记录代码执行后的时间戳
		end_time = time.time()

		# 计算代码执行所花费的时间
		execution_time = end_time - start_time
		print("初始化ranks代码执行时间: {:.2f} 秒".format(execution_time))
		
		self.search_faiss('init search faiss')

	def load_ranks(self, model):
		if model == self._now:
			return
		self._now = model
		with open(os.path.join('static/ret', model+'.json'), 'r') as f:
			self.ranks = json.load(f)
		for i, r in enumerate(self.ranks):
			self.q2i[r['qid']] = i

		return

	def search_faiss(self, q):
		
		K = 100
		print(q)
		detected = self.detect_language(q)
		start_time = time.time()
		if  detected == 'en':
			text = clip.tokenize([q]).to(device)
			with torch.no_grad():
				text_features = model.encode_text(text)
				# print(f'text_features{text_features.shape}')
				text_features_np = text_features.cpu().numpy().astype(np.float32)

				# 使用Faiss进行搜索
				D, I = self.faissSearchEn.search(text_features_np, K)
				probs = D[0]  
		elif detected == 'zh':
			text = cn_clip.tokenize([q]).to(device)
			with torch.no_grad():
				text_features = cn_model.encode_text(text)
				# print(f'text_features{text_features.shape}')
				text_features_np = text_features.cpu().numpy().astype(np.float32)

				# 使用Faiss进行搜索
				D, I = self.faissSearchCn.search(text_features_np, K)
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
			data.append({'id': self.image_keys[index], 'rank': i+1, 'score': probs[i], 'index': str(i)})

		# 记录代码执行后的时间戳
		end_time = time.time()
		execution_time = end_time - start_time
		print("search_faiss 检索执行时间: {:.2f} 秒".format(execution_time))
		self.search(q)
		return data

	def search(self, q):
		
		print(q)
		detected = self.detect_language(q)
		start_time = time.time()
		if detected == 'en':
			text = clip.tokenize([q]).to(device)
			with torch.no_grad():
				text_features = model.encode_text(text)
				# print(f'text_features{text_features.shape}')
				# text_features = text_features / text_features.norm(dim=1, keepdim=True)


				# cosine similarity as logits
				# logit_scale = model.logit_scale.exp()
				logits_per_image = self.all_image_feat @ text_features.t()
				# print(logits_per_image.shape)
				# logits_per_image = logit_scale * self.all_image_feat @ text_features.t()
				probs = logits_per_image.detach().cpu().numpy()
				probs = np.array([value[0] for value in probs])
		elif detected == 'zh':
			text = cn_clip.tokenize([q]).to(device)
			with torch.no_grad():
				text_features = cn_model.encode_text(text)
				# print(f'text_features{text_features.shape}')
				# text_features = text_features / text_features.norm(dim=1, keepdim=True)


				# cosine similarity as logits
				# logit_scale = model.logit_scale.exp()
				logits_per_image = self.cn_all_image_feat @ text_features.t()
				# print(logits_per_image.shape)
				# logits_per_image = logit_scale * self.all_image_feat @ text_features.t()
				probs = logits_per_image.detach().cpu().numpy()
				probs = np.array([value[0] for value in probs])
		# 获取前 K 个最大值的索引
		# print(probs.size)
		K = 100
		topk_indices = np.argsort(-probs)[:K].tolist()
		# print(topk_indices)
		data = []
		for i,index in enumerate(topk_indices):
			data.append({'id':self.image_keys[index],'rank':i+1,'score':probs[index]})
		# 记录代码执行后的时间戳
		end_time = time.time()

		# 计算代码执行所花费的时间
		execution_time = end_time - start_time

		print("search 检索执行时间: {:.2f} 秒".format(execution_time))
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

ranks = Ranks()
# ranks.load_ranks(resp['cur_model'])


class re_ret:
	def GET(self):
		raise web.seeother('/')

class ret:
	samples_per_page = 10
	def GET(self):
		global resp, ranks
		resp['status'] = 0
		# input_data = web.input(page=None, chosen_model=None)
		# if input_data.chosen_model is not None:
		# 	resp['cur_model'] = input_data.chosen_model
		# 	ranks.load_ranks(resp['cur_model'])

		# resp['page_info']['total_pages'] = math.ceil(len(ranks)/self.samples_per_page)

		# if input_data.page is not None:
		# 	resp['page_info']['cur_page'] = int(input_data.page)
		# else:
		# 	resp['page_info']['cur_page'] = 1

		# left = max(1, resp['page_info']['cur_page'] - 2)
		# if left == 1:
		# 	right = min(resp['page_info']['total_pages'], left + 4)
		# else:
		# 	right = min(resp['page_info']['total_pages'], resp['page_info']['cur_page'] + 2)
		# 	if right == resp['page_info']['total_pages']:
		# 		left = max(1, resp['page_info']['total_pages'] - 4)
		# resp['page_info']['pages'] = list(range(left, right+1))

		# s = (resp['page_info']['cur_page'] - 1) * self.samples_per_page
		# e = s + self.samples_per_page
		# random_ids = ranks.queries[s:e]
		# aps = [f"{ranks.ap(q):.6f}" for q in random_ids]
		# random_ids = ranks.queries[s:e]
		# resp['hangout']['random_queries'] = list(zip(random_ids, aps))

		# resp['hangout']['map'] = f"{ranks.map:.4f}"

		# plt.figure(figsize=(12,3))
		# x = list(range(len(ranks)))
		# aps = ranks.aps
		# plt.bar(x, aps)
		# sio = BytesIO()
		# plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0., transparent=True)
		# data = base64.encodebytes(sio.getvalue()).decode()
		# resp['hangout']['stat_img'] = 'data:image/png;base64,'+str(data)

		return render.ret(resp=resp)

class search:
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

		return render.ret(resp=resp)

	def POST(self):
		print('post请求')

		global resp, ranks,store_r
		resp['status'] = 1

		post_data = web.input(qid=None)
		if post_data.qid is None:
			raise web.seeother('/')

		search_query = post_data.qid
		# print(search_query)
		resp['ret']['qid'] = search_query
		resp['page_info']['cur_page'] = 1

		# r = ranks.search(search_query)
		r = ranks.search_faiss(search_query)
		store_r = r
		total_pages = math.ceil(len(r)/self.samples_per_page)
		resp['page_info']['total_pages'] = total_pages
		resp['page_info']['pages'] = list(range(resp['page_info']['cur_page'], min(total_pages, resp['page_info']['cur_page']+4)+1))

		s = (resp['page_info']['cur_page'] - 1) * self.samples_per_page
		e = s + self.samples_per_page
		resp['ret']['ranks'] = r[s:e]

		resp['ret']['positive'] = r[s:e]
		resp['ret']['ap'] = 0

		

		return render.ret(resp=resp)


app = web.application(urls, globals())




# IndexPQ
		# 创建IndexPQ索引
		# faiss_start = time.time()
		# d = 512  # 向量维度
		# ngpus = faiss.get_num_gpus()
		# print("number of GPUs:", ngpus)
		# self.faissSearchEn = faiss.IndexPQ(d, 8, 8)  # 8位量化

		# # 训练索引（必要步骤，量化器需要先进行训练）
		# self.faissSearchEn.train(self.all_image_feat.cpu().numpy().astype(np.float32))

		# # 将数据添加到索引
		# self.faissSearchEn.add(self.all_image_feat.cpu().numpy().astype(np.float32))
		# print(f'Loaded faissSearchEn: {time.time() - faiss_start} s')
  
  
		#  HNSW (Hierarchical Navigable Small World Graph)
	# 	faiss_start = time.time()
	# 	d = 512  # 向量维度
	# 	ngpus = faiss.get_num_gpus()
	# 	print("number of GPUs:", ngpus)
	# 	self.faissSearchEn = faiss.IndexHNSWFlat(d, 32)  # 32 是图中邻居的数目
	# 	self.faissSearchEn.metric_type = faiss.METRIC_INNER_PRODUCT
	# # 	self.faissSearchEn = faiss.index_cpu_to_all_gpus(  # build the index 转移至GPU
	# # 	index
	# # )
	# 	self.faissSearchEn.add(self.all_image_feat.cpu().numpy().astype(np.float32)) # 添加所有的image features到索引中
	# 	print(f'Loaded faissSearchEn: {time.time() - faiss_start} s')
		
		
		# IndexIVFFlat (Inverted File with Flat Quantization)
		# faiss_start = time.time()
		# d = 512  # 向量维度
		# nlist = 100  # 聚类中心的数量，选择合适的值
		# ngpus = faiss.get_num_gpus()
		# print("number of GPUs:", ngpus)

		# # Step 1: 创建用于聚类的量化器
		# quantizer = faiss.IndexFlatIP(d)  # 使用内积计算作为聚类量化器

		# # Step 2: 创建 IndexIVFFlat 索引 (使用内积)
		# self.faissSearchEn = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

		# # Step 3: 训练索引（在插入数据之前）
		# # 注意: IndexIVFFlat 需要先训练
		# self.faissSearchEn.train(self.all_image_feat.cpu().numpy().astype(np.float32))
		# print("Training complete")

		# # Step 4: 转移索引到 GPU 上
		# # self.faissSearchEn = faiss.index_cpu_to_all_gpus(cpu_indexEn)  # build the index and transfer to GPU

		# # Step 5: 添加数据到索引中
		# self.faissSearchEn.add(self.all_image_feat.cpu().numpy().astype(np.float32))  # 添加所有的image features到索引中
		# print(f'Loaded faissSearchEn: {time.time() - faiss_start} s')