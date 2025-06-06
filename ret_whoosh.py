from init_model import *
import faiss
import langid
import time
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
from whoosh.qparser import QueryParser
render = web.template.render(
    'templates/', base='index', globals={'json': json})
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

urls = (
	'/search/', 'search',
	'/search_data/', 'search_data',
	'/search_data_all/', 'search_data_all',
	'', 're_ret',
	'/', 'ret'
)

resp = {
	'status': 0,
	'hangout': dict(),
	'ret': dict(),
	'model_list': list(),
	'cur_model': None,
	'page_info': dict(),

	'ret_': dict(),
	'page_info_': dict(),
}



store_r = None
store_r1 = None

class Ranks(object):
	def __init__(self, datasetType="msr"):
		print(f'init Ranks ')
		print("datasetType:", datasetType)
		self.datasetType = datasetType
		start_time = time.time()
		self.ix = open_dir("indexdir", indexname='jd_ocrtext_index')
		execution_time = time.time()-start_time
		print("初始化ranks代码执行时间: {:.3f} 秒".format(execution_time))
		if self.datasetType == 'jd':
			# 读取 JSON 文件
			with open('static/JD_data/Index2Image.json', 'r') as json_file:
				image_dicts = json.load(json_file)
				self.image_keys = np.array(list(image_dicts.values()))

			print(f'Loaded image_keys: {time.time() - start_time} s')

			with h5py.File('image_features/jd_all_clip-B32_feat_concate.hdf5', 'r') as f:
				self.all_image_feat = torch.from_numpy(
				    f['all_clip_feat_concat'][:]).to(device)
			print(f'Loaded all_image_feat: {time.time() - start_time} s')

			# 读取 JSON 文件
			with open('static/JD_data/Index2Image.json', 'r') as json_file:
				image_dicts = json.load(json_file)
				self.cn_image_keys = np.array(list(image_dicts.values()))

			print(f'Loaded cn_image_keys: {time.time() - start_time} s')

			with h5py.File('image_features/jd_all_cnclip-B16_feat_concate.hdf5', 'r') as f:
				self.cn_all_image_feat = torch.from_numpy(
				    f['all_clip_feat_concat'][:]).to(device)
			print(f'Loaded cn_all_image_feat: {time.time() - start_time} s')

			print(f'all_image_feat:{self.all_image_feat.shape}')

		##################### 使用faiss加速检索###############################
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
			cpu_indexEn = faiss.read_index(
			    f"faissIndex/{datasetType}_faissSearchEn.index")
			# 将 CPU 索引转移至指定的 GPU 上
			self.faissSearchEn = faiss.index_cpu_to_gpu(res, gpu_id, cpu_indexEn)
		else:
			cpu_indexEn = faiss.IndexFlatIP(d)  # 使用内积 建立索引
			self.faissSearchEn = faiss.GpuIndexFlatIP(res, d, flat_config)
			self.faissSearchEn.add(self.all_image_feat.cpu().numpy().astype(
			    np.float32))  # 添加所有的image features到索引中
			faiss.write_index(faiss.index_gpu_to_cpu(self.faissSearchEn),
			                  f"faissIndex/{datasetType}_faissSearchEn.index")
		print(f'Loaded faissSearchEn: {time.time() - faiss_start} s')

  		# cn
    	# 初始化 GPU 资源并指定设备
		res = faiss.StandardGpuResources()
		if torch.cuda.device_count() > 1:   # 使用ali_data 需要两张卡，一张卡20G，一张10G才能跑起来
			print(f"可用 GPU 数量：{torch.cuda.device_count()}")
			gpu_id = 1  # 这里可以设置为你想要的 GPU ID，比如 0 或 1
		else:
			gpu_id = 0
		# 设置配置，将索引放到指定 GPU 上
		flat_config = faiss.GpuIndexFlatConfig()
		flat_config.device = gpu_id  # 指定 GPU ID
		if os.path.exists(f"faissIndex/{datasetType}_faissSearchCn.index"):
			cpu_indexCn = faiss.read_index(
			    f"faissIndex/{datasetType}_faissSearchCn.index")
			# 将 CPU 索引转移至指定的 GPU 上
			self.faissSearchCn = faiss.index_cpu_to_gpu(res, gpu_id, cpu_indexCn)
		else:
			cpu_indexCn = faiss.IndexFlatIP(d)  # 使用内积， 建立索引
			self.faissSearchCn = faiss.GpuIndexFlatIP(res, d, flat_config)
			self.faissSearchCn.add(self.cn_all_image_feat.cpu().numpy().astype(
			    np.float32))  # 添加所有的cn_image features到索引中
			faiss.write_index(faiss.index_gpu_to_cpu(self.faissSearchCn),
			                  f"faissIndex/{datasetType}_faissSearchCn.index")
		print(f'Loaded faissSearchCn: {time.time() - faiss_start} s')
		##################### 使用faiss加速检索###############################

		self.search_faiss('钙片')
		self.set_data()

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
		# 构建返回数据
		data = {}

		# 模型匹配
		if detected == 'en':
			text = clip.tokenize([q]).to(device)
			with torch.no_grad():
				text_features = clip_model.encode_text(text)
				# print(f'text_features{text_features.shape}')
				text_features_np = text_features.cpu().numpy().astype(np.float32)

				# 使用Faiss进行搜索
				D, I = self.faissSearchEn.search(text_features_np, K)
				probs = D[0]
		else:
			text = cn_clip.tokenize([q]).to(device)
			with torch.no_grad():
				text_features = cnclip_model.encode_text(text)
				# print(f'text_features{text_features.shape}')
				text_features_np = text_features.cpu().numpy().astype(np.float32)

				# 使用Faiss进行搜索
				D, I = self.faissSearchCn.search(text_features_np, K)
				probs = D[0]

		# 获取前 K 个最大值的索引（已经通过Faiss得出）
		topk_indices = I[0].tolist()

		deep_min, deep_max = probs[len(topk_indices)-1], probs[0]
		for i, index in enumerate(topk_indices):
			if self.datasetType == 'jd':
				image_ids = 'JD_data/Images/' + self.image_keys[index]
				if os.path.exists('./static/' + image_ids) == False or self.image_keys[index] not in img2products:
					continue
				product = img2products[self.image_keys[index]]
				deep_model_scores = float(probs[i])
				deep_model_scores_norm = (
				    deep_model_scores - deep_min) / (deep_max - deep_min)
				data[image_ids] = {'id': image_ids, 'product': product, 'rank': len(data)+1,
                       'score': deep_model_scores_norm, 'index': str(len(data)), 'score_model': deep_model_scores_norm, 'score_ocr': 0}
			else:
				if self.datasetType == 'msr':
					image_ids = 'msr_images/' + self.image_keys[index]
				if self.datasetType == 'ali':
					image_ids = 'ali_data/' + self.image_keys[index]
				data.append({'id': image_ids, 'rank': len(data)+1,
				            'score': float(probs[i]), 'index': str(len(data))})

		# 文本匹配
		with self.ix.searcher() as searcher:
			query = QueryParser("content", self.ix.schema).parse(q)
			results = searcher.search(query)
			if len(results) != 0:
				whoosh_min, whoosh_max = results[-1].score, results[0].score
			for i, index in enumerate(results):
				if self.datasetType == 'jd':
					image_ids = 'JD_data/Images/' + f"{index['vid']}.jpg"
					if os.path.exists('./static/' + image_ids) == False or f"{index['vid']}.jpg" not in img2products:
						continue
					product = img2products[f"{index['vid']}.jpg"]
					if len(results) == 1:
						whoosh_scores_norm = 1
					else:
						whoosh_scores_norm = (index.score - whoosh_min) / \
						                      (whoosh_max - whoosh_min)
					if image_ids not in data:
						data[image_ids] = {'id': image_ids, 'product': product, 'rank': i+1, 'score': whoosh_scores_norm,
						    'index': str(i), 'score_model': 0, 'score_ocr': whoosh_scores_norm}
					else:
						s_ocr = whoosh_scores_norm
						s_ = data[image_ids]['score']
						alpha = 0.5
						data[image_ids]['score'] = s_ocr * alpha + (1-alpha) * s_
						data[image_ids]['score_ocr'] = s_ocr

				else:
					if self.datasetType == 'msr':
						image_ids = 'msr_images/' + self.image_keys[index]
					if self.datasetType == 'ali':
						image_ids = 'ali_data/' + self.image_keys[index]

					data.append({'id': image_ids, 'rank': i+1,
					            'score': index.score, 'index': str(i)})

		sorted_data = sorted(list(data.values()),
		                     key=lambda x: x['score'], reverse=True)
		# 更新 'rank' 和 'index'
		for new_rank, item in enumerate(sorted_data, start=1):
			item['rank'] = new_rank
			item['index'] = str(new_rank - 1)
        # 记录代码执行后的时间戳
		end_time = time.time()
		execution_time = end_time - start_time
		print("search_faiss 检索执行时间: {:.3f} 秒".format(execution_time))
		return sorted_data

	def ap(self, q):
		return self.ranks[self.q2i[q]]['ap']['top-inf']
		# return self.ranks[self.q2i[q]]['ap']

	def detect_language(self, text):
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
	
	def set_data(self):
		samples_per_page = 8
		starttime = time.time()
		global resp, ranks, store_r1
		# resp['status'] = 1

		search_query = '钙片'
		search_query1 = '芦荟胶囊'
		search_query2 = '口服液'
		# print(search_query)
		resp['ret_']['qid'] = search_query + " & " + search_query1  + " & " + search_query2# 合并查询标识
		resp['page_info_']['cur_page'] = 1

		# 查询两个关键词的结果
		r = self.search_faiss(search_query)
		r1 = self.search_faiss(search_query1)
		r2 = self.search_faiss(search_query2)
		# 合并两个查询结果
		combined_results = r + r1 + r2

		# 去重：根据 image_ids 去重
		unique_results = {}
		for item in combined_results:
			if item['id'] not in unique_results:
				unique_results[item['id']] = item
			else:
				# 如果有重复项，取评分更高的结果
				if item['score'] > unique_results[item['id']]['score']:
					unique_results[item['id']] = item

		# 转换为列表并按评分排序
		sorted_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
		store_r1 = sorted_results

		# 分页处理
		total_pages = math.ceil(len(sorted_results) / samples_per_page)
		resp['page_info_']['total_pages'] = total_pages
		resp['page_info_']['pages'] = list(range(resp['page_info_']['cur_page'], min(
			total_pages, resp['page_info_']['cur_page'] + 4) + 1))

		s = (resp['page_info_']['cur_page'] - 1) * samples_per_page
		e = s + samples_per_page
		resp['ret_']['ranks'] = sorted_results[s:e]

		resp['ret_']['positive'] = sorted_results[s:e]
		resp['ret_']['ap'] = 0

		endtime = time.time()
		print("setdata 响应时间为{:.3f} 秒".format(endtime - starttime))
		return render.ret(resp=resp)


ranks = Ranks(datasetType=datasetType)
# ranks.load_ranks(resp['cur_model'])


class re_ret:
	def GET(self):
		raise web.seeother('/')


class ret:
	samples_per_page = 8

	def GET(self):
		global resp, ranks, store_r1
		resp['status'] = 0
		
		#为了支持预展示分野
		input_data = web.input(page=None)


		if input_data.page:
			resp['page_info_']['cur_page'] = int(input_data.page)

		s = (resp['page_info_']['cur_page'] - 1) * self.samples_per_page
		e = s + self.samples_per_page
		# r = ranks.search(resp['ret']['qid'])
		if store_r1 is None:
			print('dshak')
			r = ranks.search(resp['ret_']['qid'])
		else:
			r = store_r1
		# resp['ret']['ranks'] = r[s:e]
		resp['ret_']['positive'] = r[s:e]
		total_pages = math.ceil(len(r)/self.samples_per_page)
		left = max(1, resp['page_info_']['cur_page'] - 2)
		if left == 1:
			right = min(total_pages, left + 4)
		else:
			right = min(total_pages, resp['page_info_']['cur_page'] + 2)
			if right == total_pages:
				left = max(1, total_pages - 4)
		resp['page_info_']['pages'] = list(range(left, right+1))
		# 

		return render.ret(resp=resp)


class search:
	samples_per_page = 8

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
		starttime = time.time()
		global resp, ranks, store_r
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
		resp['page_info']['pages'] = list(range(resp['page_info']['cur_page'], min(
		    total_pages, resp['page_info']['cur_page']+4)+1))

		s = (resp['page_info']['cur_page'] - 1) * self.samples_per_page
		e = s + self.samples_per_page
		resp['ret']['ranks'] = r[s:e]

		resp['ret']['positive'] = r[s:e]
		resp['ret']['ap'] = 0
		endtime = time.time()
		# ranks.search(search_query)
		print("post 请求响应时间为{:.3f} 秒".format(endtime-starttime))
		return render.ret(resp=resp)


class search_data:
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
			post_data = web.input(qid=None, curpage=None)
			if post_data.qid is None:
				return json.dumps({'error': 'No query provided'})

			search_query = post_data.qid
			resp['ret']['qid'] = search_query
			resp['page_info']['cur_page'] = post_data.curpage

			r = ranks.search_faiss(search_query)
			store_r = r

			total_pages = math.ceil(len(r)/self.samples_per_page)
			resp['page_info']['total_pages'] = total_pages
			resp['page_info']['pages'] = list(range(1, min(total_pages, 5) + 1))

			s = (int(post_data.curpage)-1) * self.samples_per_page
			e = s + self.samples_per_page

			# 构建JSON响应
			response_data = {
				'status': 'success',
				'data': {
					'results': r[s:e],
					'total_pages': total_pages,
					'current_page': post_data.curpage,
					'query': search_query
				}
			}

			return json.dumps(response_data)

		except Exception as e:
			return json.dumps({
				'status': 'error',
				'message': str(e)
			})


class search_data_all:
	def OPTIONS(self):
        # 设置CORS响应头
		web.header('Access-Control-Allow-Origin', '*')  # 允许所有来源
		web.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')  # 允许的 HTTP 方法
		web.header('Access-Control-Allow-Headers', 'Content-Type')  # 允许的请求头
		return ""  # 返回空字符串
    
	def POST(self):
		web.header('Content-Type', 'application/json; charset=utf-8')
		web.header('Access-Control-Allow-Origin', '*')  # 允许所有来源
		web.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')  # 允许的 HTTP 方法
		web.header('Access-Control-Allow-Headers', 'Content-Type')  # 允许的请求头
		starttime = time.time()
		global resp, ranks, store_r
		resp['status'] = 1

		try:
			post_data = web.input(qid=None)
			if post_data.qid is None:
				return json.dumps({'error': 'No query provided'})

			search_query = post_data.qid
			resp['ret']['qid'] = search_query

			r = ranks.search_faiss(search_query)
	
	

            # 构建JSON响应
			response_data = {
                'status': 'success',
                'data': {
                    'results': r,  # 确保 r 已经是可序列化的
                    'query': search_query
                }
            }
			
			return json.dumps(response_data)
			
		except Exception as e:
			return json.dumps({
				'status': 'error',
				'message': str(e)
			})



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