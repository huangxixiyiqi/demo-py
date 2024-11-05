import os
import web
import math
import json
import random
from collections import defaultdict


render = web.template.render('templates/', base='index')

urls = (
	'', 're_grouping',
	'/(.*)', 'grouping'
)

gs = defaultdict(list)
for file in os.listdir('/home/zz/code/zz_code/demo/static/grouping'):
	model = file.split('.')[0] 
	with open(os.path.join('/home/zz/code/zz_code/demo/static/grouping', file), 'r') as f:
		for l in f:
			l = l.strip().split()
			gs[model].append(l)
	gs[model].sort(key=lambda x: len(x), reverse=True)
	# random.shuffle(gs[model])

models = {'model_list': list(gs.keys()), 'cur_model': list(gs.keys())[0]}

resp = dict()

def reset():
	global resp
	resp = dict()
	resp['page_info'] = dict()

class re_grouping:
	def GET(self):
		raise web.seeother('/')

class grouping:
	samples_per_page = 6
	n_per_sample = 10
	def GET(self, name):
		global models, resp
		reset()

		input_data = web.input(page=None, chosen_model=None)

		if input_data and input_data.chosen_model:
			models['cur_model'] = input_data.chosen_model

		resp['page_info']['total_pages'] = math.ceil(len(gs[models['cur_model']])/self.samples_per_page)
		
		if input_data and input_data.page:
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
		resp['page_info']['page'] = list(range(left, right+1))

		s = (resp['page_info']['cur_page'] - 1) * self.samples_per_page
		e = s + self.samples_per_page
		
		resp['groups'] = gs[models['cur_model']][s:e]

		for i, group in enumerate(resp['groups']):
			if len(group) > self.n_per_sample:
				resp['groups'][i] = random.sample(resp['groups'][i], self.n_per_sample)

		return render.grouping(models=models, resp=resp)

app = web.application(urls, globals())