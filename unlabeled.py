import os
import web
import math
import json
import random
from collections import defaultdict


render = web.template.render('templates/', base='index')

urls = (
	'', 're_unlabeled',
	'/(.*)', 'unlabeled'
)

idlist = []
with open('/home/zz/code/zz_code/demo/static/unlabeled/unlabeled-data-id', 'r') as f:
	for l in f:
		l = l.strip().split('.')[0]
		idlist.append(l)
idlist.sort()
resp = {}

def reset():
	global resp
	resp = {}

class re_unlabeled:
	def GET(self):
		raise web.seeother('/')

class unlabeled:
	samples_per_page = 24
	def GET(self, name):
		global resp
		reset()

		resp['total_pages'] = math.ceil(len(idlist)/self.samples_per_page)
		input_data = web.input()
		if not input_data:
			resp['cur_page'] = 1
		else:
			resp['cur_page'] = int(input_data.page)

		left = max(1, resp['cur_page'] - 2)
		if left == 1:
			right = min(resp['total_pages'], left + 4)
		else:
			right = min(resp['total_pages'], resp['cur_page'] + 2)
			if right == resp['total_pages']:
				left = max(1, resp['total_pages'] - 4)
		resp['page'] = list(range(left, right+1))

		s = (resp['cur_page'] - 1) * self.samples_per_page
		e = s + self.samples_per_page
		resp['ids'] = idlist[s:e]

		return render.unlabeled(resp=resp)

app = web.application(urls, globals())