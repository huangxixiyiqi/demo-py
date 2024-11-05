import web
import ret
# import grouping
# import unlabeled
# import upload

render = web.template.render('templates/')


urls = (
	'/ret', ret.app,
	# '/grouping', grouping.app,
	# '/unlabeled', unlabeled.app,
	# '/upload', upload.app,
	'/(.*)', 'index'
)

class index:
	def GET(self, name):
		return render.index(content=None)

if __name__ == '__main__':
	app = web.application(urls, globals())
	
	
	app.run()