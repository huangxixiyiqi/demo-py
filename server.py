'''
Author: huangxixi huangxixiyiqi@gmain.com
Date: 2024-11-05 09:38:12
LastEditors: huangxixi huangxixiyiqi@gmain.com
LastEditTime: 2025-02-09 19:40:18
FilePath: /demo/server.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import web
import ret_whoosh as ret
import i2i
import i2t


render = web.template.render('templates/')


urls = (
	'/ret', ret.app,
	'/i2i', i2i.app,
	'/i2t', i2t.app,
	'/(.*)', 'index'
)

class index:
	def GET(self, name):
		return render.index(content=None)

if __name__ == '__main__':
	app = web.application(urls, globals())
	web.httpserver.runsimple(app.wsgifunc(), ("0.0.0.0", 8080))
	app.run()