'''
Author: huangxixi huangxixiyiqi@gmail.com
Date: 2024-12-03 16:55:55
LastEditors: huangxixi huangxixiyiqi@gmail.com
LastEditTime: 2025-02-19 10:07:02
FilePath: /demo/whoosh_text.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from whoosh.fields import TEXT, SchemaClass
from jieba.analyse import ChineseAnalyzer
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser, MultifieldParser
import os
analyzer = ChineseAnalyzer()
class ArticleSchema(SchemaClass):
    vid = TEXT(stored=True, analyzer=analyzer)
    content = TEXT(stored=True, analyzer=analyzer)
    
schema = ArticleSchema()
indexdir = "indexdir/"
if not os.path.exists(indexdir):
    os.mkdir(indexdir)
    
# ix = create_in(indexdir, schema, indexname='ali_ocrtext_index')
# writer = ix.writer()
# with open("/home/hl/code/demo/static/ali_data/ocr_text.txt", 'r') as f:
#     for line in f.readlines():
#         vid = line.split("#")[0]
#         cap = line.split(" ", 1)[1].strip()
#         writer.add_document(vid=vid, content=cap)
# writer.commit()

ix = open_dir("indexdir", indexname='ali_ocrtext_index')
# with ix.searcher() as searcher:
#     results = searcher.find("content", "盖中盖")
#     print(results[0])
with ix.searcher() as searcher:
    query = QueryParser("content", ix.schema).parse("盖中盖")
    results = searcher.search(query)
    print(results[0])