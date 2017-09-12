#!/usr/bin/env python
# -*- coding:utf-8 -*-
import ConfigParser
import os
import re
import sys

import time
import tornado.ioloop
import tornado.web

from matrixText.matrix_learn import load_model
from matrixText.matrix_text import matrix_similar_src_modeled

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

server_data_dir = '/home/uww/fastText/anyou/'
simi_file_name = 'servectmp.txt'
thread_num = 64

word_vec_dim = 100

seg_word_server_url = 'http://127.0.0.1:8080/SegService/servlet/Segment'

minshi_model = None
xingshi_model = None


def save_to_similar_doc(simi_src, anyou_type, word_vec_dim=100):
    # file_model = server_data_dir + anyou_type + '/wordVecModelAs' + '/cbowAnsjModel' + str(model_id) + '.bin'
    file_vec = server_data_dir + simi_file_name
    if anyou_type == 'minshi':
        matrix_similar_src_modeled(content=str(simi_src), file_vec=file_vec, file_model=minshi_model,
                                   ansj_serve_url=seg_word_server_url, dim=word_vec_dim)
    elif anyou_type == 'xingshi':
        matrix_similar_src_modeled(content=str(simi_src), file_vec=file_vec, file_model=xingshi_model,
                                   ansj_serve_url=seg_word_server_url, dim=word_vec_dim)


# usage demo: 64 100 6 12 /home/uww/fastText/anyou/ vectemp.txt minshi 9131 9130
# Usage java -jar selectSimDoc.jar [thread_num wordVecDim minSententLen maxSententLen dataDir fileName anyouType anyouFile]
def mp_select_similar_docs(mp_thread_num, word_vec_dim, min_sentent_len, max_sentent_len,
                           model_basedir, src_file_name, anyou_type, targetDMs):
    select_cmd = 'java -jar selectSimDoc.jar ' + str(mp_thread_num) + ' ' + str(word_vec_dim) + ' ' + \
                 str(min_sentent_len) + ' ' + str(max_sentent_len) + ' ' + model_basedir + ' ' + \
                 src_file_name + ' ' + anyou_type
    targetDMs = re.compile(r'\d+').findall(targetDMs)
    for id in targetDMs:
        select_cmd += ' ' + str(id)
    print(select_cmd)
    res_cmd = os.popen(select_cmd).readlines()
    res_out = ';'.join(res_cmd)
    return res_out


class SimilarAnyouHandler(tornado.web.RequestHandler):
    def get(self):
        pass

    def post(self):
        simi_src = self.get_argument('simisrc')
        anyou_type = self.get_argument('anyoutype')
        min_sent_len = self.get_argument('minsentlen')
        max_sent_len = self.get_argument('maxsentlen')
        anyou_ids = self.get_argument('anyouids')
        print '---link server---'
        start_time = time.time()
        print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        save_to_similar_doc(simi_src=simi_src, anyou_type=anyou_type, word_vec_dim=word_vec_dim)
        result = mp_select_similar_docs(mp_thread_num=thread_num, word_vec_dim=word_vec_dim,
                                        min_sentent_len=min_sent_len, max_sentent_len=max_sent_len,
                                        model_basedir=server_data_dir, src_file_name=simi_file_name,
                                        anyou_type=anyou_type, targetDMs=anyou_ids)
        print '---complete---'
        print 'Total time running: ', time.time() - start_time
        self.write(str(result))


if __name__ == "__main__":
    print '---init server---'
    cp = ConfigParser.SafeConfigParser()
    cp.read('mp_server.conf')
    server_port = cp.get('server', 'port')
    server_data_dir = cp.get('server', 'data_dir')
    thread_num = cp.get('server', 'thread_num')
    word_vec_dim = int(cp.get('server', 'word_vec_dim'))
    seg_word_server_url = cp.get('word_seg_server', 'url')

    minshi_model_file = server_data_dir + 'minshi/wordVecModelAs' + '/cbowAnsjModel' + str(9000) + '.bin'
    minshi_model = load_model(minshi_model_file)
    xingshi_model_file = server_data_dir + 'xingshi/wordVecModelAs' + '/cbowAnsjModel' + str(1) + '.bin'
    xingshi_model = load_model(xingshi_model_file)

    print '---start server---'
    application = tornado.web.Application([
        (r'/mpJavaSimilarAnyou', SimilarAnyouHandler)
    ])

    application.listen(server_port)
    tornado.ioloop.IOLoop.instance().start()
