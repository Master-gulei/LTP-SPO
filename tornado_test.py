#!/usr/bin/env python
# coding=utf-8

from tornado.web import Application, RequestHandler, url
import tornado.ioloop
from tornado.httpserver import HTTPServer
from fact_triple_extraction import in_file_name, out_file_name, extraction_start


# 定义处理类型
class IndexHandler(RequestHandler):
    # 添加一个处理get请求方式的方法
    def get(self):
        self.write("<a href='" + self.reverse_url("parse") + "'>三元组示例入口</a>")


class TestHandler(RequestHandler):
    # 添加一个处理get请求方式的方法
    def get(self):
        self.write("请输入测试句子：")
        self.write("<form action='" + self.reverse_url("parse") + "' method='get'>" +
                   "<input type='text' name='sentence' />" + "<input type='submit' value='Submit' />"
                   )


class ParseHandler(RequestHandler):
    def get(self):
        sentence = self.get_argument("sentence", default=None)
        spo_list = extraction_start().run(in_file_name, out_file_name, sentence)
        if len(spo_list) > 50:
            spo_list = spo_list[:50]
        for item in spo_list:
            for key in item:
                self.write(key)
                for s, p, o in item[key]:
                    self.write("<li>" + s + "—" + p + "—" + o + "</li>")


if __name__ == '__main__':
    # 创建一个应用对象
    app = Application(
        [
            (r"/", IndexHandler),
            (r"/test", TestHandler),
            url(r"/parse", ParseHandler, name="parse"),
        ]
    )

    http_server = HTTPServer(app)
    # 绑定一个监听端口
    app.listen(8888)
    # 启动web程序，开始监听端口的连接
    tornado.ioloop.IOLoop.current().start()
