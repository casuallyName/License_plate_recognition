# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/2/6 下午12:13
# @Author: Casually
# @File: Server.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm

from _collections import OrderedDict
# 导入Flask类
from flask import Flask, request, jsonify
from json_utils import jsonify
import numpy as np
import cv2
import time
from collections import OrderedDict
from Recognition import PlateRecognition

# 实例化
app = Flask(__name__)
PR = PlateRecognition()

# 设置编码-否则返回数据中文时候-乱码
app.config['JSON_AS_ASCII'] = False


# route()方法用于设定路由；类似spring路由配置
@app.route('/', methods=['POST'])  # 在线识别
def forecast():
    # 获取输入数据
    stat = time.time()
    file = request.files['image']
    img_bytes = file.read()
    image = np.asarray(bytearray(img_bytes), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    RES = PR.VLPR(image)
    if RES is not None:
        result = OrderedDict(
            Error=0,
            Errmsg='success',
            InputTime=RES['InputTime'],
            UseTime='{:.2f}'.format(time.time() - stat),  # RES['UseTime'],
            Number=RES['Number'],
            From=RES['From'],
            Type=RES['Type'],
            List=RES['List'])
    else:
        result = OrderedDict(
            Error=1,
            Errmsg='unsuccess')
    return jsonify(result)


if __name__ == '__main__':
    # app.run(host, port, debug, options)
    # 默认值：host=127.0.0.1（localhost）, port=5000, debug=false
    app.run()

    # 本地路由地址，局域网下的主机均可通过该地址完成POST请求
    # app.run(host='192.168.1.100' )

    # 部署到服务器
    # from waitress import serve
    # serve(app, host=' IP  ', port=5000)

