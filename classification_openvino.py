#!/usr/bin/env python
"""
參考 https://docs.openvinotoolkit.org/latest/_inference_engine_ie_bridges_python_sample_classification_sample_README.html

依據 OpenVINO 預設路徑下
"C:\Program Files (x86)\IntelSWTools\openvino_2019.2.242\inference_engine\samples\python_samples\classification_sample\classification_sample.py" 修改

預設模型下載後會存放在預設路徑下C:/Users/jack_/Documents/Intel/OpenVINO/openvino_models/ir/FP16/classification/squeezenet/1.1/caffe/

"""
from __future__ import print_function
import cv2
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore

time_0 = time.clock()

#指定IR檔(xml, bin)
model_xml = "squeezenet1.1.xml"
model_bin = "squeezenet1.1.bin"

#建立推論引擎
ie = IECore()

#載入模型
net = IENetwork(model=model_xml, weights=model_bin)

#準備輸入、輸出空間
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
net.batch_size = 1

time_1 = time.clock()

#讀取和預處理影像
n, c, h, w = net.inputs[input_blob].shape #取得批次數量、通道數及影像高、寬
image = cv2.imread("car.png") # 指定輸入影像
image = cv2.resize(image, (227, 227)) # 統一縮至227x277 for Squeezenet
image = image.transpose((2, 0, 1))    # 變更資料格式從 HWC 到 CHW

time_2 = time.clock()

#載入模型到指定裝置
exec_net = ie.load_network(network=net, device_name="CPU") # 裝置名稱 CPU, GPU, MYRIAD

time_3 = time.clock()

#開始同步模式推論
res = exec_net.infer(inputs={input_blob: image}) # 得到推論結果

time_4 = time.clock()

#處理輸出結果，取出1000類機率結果，陣列大小[1,1000,1,1]
res = res[out_blob]

# 指定標籤檔(可省略)
f = open("squeezenet1.1.labels", 'r') # 開啟ImageNet 1000分類標籤檔
labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f] # 將標籤分割到陣列

classid_str = "classid"
probability_str = "probability"
label_str = "label"
number_top = 5 # 取機率前五名

for i, probs in enumerate(res):
    probs = np.squeeze(probs) 
    top_ind = np.argsort(probs)[-number_top:][::-1] # 排序機率值

    print(classid_str, probability_str, label_str)
    print("{} {} {}".format('-' * len(classid_str), '-' * len(probability_str), '-' * len(label_str)))

    # 顯示機率前五名id、機率值及標籤說明
    for id in top_ind:
        print("{}{}{:.7f}{}{}".format(id, '     ', probs[id], '   ', labels_map[id]))

    print("\n")

time_5 = time.clock()

# 顯示各分段執行時間
print("inital time = {:.4f} sec.".format(time_1 - time_0))
print("load image time = {:.4f} sec.".format(time_2 - time_1))
print("load model time = {:.4f} sec.".format(time_3 - time_2))
print("inference time = {:.4f} sec.".format(time_4 - time_3))
print("display time = {:.4f} sec.".format(time_5 - time_4))
print("total time = {:.4f} sec.".format(time_5 - time_0))
