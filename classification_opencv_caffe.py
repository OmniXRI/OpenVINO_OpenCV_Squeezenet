'''
參考OpenCV官網文件Deep Neural Network (DNN) module說明
https://docs.opencv.org/4.1.1/d6/d0f/group__dnn.html

從3.4.2版後支援直接讀取OpenVINO IR檔函式(cv2.dnn.readNetFromModelOptimizer)
'''

import cv2
from cv2 import dnn
import numpy as np 
import time

time_0 = time.clock()

#指定Caffe格式參數及模型檔(prototxt, caffemodel)
prototxt = "squeezenet_v1.1.prototxt"
caffemodel = "squeezenet_v1.1.caffemodel"

# 載入Caffe模型
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

# 設定後端 DNN_BACKEND_DEFAULT, DNN_BACKEND_HALIDE, DNN_BACKEND_INFERENCE_ENGINE, DNN_BACKEND_OPENCV, DNN_BACKEND_VKCOM
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# 設定目標執行裝置 DNN_TARGET_CPU, DNN_TARGET_OPENCL, DNN_TARGET_OPENCL_FP16, DNN_TARGET_MYRIAD, DNN_TARGET_VULKAN, DNN_TARGET_FPGA
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

time_1 = time.clock()

# 讀取輸入影像
image = cv2.imread("car.png")

time_2 = time.clock()

#blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

# 設定輸出格式 [1, 3, 227, 227]
out_blob = cv2.dnn.blobFromImage(image, # 輸入影像
                                scalefactor=1.0, # 輸入資料尺度
                                size=(227, 227), # 輸出影像尺寸
                                mean=(0, 0, 0), # 從各通道減均值
                                swapRB=False, # R、B通道是否交換i
                                crop=False) # 是否截切

# 設定網路
net.setInput(out_blob)

time_3 = time.clock()

# 進行推論，輸出結果陣列大小[1,1000,1,1]
res = net.forward()

time_4 = time.clock()

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

print("load model time = {:.4f} sec.".format(time_1 - time_0))
print("load image time = {:.4f} sec.".format(time_2 - time_1))
print("setting model time = {:.4f} sec.".format(time_3 - time_2))
print("inference time = {:.4f} sec.".format(time_4 - time_3))
print("display time = {:.4f} sec.".format(time_5 - time_4))
print("total time = {:.4f} sec.".format(time_5 - time_0))
