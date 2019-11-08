# OpenVINO_OpenCV_Squeezenet

檔案內容說明：

classification_sample.py OpenVINO原始範例  
classification_openvino.py 以OpenVINO推論引擎(IE)載入IR（xml, bin)優化模型  
classification_opencv_ir.py 以OpenCV readNetFromModelOptimizer函式直接載入OpenVINO IR（xml, bin)優化模型  
classification_opencv_caffe.py 以OpenCV readNetFromCaffe函式直接讀取原始Squeezenet Caffe模型  
car.png 測試用圖檔  
squeezenet1.1.xml OpenVINO Model Zoo Squeezenet v1.1 IR檔(模型檔)  
squeezenet1.1.bin OpenVINO Model Zoo Squeezenet v1.1 IR檔(權重檔)  
squeezenet1.1.labels ImageNet 1000分類標籤檔  
squeezenet_v1.1.prototxt Squeezenet v1.1 Caffe格式模型檔  
squeezenet_v1.1.caffemodel Squeezenet v1.1 Caffe格式權重檔

比較SqueezeNet Version 1.1使用OpenVINO Inferenc Engin配合IR(xml, bin) 優化模型, OpenCV直接讀取IR(xml, bin) 優化模型及OpenCV直接讀取Caffe原始模型執行效率差異。

Squeezenet_cpu_result.txt 使用相同CPU執行OpenVINO, OpenCV+IR, OpenCV+Caffe三種方式比較結果檔案。  
Squeezenet_openvino_result.txt 使用OpenVINO, 分別以CPU, GPU, VPU三種裝置比較結果檔案。  
Squeezenet_opencv_IE_result.txt 使用OpenCV+IR, 分別以CPU, GPU(OPENCL, OPENCL_FP16)三種裝置比較結果檔案。  
Squeezenet_opencv_Caffe_result.txt 使用OpenCV+Caffe, 分別以CPU, GPU(OPENCL, OPENCL_FP16)三種裝置比較結果檔案。  
Squeezenet_Compare_Table.xlsx 所有比較表

更多完整說明待更新…
