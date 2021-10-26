Currently a collection of different ways to inference.  
I have been testing for performance on one of Nader's models on a fairly small GPU (MX450).  

For about 25000 patches these are the results.

TensorRT GPU (about 75%) 5 minutes  
TensorLite CPU (about 75%) 48 minutes  
onnx_runtime GPU (about 90%) 7 minutes  
Tensorflow will not run (out of memory).    

#Notes
TensorLite only supports GPU for Android or IOS   
TensorRT model I think was simplified so the results may less precise. Also I think it could be simplified further to go faster.   

To create the onnx model  
python -m tf2onnx.convert --opset 11 --tflite src/model.tflite  --output src/model.onnx

To run on a linux box install the nividia drivers and run the scripts   
docker-bash.sh - for TensorLite and onnx-runtime
    This one also needs the libraries in requirements.txt
trt_docker.sh - for TensorRT   

