#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
# This sample uses an ONNX ResNet50 Model to create a TensorRT Inference Engine
import random
import sys

import numpy as np
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit
import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import csv

sys.path.insert(1, os.path.join(sys.path[0], ".."))
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
DTYPE = trt.float32

def GiB(val):
    return val * 1 << 30


# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    # network.add_input("foo", trt.float32, (1, 256, 256, 3))

    config = builder.create_builder_config()

    profile = builder.create_optimization_profile();
    profile.set_shape("input_1", (1, 256, 256, 3), (1, 256, 256, 3), (1, 256, 256, 3))
    config.add_optimization_profile(profile)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.max_workspace_size = GiB(1)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, 'rb') as model:
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print (parser.get_error(error))
            return None

    print("before return")
    engine = builder.build_engine(network, config)
    print("before return")
    return engine


def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        # c, h, w = ModelData.INPUT_SHAPE
        # image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()
        image_arr = np.asarray(image).astype(trt.nptype(DTYPE)).ravel()
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return image_arr

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image


def read_engine_from_file():
    with open('./engine.trt', 'rb') as f:
        engine_bytes = f.read()
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
    return engine


def make_engine_from_onnx():
    onnx_model_file = 'model.onnx'
    # Build a TensorRT engine.
    engine = build_engine_onnx(onnx_model_file)
    with open('./engine.trt', 'wb') as f:
        f.write(bytearray(engine.serialize()))
    print("I made an engine")
    return engine


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        print(str(binding))
        print(str(engine.get_binding_shape(binding)))
        print(str(engine.max_batch_size))
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        print(str(size))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def read_csv(input_csv_path):
    patches = []
    with open(input_csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            patches.append(row)
    return patches


def main():
    try:
        engine = read_engine_from_file()
    except:
        engine = make_engine_from_onnx()

    # Allocate buffers and create a CUDA stream.
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    # Contexts are used to perform inference.
    context = engine.create_execution_context()

    points_file = 'C:/aims/reef-scanner/images/20210727_233059_Seq02/photos.csv'
    patch_dicts = read_csv(points_file)
    for patch in patch_dicts:
        patch_file = patch['patch_file']
        vector_filename = patch_file + ".patch.txt"
        with open(vector_filename, "w") as vector_file:
            load_normalized_test_case(patch_file, inputs[0].host)
            trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            vector_file.write(str(trt_outputs))


if __name__ == '__main__':
    main()
