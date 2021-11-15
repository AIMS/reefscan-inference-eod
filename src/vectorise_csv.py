import onnxruntime
import tensorflow as tf
from patch_loader import PatchLoader
from patch_loader import get_patch
from PIL import Image
import keras

import numpy as np
from keras.models import load_model, Model
from time import time
import csv

BATCH_SIZE = 100
WORKERS = 1000

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if len(gpu_devices) == 0:
    print ("WARNING: no GPU found")
else:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True



IMG_PATH_KEY = "image_path"
POINT_X_KEY = "pointx"
POINT_Y_KEY = "pointy"
LABEL_KEY = "unlabelled"


def convertTime(seconds):
    mins, sec = divmod(seconds, 60)
    hour, mins = divmod(mins, 60)
    if hour > 0:
        return "{:.0f} hour, {:.0f} minutes".format(hour, mins)
    elif mins > 0:
        return "{:.0f} minutes".format(mins)
    else:
        return "{:.0f} seconds".format(sec)


def inference(input_csv_path):

    weight_file = 'weights.best.hdf5'
    model_file = 'model.onnx'

    # load the model
    feature_layer_name = 'global_average_pooling2d_1' #'avg_pool'
    tic = time()
    # model = load_model(model_file)
    model = load_model(weight_file)
    print(model.summary())
    model = Model(inputs=model.inputs,
                  outputs=model.get_layer(feature_layer_name).output)
    print('Completed load model in {}'.format(convertTime(time()-tic)))

    patches = read_csv(input_csv_path)
    # patches = patches[0:10]

    batch_size = BATCH_SIZE
    val_generator = PatchLoader(patches, batch_size)

    tic = time()
    print('Starting inference...')
    predictions = model.predict_generator(val_generator,
                                          steps=(len(patches) // batch_size)+1,
                                          verbose=1,
                                          workers=WORKERS)

    print('Completed inference of {} points in {}'.format(len(patches), convertTime(time()-tic)))

    print(np.shape(predictions), np.shape(predictions[0]))

    i = 0
    np.set_printoptions(linewidth=np.inf)
    for patch in patches:
        vector = predictions[i]
        vector_string = str(vector.tolist())
        patch["feature_vector"] = vector_string
        # vector_filename = patch["patch_file"] + ".patch.txt"
        # with open(vector_filename, "w") as vector_file:
        #     vector_file.write(str(predictions[i]))
        i += 1

    write_csv(patches, "c:/temp/reefscan-vectors1.csv")

def inference_onnx(input_csv_path):
    sess_options = onnxruntime.SessionOptions()

    sess_options.intra_op_num_threads = 5
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    # sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = onnxruntime.InferenceSession(('model.onnx'), sess_options=sess_options)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    patch_dicts = read_csv(input_csv_path)
    batch_count = int(np.ceil(len(patch_dicts) / BATCH_SIZE))
    for batch in range(0, batch_count):
        start_batch = batch * BATCH_SIZE
        end_batch = start_batch + BATCH_SIZE - 1
        batch_patches = patch_dicts[start_batch:end_batch]

        patches = get_patches(batch_patches)
        vectors = session.run([label_name], {input_name: patches})[0]

        i = 0
        for patch in batch_patches:
            patch["feature_vector"] = vectors[i]
            # vector_filename = patch["patch_file"] + ".patch.txt"
            # with open(vector_filename, "w") as vector_file:
            #     vector_file.write(str(vectors[i]))
            i += 1


        print(f"finished batch {batch} of {batch_count}")


def inference_lite(input_csv_path):

    model_file = 'model.tflite'

    # Load the TFLite model in TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_file, num_threads=WORKERS)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    # Test the model on random input data.
    # input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    patch_dicts = read_csv(input_csv_path)

    batch_count = int(np.ceil(len(patch_dicts) / BATCH_SIZE))
    print ("init finished")
    for batch in range(0, batch_count):
        start_batch = batch * BATCH_SIZE
        end_batch = start_batch + BATCH_SIZE - 1
        batch_patches = patch_dicts[start_batch:end_batch]

        patches = get_patches(batch_patches)
        interpreter.allocate_tensors()
        patch_count = len(batch_patches)

        interpreter.resize_tensor_input(input_index, tensor_size=(patch_count, 256, 256, 3))
        interpreter.allocate_tensors()

        print("batch ready")

        vectors = inference_patches(input_index, interpreter, output_index, patches)
        print("batch inferenced")

        i = 0
        for patch in batch_patches:
            vector_filename = patch["patch_file"] + ".patch.txt"
            with open(vector_filename, "w") as vector_file:
                vector_file.write(str(vectors[i]))
            i += 1
        print(f"finished batch {batch} of {batch_count}")

def get_patches(patch_dicts):
    patches = np.empty(shape=(len(patch_dicts), 256, 256, 3), dtype=np.float32)
    index = 0
    for patch_dict in patch_dicts:
        patch = Image.open(patch_dict['patch_file'])
        patch = keras.preprocessing.image.img_to_array(patch)
        patches[index] = patch
        index += 1
    return patches


def inference_patches(input_index, interpreter, output_index, patches):
    interpreter.set_tensor(input_index, patches)
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_index)
    return output_data


def read_csv(input_csv_path):
    patches = []
    with open(input_csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            patches.append(row)
    return patches


def write_csv(dicts, csv_path):
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dicts[0].keys())
        writer.writeheader()
        for row in dicts:
            writer.writerow(row)


if __name__ == '__main__':
    # points_file = 'C:/aims/reef-scan/patches/photos.csv'
    points_file = 'C:/greg/reefscan_ml/from-reefmon/points.csv'
    print (points_file)

    tic = time()
    inference(points_file)
    print('Completed load model in {}'.format(convertTime(time()-tic)))
