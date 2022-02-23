import numpy as np

BATCH_SIZE = 32

dummy_input_batch = np.zeros((BATCH_SIZE, 256, 256, 3))

PRECISION = "FP32"

from helper import ModelOptimizer # using the helper from <URL>

model_dir = 'tmp_savedmodels/resnet50_saved_model'

opt_model = ModelOptimizer(model_dir)