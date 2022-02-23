from keras.models import load_model, Model
import tensorflow as tf
import tensorflow_model_optimization as tfmot

weight_file = 'weights.best.hdf5'
# load the model
feature_layer_name = 'global_average_pooling2d_1'  # 'avg_pool'
model = load_model(weight_file)
# print (model.summary())
model = Model(inputs=model.inputs,
              outputs=model.get_layer(feature_layer_name).output)

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model)

model_for_pruning.summary()

# model_for_export = tfmot.sparsity.keras.strip_pruning(model)

