from keras.models import load_model, Model

weight_file = 'weights.best.hdf5'
# load the model
feature_layer_name = 'global_average_pooling2d_1'  # 'avg_pool'
model = load_model(weight_file)
# print (model.summary())
model = Model(inputs=model.inputs,
              outputs=model.get_layer(feature_layer_name).output)

model.save('model_dir')
