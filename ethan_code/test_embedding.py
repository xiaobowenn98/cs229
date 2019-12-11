import tensorflow as tf

model_file = "../../project/elmo.hdf5"
model_new = tf.keras.models.load_model(model_file)
print (model_new.predict(["help"]))