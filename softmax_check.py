from tensorflow.keras.models import load_model
from efficientnet import *

model = load_model("./weights/idrid_1440x960/weights/PMCNet_idrid_1440x960_100.h5", compile=False)
print(model.layers[-1].name)
print(model.layers[-1].get_config())
