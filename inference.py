import tensorflow as tf # pyright: ignore[reportMissingModuleSource]
import numpy as np

model=tf.keras.models.load_model("model.keras")

img=tf.keras.utils.load_img(
    "test.jpg",
    target_size=(224,224)
)

img=tf.keras.utils.img_to_array(img)/255
img=np.expand_dims(img,0)

pred=model.predict(img)

if pred>0.5:
    print("Dog")
else:
    print("Cat")