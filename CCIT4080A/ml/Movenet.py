import tensorflow as tf
import numpy as np

# defines a class Movenet with methods for initialization, model loading, and pose estimation.
class Movenet(object):
    # The class constructor __init__ sets up the TensorFlow Lite interpreter and retrieves input and output details.
    def __init__(self, model_name: str):
        self.model_path, self.input_size = self.load_model(model_name)
        Interperter = tf.lite.Interpreter(model_path = self.model_path)
        Interperter.allocate_tensors()
        self.input_detail = Interperter.get_input_details()[0]["index"]
        self.output_detail = Interperter.get_output_details()[0]["index"]
        self.interperter = Interperter
    # The load_model method selects the model path and input size based on the provided model name.
    def load_model(self, model_name : str):
        if model_name == "Movenet lightning (float 16)":
            model_path = "lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite"
            input_size = 192

        elif model_name == "Movenet thunder (float 16)":
            model_path = "lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite"
            input_size = 256

        elif model_name == "Movenet lightning (int 8)":
            model_path = "lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite"
            input_size = 192

        elif model_name == "Movenet thunder (int 8)":
            model_path = "lite-model_movenet_singlepose_thunder_tflite_int8_4.tflite"
            input_size = 256

        return model_path, input_size
    # The movenet method processes the input image, performs inference, and returns the detected keypoints with scores.
    def movenet(self, input_image : np.ndarray):
        input_image = tf.expand_dims(input_image, axis= 0)
        input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
        input_image = tf.cast(input_image, dtype = np.uint8)
        self.interperter.set_tensor(self.input_detail, input_image)
        self.interperter.invoke()
        keypoints_with_scores = self.interperter.get_tensor(self.output_detail)
        return keypoints_with_scores # [1,1,17,3]



