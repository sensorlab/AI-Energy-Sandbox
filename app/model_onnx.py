import numpy as np
import onnxruntime as ort


class Model:
    def __init__(self, checkpoint_path: str):
        """
        Initialization of the model.
        """
        self.checkpoint_path = checkpoint_path
        self.session = None
        self.input_name = None
        self.output_dim = None

    def load_checkpoint(self):
        """
        Loads weights from ONNX checkpoint.
        """
        self.session = ort.InferenceSession(
            self.checkpoint_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, df, target_index: int):
        """
        Makes prediction based on the dataframe df with input features. Returns predictions for specific target_index
        of shape (len(df),).
        """
        data = df.fillna(0).values.astype(np.float32)
        outputs = self.session.run(None, {self.input_name: data})
        preds = np.asarray(outputs[0]).reshape(len(df), -1)
        return preds[:, target_index]
