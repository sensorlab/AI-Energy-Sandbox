import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType


model = xgb.XGBRegressor()

# Fit the model

# Export to ONNX for the unified evaluation pipeline.
# Match `initial_types` to the trained feature count.
initial_types = [("input", FloatTensorType([None, 4]))]
onnx_model = convert_xgboost(model, initial_types=initial_types, target_opset=17)
with open("checkpoint.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
