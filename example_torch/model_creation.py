import torch


class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = SimpleNN(input_size=5, hidden_size=10, output_size=1)

# Export to ONNX for the unified evaluation pipeline.
dummy_input = torch.randn(1, 5)
torch.onnx.export(
    model,
    dummy_input,
    "checkpoint.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
