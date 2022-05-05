inp = "modules/hpe/modules/signatured/bbone1"
out = "modules/hpe/modules/onnxs/bbone1.onnx"
command = "python -m tf2onnx.convert --saved-model {} --output {} --opset 9".format(inp, out)
print("Launch ", command)
