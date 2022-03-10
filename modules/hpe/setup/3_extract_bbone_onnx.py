inp = "modules/hpe/modules/signatured/bbone"
out = "modules/hpe/modules/onnxs/bbone.onnx"
command = "python -m tf2onnx.convert --saved-model {} --output {}".format(inp, out)
print("Launch ", command)
