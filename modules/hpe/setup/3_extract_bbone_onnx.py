BATCH_SIZE = 5
inp = "modules/hpe/modules/signatured/bbone{}".format(BATCH_SIZE)
out = "modules/hpe/modules/onnxs/bbone{}.onnx".format(BATCH_SIZE)
command = "python -m tf2onnx.convert --saved-model {} --output {} --opset 9".format(inp, out)
print("Launch ", command)
