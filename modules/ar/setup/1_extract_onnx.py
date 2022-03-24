from modules.ar.utils.model import CNN_TRX
from utils.params import TRXConfig
import torch
import onnxruntime as ort
import numpy as np

if __name__ == "__main__":
    args = TRXConfig()
    model = CNN_TRX(args).to(args.device)  # declare necessary machine learning things
    model_data = torch.load('modules/ar/modules/raws/FULL.pth')
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    input_names = ["support", "labels", "query"]
    output_names = ['pred']

    support = torch.randn((args.way * args.seq_len, args.n_joints * 3)).to(args.device)
    query = torch.randn((args.seq_len, args.n_joints * 3)).to(args.device)
    labels = torch.IntTensor(np.array(list(range(args.way)))).to(args.device)

    res = model(support, labels, query)  # Check that arguments are good

    onnx_file_name = "modules/ar/modules/onnxs/FULL.onnx"
    # Export the model
    print('Export the onnx model static ...')
    torch.onnx.export(model,
                      (support, labels, query),
                      onnx_file_name,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes=None)
    print('Onnx model exporting done')

    # Check consistency
    res_pytorch = model(support, labels, query)

    sess = ort.InferenceSession('modules/ar/modules/onnxs/FULL.onnx')
    res_onnx = sess.run(None, {'support': support.cpu().detach().numpy(),
                               "labels": labels.cpu().detach().numpy(),
                               "query": query.cpu().detach().numpy()})
    res_onnx = res_onnx[0]
    res_pytorch = res_pytorch['logits'].detach().cpu().numpy()

    print(res_pytorch - res_onnx)
