from modules.ar.trx.utils.model import CNN_TRX
from utils.params import TRXConfig
import torch
import onnxruntime as ort
import numpy as np

if __name__ == "__main__":
    args = TRXConfig()
    model = CNN_TRX(args).to(args.device)  # declare necessary machine learning things
    model_data = torch.load(args.model_path)
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    input_names = ["support", "labels", "query"]
    output_names = ['pred']

    support = torch.randn((1, args.way * args.seq_len, args.n_joints * 3), requires_grad=True).to(args.device)
    query = torch.randn((1, args.seq_len, args.n_joints * 3), requires_grad=True).to(args.device)
    labels = torch.IntTensor(np.array([list(range(args.way))])).to(args.device)

    res = model(support, labels, query)  # Check that arguments are good

    onnx_file_name = "modules/ar/trx/checkpoints/trx.onnx"
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

    sess = ort.InferenceSession('modules/ar/trx/checkpoints/trx.onnx')
    res_onnx = sess.run(None, {'support': support.cpu().detach().numpy(),
                               "labels": labels.cpu().detach().numpy(),
                               "query": query.cpu().detach().numpy()})

    pass
