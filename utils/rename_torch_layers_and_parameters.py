import torch
from modules.ar.utils.model import TRXOS
from utils.params import TRXConfig

args = TRXConfig()

ar = TRXOS(TRXConfig(), add_hook=False)
# Fix dataparallel
all_dict = torch.load(args.final_ckpt_path, map_location=torch.device(0))
state_dict = all_dict['model_state_dict']
state_dict = {param.replace('features_extractor', 'features_extractor.sk'): data for param, data in state_dict.items()}
state_dict["post_resnet.l1.weight"] = torch.zeros((256, 2048)).cuda()
state_dict["post_resnet.l1.bias"] = torch.zeros((256,)).cuda()
all_dict['model_state_dict'] = state_dict

ar.load_state_dict(state_dict)

torch.save(all_dict, args.final_ckpt_path)
