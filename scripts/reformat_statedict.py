import sys
import torch

sd_path = sys.argv[1]

sd = torch.load(sd_path)

# Check if already reformatted by checking if first key has model. prefix
# if not list(sd.keys())[0].startswith('model.'):
#     print('SD seems already reformatted: ', sd.keys())
#     sys.exit(0)
# Remove model i.e. model.h.1 -> h.1
# sd = {k[17:] if k.startswith('base_model.model.') else k: v for k, v in sd.items()}
sd = {f"model.base_model.model.{k}": v for k, v in sd.items()}

torch.save(sd, sd_path)