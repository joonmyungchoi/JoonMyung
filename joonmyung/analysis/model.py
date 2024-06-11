from collections import OrderedDict
from joonmyung.utils import isDir
from timm import create_model
import models.deit
import torch
import os

class JModel():
    def __init__(self, num_classes = None, model_path= None, device="cuda", p=False):
        # Pretrained_Model
        self.num_classes = num_classes

        if model_path:
            self.model_path = os.path.join(model_path, "checkpoint.pth")

        if p and model_path:
            print("file list : ", sorted(os.listdir(model_path), reverse=True))
        self.device = device

    def load_state_dict(self, model, state_dict):
        state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
        model.load_state_dict(state_dict)


    def getModel(self, model_type=0, model_name ="deit_tiny", **kwargs):

        if model_type == 0:
            model = create_model(model_name, pretrained=True, num_classes=self.num_classes, in_chans=3, global_pool=None, scriptable=False)
        elif model_type == 1:
            model = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=True)
        elif model_type == 2:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            args = checkpoint['args']
            model = create_model(
                        args.model,
                        pretrained=args.pretrained,
                        num_classes=args.nb_classes,
                        drop_rate=args.drop,
                        drop_path_rate=args.drop_path,
                        drop_block_rate=None,
                        img_size=args.input_size,
                        token_nums=args.token_nums,
                        embed_type=args.embed_type,
                        model_type=args.model_type
                    ).to(self.device)
            state_dict = []
            for n, p in checkpoint['model'].items():
                if "total_ops" not in n and "total_params" not in n:
                    state_dict.append((n, p))
            state_dict = dict(state_dict)
            model.load_state_dict(state_dict)
        else:
            raise ValueError
        model.eval()

        return model.to(self.device)

