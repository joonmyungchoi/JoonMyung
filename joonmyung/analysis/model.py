from joonmyung.utils import isDir
from timm import create_model
import torch
import os

class JModel():
    def __init__(self, num_classes = None, root_path= None,
                 device="cuda", p=False):
        # Pretrained_Model
        self.num_classes = num_classes

        # Other
        if root_path:
            self.model_path = os.path.join(root_path, "checkpoint_{}.pth")
        if p and root_path:
            print("file list : ", sorted(os.listdir(root_path), reverse=True))
        self.device = device

    def getModel(self, epoch=0, model_number=0, model_name = "deit_tiny", **kwargs):
        model, args = None, None
        if model_number == 0:
            model = create_model(model_name, pretrained=True, num_classes=self.num_classes, in_chans=3, global_pool=None, scriptable=False).to(self.device)
        elif model_number == 1:
            model = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=True).to(self.device)
        # elif model_number == 2:
        #     model_path = self.model_path.format(str(epoch))
        #     if not isDir(model_path): raise FileExistsError
        #
        #     checkpoint = torch.load(model_path, map_location='cpu')
        #     args = checkpoint['args']
        #     if "token_merging" in kwargs.keys():
        #         args.task_type[1] = 1 if kwargs["token_merging"] else 0
        #
        #     model = get_model(args).to(self.device)
        #     state_dict = []
        #     for n, p in checkpoint['model'].items():
        #         if "total_ops" not in n and "total_params" not in n:
        #             state_dict.append((n, p))
        #     state_dict = dict(state_dict)
        #     model.load_state_dict(state_dict)

        model.eval()

        return model, args


