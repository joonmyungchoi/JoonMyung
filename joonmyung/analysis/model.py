
from collections import OrderedDict

import glob
from pprint import pprint

from timm import create_model
from clip import clip
import torch
import os

from clip import clip
class ZeroShotInference():
    def __init__(self, model, classnames,
                 prompt = "a photo of a {}.", device = "cuda"):

        prompts = [prompt.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device)

        with torch.no_grad():
            text_features = model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.model = model

    def __call__(self, image):
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


class JModel():
    def __init__(self, num_classes = None, root_path= None, device="cuda"):
        self.num_classes = num_classes

        self.root_path = root_path
        self.model_path = glob.glob(os.path.join(root_path, "*.pth"))
        pprint(self.model_path)

        self.device = device

    def load_state_dict(self, model, state_dict):
        state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
        model.load_state_dict(state_dict)


    def getModel(self, model_type=0, model_name ="deit_tiny"):

        if model_type == 0:
            model = create_model(model_name, pretrained=True, num_classes=self.num_classes, in_chans=3, global_pool=None, scriptable=False)

        elif model_type == 1:
            model = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=True)

        elif model_type == 2:
            url = clip._MODELS[model_name]
            model_path = clip._download(url, self.root_path)
            model = torch.jit.load(model_path, map_location="cpu")
            model = clip.build_model(model.state_dict())

        elif model_type == 3:
            checkpoint = torch.load(self.root_path, map_location='cpu')
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

