

from timm.models.vision_transformer import Attention, Block
from collections import OrderedDict
from joonmyung.clip import clip
from timm import create_model
from pprint import pprint
import glob

import torch
import os

from torch import nn
class ResidualAttentionBlock(nn.Module):
    def forward(self, x: torch.Tensor, attn_mask):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ZeroShotInference():
    def __init__(self, model, classnames,
                 prompt = "a photo of a {}.", device = "cuda"):

        for i, resblock in enumerate(model.visual.transformer.resblocks):
            resblock.__class__ = ResidualAttentionBlock
            attn = Attention(resblock.attn.embed_dim, resblock.attn.num_heads, qkv_bias=True)
            self.convert_attention_block(resblock.attn, attn)
            resblock.attn = attn
        model.visual.TBD = False

        prompts = [prompt.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device)

        with torch.no_grad():
            text_features = model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) # (1000(T), 512(D))

        self.text_features = text_features
        self.model = model

    def convert_attention_block(self, src, dst):
        src_state_dict, dst_state_dict = src.state_dict(), dst.state_dict()
        src_to_dst_keys = [("in_proj_weight", "qkv.weight"), ("in_proj_bias", "qkv.bias"), ("out_proj.weight", "proj.weight"), ("out_proj.bias", "proj.bias")]
        for src_key, dst_key in src_to_dst_keys:
            dst_state_dict[dst_key] = src_state_dict[src_key]
        dst.load_state_dict(dst_state_dict)
        dst.to(device = src_state_dict["in_proj_weight"].device, dtype = src_state_dict["in_proj_weight"].dtype)

    def __call__(self, image):
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


class JModel():
    def __init__(self, num_classes = 1000, root_path= "/hub_data1/joonmyung/weights", device="cuda"):
        self.num_classes = num_classes

        self.root_path = root_path
        self.model_path = glob.glob(os.path.join(root_path, "*.pth"))
        pprint(self.model_path)

        self.device = device

    def load_state_dict(self, model, state_dict):
        state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
        model.load_state_dict(state_dict)



    def getModel(self, model_type=0, model_name ="deit_tiny"):
        preprocess = None
        if model_type == 0:
            model = create_model(model_name, pretrained=True, num_classes=self.num_classes, in_chans=3, global_pool=None, scriptable=False)
        elif model_type == 1:
            model = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=True)
        elif model_type == 2:
            model, _ = clip.load(model_name)
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

