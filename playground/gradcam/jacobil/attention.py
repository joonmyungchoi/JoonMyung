# from joonmyung.draw import drawImgPlot
# from joonmyung.analysis import JDataset
# import numpy as np
# import torch
#
#
# # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#
# @torch.no_grad()
# def grad_rollout(attentions, gradients, discard_ratio):
#     B, C, H, W = attentions[0].shape
#     result = torch.eye(W).unsqueeze(0).expand(B, -1, -1) # (1, 197, 197)
#     I = torch.eye(W).unsqueeze(0).expand(B, -1, -1) # (1, 197, 197)
#     for attention, grad in zip(attentions, gradients):
#         # attention : (1(B), 3(H), 197(h), 197(w)), grad : (1(B), 3(H), 197(h), 197(w))
#         weights = grad
#         attention_heads_fused = (attention * weights).mean(axis=1) # (1(B), 197(T), 197(T))
#         attention_heads_fused[attention_heads_fused < 0] = 0
#
#         # Drop the lowest attentions, but
#         # don't drop the class token
#         flat = attention_heads_fused.view(B, -1) # (1, 38809)
#         _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False) # 38809 → 34928
#         # indices = indices[indices != 0]
#         flat[0, indices] = 0
#
#         a = (attention_heads_fused + 1.0 * I) / 2
#         a = a / a.sum(dim=-1, keepdim=True)
#         result = torch.matmul(a, result)
#
#     # Look at the total attention between the class token,
#     # and the image patches
#     mask = result[:, 0, 1:] # (1(B), 197, 197) → (1(B), 196)
#     # In case of 224x224 image, this brings us from 196 to 14
#     width = int(W ** 0.5)
#     mask = mask / mask.max(dim=1)[0]
#     mask = mask.reshape(B, width, width).numpy()
#     return mask # (14, 14)
#
#
# def attn_rollout(attentions, discard_ratio, head_fusion):
#     result = torch.eye(attentions[0].size(-1))
#     with torch.no_grad():
#         for attention in attentions:
#             if head_fusion == "mean":
#                 attention_heads_fused = attention.mean(axis=1)
#             elif head_fusion == "max":
#                 attention_heads_fused = attention.max(axis=1)[0]
#             elif head_fusion == "min":
#                 attention_heads_fused = attention.min(axis=1)[0]
#             else:
#                 raise "Attention head fusion type Not supported"
#
#             # Drop the lowest attentions, but
#             # don't drop the class token
#             flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
#             _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
#             indices = indices[indices != 0]
#             flat[0, indices] = 0
#
#             I = torch.eye(attention_heads_fused.size(-1))
#             a = (attention_heads_fused + 1.0 * I) / 2
#             a = a / a.sum(dim=-1)
#
#             result = torch.matmul(a, result)
#
#     # Look at the total attention between the class token,
#     # and the image patches
#     mask = result[0, 0, 1:]
#     # In case of 224x224 image, this brings us from 196 to 14
#     width = int(mask.size(-1) ** 0.5)
#     mask = mask.reshape(width, width).numpy()
#     mask = mask / np.max(mask)
#     return mask
#
# class Rollout:
#     def __init__(self, model, attention_layer_name='attn_drop', head_fusion = "mean",
#                  discard_ratio=0.9):
#         self.model = model
#         self.discard_ratio = discard_ratio
#         self.head_fusion = head_fusion
#         for name, module in self.model.named_modules():
#             if attention_layer_name in name:
#                 module.register_forward_hook(self.get_attention)
#                 module.register_backward_hook(self.get_gradient)
#
#     def get_attention(self, module, input, output):
#         self.attentions.append(output.cpu())
#
#     def get_gradient(self, module, grad_input, grad_output):
#         self.attention_gradients.append(grad_input[0].cpu())
#
#     def get_grad_rollout(self, input_tensor, category_index, device):
#         self.attention_gradients, self.attentions = [], []
#         self.model.zero_grad()
#         output = self.model(input_tensor)
#         category_mask = torch.zeros(output.size(), device=device)
#         category_mask[:, category_index] = 1
#         loss = (output * category_mask).sum()
#         loss.backward()
#
#         return grad_rollout(self.attentions,
#                             self.attention_gradients,
#                             self.discard_ratio)
#
#
#     def get_attn_rollout(self, input_tensor):
#         self.attentions = []
#         with torch.no_grad():
#             self.model(input_tensor)
#
#         return attn_rollout(self.attentions, self.discard_ratio, self.head_fusion)
#
#
# if __name__ == '__main__':
#     # Section A. Data
#     # root_path, dataset_name = "/data1/joonmyung/data/imagenet", "imagenet"
#     root_path, dataset_name = "/hub_data2/joonmyung/data/imagenet", "imagenet"
#     # root_path, dsataset_name = "/data/opensets/imagenet-pytorch", "imagenet"
#     # data_num = [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]]
#     data_num = [[3, 1]]
#     device, ex_type = "cuda", "grad"
#
#     dataset = JDataset(root_path, dataset_name, device=device)
#     samples, targets, imgs, label_names = dataset.getItems(data_num)
#
#     # Section B. Model
#     model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True).to(device)
#     model.eval()
#
#     discard_ratio, head_fusion, category_index = 0.9, "mean", targets
#     rollout = Rollout(model, head_fusion="mean", discard_ratio=discard_ratio)
#
#     if ex_type == "grad":
#         masks = rollout.get_grad_rollout(samples, category_index, device=device)
#     else:
#         masks = rollout.get_attn_rollout(samples)
#     from joonmyung.draw import overlay
#     a = overlay(samples, masks, dataset_name)
#     drawImgPlot(a)
#
#     # for img, mask in zip(imgs, masks):
#     #     np_img = np.array(img)[:, :, ::-1]
#     #     mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
#     #     mask = show_mask_on_image(np_img, mask)
