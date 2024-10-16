from draw import drawImgPlot, unNormalize, saliency, drawHeatmap, make_visualization
from utils import to_np, read_classnames
import numpy as np
import torch



def drawAnalysis(model, dataset, idxs, compression, frame_num = 1, patch_size=1, image_size=224, views=[False, True, True], device="cuda"):
    patch_num = int(image_size / patch_size)

    for idx in idxs:
        image, _ = dataset[idx]
        image = image.to(device, non_blocking=True).unsqueeze(0)  # (1(B), 4(F), 3(C), 224(H), 224(W))
        model.encode_vision(image, test=True)
        attns = model.vision_encoder.info["analysis"]["attn"]

        if views[0]: # RAW IMAGE
            print(f"IMAGE IDX : {idx}")
            drawImgPlot(unNormalize(image[0], "imagenet"), col=frame_num)

        if views[1]: # MERGE POSITION
            source = model.vision_encoder.info["compression"]["source"][:, 1:, 1:] # (1, 685, 1025)
            r_prune, r_merge = make_visualization(image, source, patch_size=patch_size, token_nums=0,
                                                         min_merge_nums=1, prune=False, merge=True, unmerge=False)
            r_merge = np.uint8(to_np(r_merge) * 255)
            ung = ((source.sum(dim=2) == 1)[:, :, None] * source).sum(dim=1).reshape(frame_num, patch_num, patch_num)
            ung = np.uint8(to_np(ung[:, :, None, :, None].repeat(1, 1, patch_size, 1, patch_size).reshape(-1, 1, image_size, image_size)))
            imgs = np.uint8(to_np(unNormalize(image.reshape(-1, 3, image_size, image_size), "imagenet")) * 255)
            result_a = (r_merge * (1 - ung)) + imgs * ung
            drawImgPlot(torch.from_numpy(result_a), col=frame_num)

        if views[2] and compression[2][0]: # MASS POSITION
            source = model.vision_encoder.info["compression"]["source"][:, 1:, 1:]  # (1, 685, 1025)
            size = model.vision_encoder.info["compression"]["size"][:, 1:]  # (1, 685, 1)
            score = ((size / source.sum(dim=-1)[:, :, None]) * source).sum(dim=1).reshape(frame_num, patch_num, patch_num)
            drawHeatmap(score, col=frame_num)


if __name__ == "__main__":
    from dataset import JDataset
    from model import ZeroShotInference, JModel

    classnames = read_classnames("/hub_data1/joonmyung/data/imagenet/classnames.txt")
    model, preprocess = JModel().getModel(2, "ViT-B/16")
    model = ZeroShotInference(model, classnames, prompt="a photo of a {}.")

    compression, idxs = [[1, 0, 10, 0, 1, 1], [1, 10, 25, 1], [0]], range(1000)
    views = [False, True, True] # [RAW, MERGE, MASS]
    dataset = JDataset("/hub_data1/joonmyung/data/imagenet", "imagenet", train=False)
    drawAnalysis(model, dataset, idxs, compression, frame_num=1, patch_size=14, image_size=224, views=views, device="cuda")


# STEP I. GET INFORMATION
    # SELF-ATTN ✓
    # TEXT/VISION FEATURES ✓
    # SIZE/SOURCE

# STEP II. DRAW ANALYSIS
    # TEXT - VISION SIMILARITY
        # FINAL   - ALL
        # AVERAGE - ALL
        # N-LAYER - N-LAYER

    # TOKEN MERGING
        # IMAGE
        # SIZE
        # SOURCE

    # VID-TLDR
        # ATTN
        # IMAGE