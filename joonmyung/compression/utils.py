from collections import defaultdict
import torch
def getImpBase(attn, start=None, end=None, cls=False):
    attn_base = attn[:, :, 0].mean(dim=1) if cls else attn.mean(dim=(1,2))
    return attn_base[:, start:end]

def getImpFitprune(attn, start=None, end=None):
    attn_headmax = attn.max(dim=1).values

    attn_self  = attn_headmax[:, start:end, start:end].sum(dim=1) / (end - start)
    attn_cross = attn_headmax[:, end:, start:end].sum(dim=1)
    importance  = attn_self * attn_cross
    return importance

def getImpFastV(attn, start=None, end=None):
    attn_headavg = attn.mean(dim=1)
    importance = attn_headavg[:, -1, start:end]
    return importance

def getL2Norm(feat, start = None, end = None):
    return torch.norm(feat, p=2, dim=-1)[:, start:end]

def getVidTLDR(attn, start = None, end = None):
    attn_headavg = attn.mean(dim=1) # B T T
    importance = -(attn_headavg * torch.log(attn_headavg)).mean(dim=1)[start:end]
    return importance

def unPrune(values, source):
    if source == None:
        return values
    result = torch.zeros_like(source, device=source.device, dtype=values.dtype)
    result[source] = values
    return result


def getAttnFrom(attn, start=None, end=None, cls=False, enc=False):
    attn_headavg = attn.mean(dim=1)
    if not enc and attn.shape[2] != 1:
        vis2vis_ratio  = attn_headavg[:, start:end, start:end].mean(dim=-2).sum(dim=-1)
        vis2text_ratio = attn_headavg[:, end:, start:end].mean(dim=-2).sum(dim=-1)
        vis2last_ratio = attn_headavg[:, -1, start:end].sum(dim=-1)
        result = torch.cat([vis2vis_ratio, vis2text_ratio, vis2last_ratio], dim=0)
    elif cls:
        patch2cls_ratio   = attn_headavg[:, 1:, 1:].mean(dim=-2).sum(dim=-1)
        patch2patch_ratio = attn_headavg[:,  0, 1:].sum(dim=-1)
        result = torch.cat([patch2cls_ratio, patch2patch_ratio], dim=0)
    else:
        result = None

    return result

def getAnalysis(info, attn = None, feat = None, enc= False):
    if attn is not None and len(attn.shape) == 3: attn = attn[None]
    if feat is not None and len(feat.shape) == 2: feat = feat[None]


    if info["analysis"]["use"]:
        cls, source = info["analysis"]["cls"], info["compression"].get("source", None)
        [start, end] = info["analysis"].get("img_idx", [None, None])

        if attn is not None: # (B, H, T, T)
            # PART I.  INFORMATION
            info["analysis"]["vis_ratio"].append(getAttnFrom(attn, start=start, end=end, cls=cls, enc=enc))

            # PART II. VISUALIZATION
            info["analysis"]["base"].append(unPrune(getImpBase(attn, start, end, cls=cls), source))
            info["analysis"]["vidTLDR"].append(unPrune(getVidTLDR(attn, start, end), source))
            if start != None and end != None:
                info["analysis"]["fastV"].append(getImpFastV(attn, start, end))
                info["analysis"]["fitPrune"].append(getImpFitprune(attn, start, end))

        if feat is not None:
            info["analysis"]["norm2"].append(unPrune(getL2Norm(feat, start, end), source))
            # info["analysis"]["hidden_states"].append(feat.detach())

    if info["compression"]["use"]:
        cls, importance = info["compression"]["cls"], None
        [start, end] = info["compression"]["img_idx"]

        if attn is not None and info["compression"]["info_type"] == 0:    # attn : BASE
            importance = getImpBase(attn, start=start, end = end, cls=cls)
        elif attn is not None and info["compression"]["info_type"] == 1:  # attn : vid-TLDR
            importance = getVidTLDR(attn, start=start, end = end)
        elif attn is not None and info["compression"]["info_type"] == 2:  # attn : fastV
            importance = getImpFastV(attn, start = start, end = end)
        elif attn is not None and info["compression"]["info_type"] == 3:  # attn : fitPrune
            importance = getImpFitprune(attn, start = start, end = end)
        elif feat is not None and info["compression"]["info_type"] == 4:  # feat : norm2
            importance = getL2Norm(feat, start=start, end = end)

        if importance is not None: info["compression"]["importance"] = importance

def resetInfo(info, compression = None):
    if info["analysis"]["use"]:
        # PART I. INFORMATION
        info["analysis"]["vis_ratio"] = []

        # PART II. VISUALIZATION
        info["analysis"]["base"]     = []
        info["analysis"]["vidTLDR"]  = []
        info["analysis"]["fastV"]    = []
        info["analysis"]["fitPrune"] = []
        info["analysis"]["norm2"]    = []

        info["analysis"]["img_idx"] = [None, None]


    if compression is not None:
        info["compression"]["use"] = True
        info["compression"]["info_type"]       = compression[0][0]
        info["compression"]["prune_r_layer"]   = compression[0][1]
        info["compression"]["prune_r"]         = compression[0][2]
        info["compression"]["prune_thr_layer"] = compression[0][3]
        info["compression"]["prune_thr"]       = compression[0][4]
        info["compression"]["border_remove"]   = compression[0][5]
        info["compression"]["group_num"]       = compression[0][6]



        info["use_flash_attn"] = False if info["compression"]["info_type"] in [0, 1, 2, 3] else True

        info["compression"]["tau_sim"] = compression[1][0]
        info["compression"]["tau_info"] = compression[1][1]
        info["compression"]["tau_size"] = compression[1][2]
        info["compression"]["pooling_type"] = compression[1][3]
        info["compression"]["mass"] = compression[2][0]

    if info["compression"]["use"]:
        info["compression"]["size"] = None
        info["compression"]["source"] = None
        info["compression"]["img_idx"] = [None, None]



def saveResult(result_sep, result_all):
    em, f1 = result_all["overall"]["list_em"], result_all["overall"]["list_f1"]
    mod_image, mod_table, mod_text = result_all["modalities"]["image"]["list_f1"], result_all["modalities"]["table"]["list_f1"], result_all["modalities"]["text"]["list_f1"]
    hop_single, hop_multi = result_all["hop_types"]["Single-hop"]["list_f1"], result_all["hop_types"]["Multi-hop"]["list_f1"]
    drop_ratio_enc = sum([v["drop_ratio_enc"].item() for v in result_sep.values()]) / len(result_sep)
    drop_ratio_dec = sum([v["drop_ratio_dec"].item() for v in result_sep.values()]) / len(result_sep)

    results = {"em" : em, "f1" : f1, "mod_image" : mod_image, "mod_table" : mod_table, "mod_text" : mod_text, "hop_single": hop_single, "hop_multi": hop_multi,
               "drop_ratio_enc" : drop_ratio_enc, "drop_ratio_dec" : drop_ratio_dec}




