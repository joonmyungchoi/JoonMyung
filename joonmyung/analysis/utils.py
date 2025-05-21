from collections import defaultdict

def getImpFitprune(attn, start, end):
    # attn : (L, H, T, T)
    attn_headmax = attn.max(dim=1).values.squeeze()

    attn_self  = attn_headmax[:, start:end, start:end].sum(dim=1) / (end - start)
    attn_cross = attn_headmax[:, end:, start:end].sum(dim=1)
    importance  = attn_self * attn_cross
    return importance

def getImpFastV(attn, start, end):
    # attn : (L, H, T, T)
    attn_headavg = attn.mean(dim=1)
    importance = attn_headavg[:, -1, start:end]

    return importance

def resetInfo(info):
    if info["analysis"]["use"]:
        info["analysis"]["attn"] = []
        info["analysis"]["hidden_states"] = []

        info["analysis"]["diff"] = defaultdict(list)
        info["analysis"]["norm1"] = defaultdict(list)
        info["analysis"]["norm2"] = defaultdict(list)

    if info["compression"]["use"]:
        info["compression"]["attn"] = None
        info["compression"]["diff"] = None
        info["compression"]["source"] = None



def setInfo():
    return {"use_flash_attn": True,
             "compression":{"use": False, "attn": None, "diff": None, "norm1": None, "norm2": None},
             "analysis":{"use": False, "attn": [], "hidden_states": [], "diff": defaultdict(list), "norm1": defaultdict(list), "norm2": defaultdict(list)}
            }