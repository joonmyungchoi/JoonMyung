from collections import defaultdict

from joonmyung.compression.compression import needAttn, needNaive
import torch.nn.functional as F
import numpy as np
import torch

def getVisualToken(x, start = None, end = None):
    if x == None:
        return None
    return x[:, start:end]

def getImpBase(attn, start=None, end=None, cls=False):
    attn_base = attn[:, :, 0].mean(dim=1) if cls else attn.mean(dim=(1,2))
    return attn_base[:, start:end]

def getImpFitprune(attn, start=None, end=None):
    attn_headmax = attn.max(dim=1).values
    attn_self = attn_headmax[:, start:end, start:end].mean(dim=1)
    attn_cross = attn_headmax[:, end:, start:end].mean(dim=1)
    importance  = attn_self * attn_cross
    return importance

def getImpFastV(attn, start=None, end=None):
    attn_headavg = attn.mean(dim=1)
    importance = attn_headavg[:, -1, start:end]
    return importance

def getL2Norm(feat, start = None, end = None):
    return torch.norm(feat, p=2, dim=-1)[:, start:end]

def getComplexity(feat, start=None, end=None):
    feat_norm = F.normalize(feat.to(torch.float32), dim=-1)[:, start:end]
    return 1 - (feat_norm @ feat_norm.transpose(-1, -2)).mean(dim=-1)

def getImpVidTLDR(attn, start = None, end = None):
    attn_headavg = attn.mean(dim=1) # B T T
    importance = -(attn_headavg * torch.log(attn_headavg)).mean(dim=1)[start:end]
    return importance

def unPrune(values, source):
    if source == None:
        return values
    result = torch.zeros_like(source, device=source.device, dtype=values.dtype)
    result[source] = values
    return result

def splitAttn(attn, start, end): # START | PROMPT | VIS | TEXT | LAST
    attn = attn.mean(dim=1)
    return torch.stack([attn[:, 0], attn[:, :start].sum(dim=-1), attn[:, start:end].sum(dim=-1), attn[:, end:].sum(dim=-1), attn[:, -1]], dim=-1)

def getAttnRatio(attn, start=None, end=None, cls=False, enc=False):
    attn_headavg = attn.mean(dim=1) # (1(B), 2551(T), 2551(T))
    N = attn.shape[2]
    if not enc and N != 1: # DECODER
        prt_s    = splitAttn(attn_headavg[:,  :1],           start = start, end = end)
        prt      = splitAttn(attn_headavg[:, :start],        start = start, end = end)
        vis      = splitAttn(attn_headavg[:, start:end],     start = start, end = end)
        txt      = splitAttn(attn_headavg[:, end:],          start = start, end = end)
        txt_e    = splitAttn(attn_headavg[:, -1:],           start = start, end = end)
        result_full = torch.stack([prt_s, prt, vis, txt, txt_e], dim=1)
        results_text = attn_headavg[:, end:, end:].sum(dim=1) / torch.arange(N - end, 0, -1, device=attn.device)
        return result_full, results_text
    elif cls: # ENCODER & CLS
        patch2cls_ratio   = attn_headavg[:, 1:, 1:].mean(dim=-2).sum(dim=-1)
        patch2patch_ratio = attn_headavg[:,  0, 1:].sum(dim=-1)
        result = torch.cat([patch2cls_ratio, patch2patch_ratio], dim=0)
        return result

@torch.no_grad()
def head_agreement_from_argmax(argmax_idx, Tk):
    B, H = argmax_idx.shape
    # one-hot accumulate counts per position (B,Tk)
    counts = torch.zeros(B, Tk, device=argmax_idx.device, dtype=torch.int32)
    # scatter-add(1) for each head
    counts.scatter_add_(dim=1, index=argmax_idx, src=torch.ones_like(argmax_idx, dtype=counts.dtype))
    max_count = counts.max(dim=1).values  # (B,)
    agreement = max_count.to(torch.float32) / float(H)
    return agreement

def pmax_from_scores(scores):
    m, idx = scores.max(dim=-1) # (B, H)
    lse = torch.logsumexp(scores, dim=-1)  # (B,H)
    pmax = torch.exp(m - lse).clamp(max=1.0)

    return pmax.mean(dim=1), pmax.std(dim=1)

def topk_entropy_from_scores(scores, k = 32):
    B, H, Tk = scores.shape
    topk_vals, _ = scores.topk(k=min(k, Tk), dim=-1)  # (B,H,k)
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)  # (B,H,1)
    p_topk = torch.exp(topk_vals - lse)  # (B,H,k)
    p_rest = (1.0 - p_topk.sum(dim=-1)).clamp_min(0.0)  # (B,H)
    H_topk = -(p_topk * (topk_vals - lse)).sum(dim=-1)  # (B,H)
    k_eff = p_topk.shape[-1]
    rest_slots = (Tk - k_eff)
    H_rest = torch.zeros_like(p_rest)
    if rest_slots > 0:
        H_rest = -p_rest * ((p_rest.clamp_min(1e-12)).log() - torch.log(torch.tensor(rest_slots, device=scores.device)))

    H = H_topk + H_rest
    return H

def getDelta(info, feat, feat_prev, name):
    if feat.shape[1] != 1:
        info["analysis"]["interpret"][f"{name}_l2"].append(torch.round((feat - feat_prev).norm(dim=-1).to(torch.float32))) # (B, T)
        info["analysis"]["interpret"][f"{name}_cosine"].append(torch.round(torch.cosine_similarity(feat, feat_prev, dim=-1).to(torch.float32), decimals=3)) # (B, T)

def getAnalysis(info, attn = None, feat = None, enc= False, layer_idx = False):
    if attn is not None and len(attn.shape) == 3: attn = attn[None]
    if feat is not None and len(feat.shape) == 2: feat = feat[None]
    info_temp = info["temp"]
    info_ana  = info["analysis"]
    info_comp = info["compression"]

    if info_ana["use"]:
        i_start, i_end, i_len = info_comp["img_idx"]
        cls, source, group_num = info_ana["cls"], info["compression"].get("source", None), info["compression"].get("group_num", 1)
        source_vis = getVisualToken(source, i_start, 2523)
        if source_vis is not None and group_num > 1:
            source_vis = source_vis.unsqueeze(-1).expand(-1, -1, group_num).reshape(source_vis.shape[0], -1)

        if attn is not None and attn.shape[2] != 1: # (B, H, T, T)
            attn = attn.to(torch.float32)
            info_ana["base"].append(unPrune(getImpBase(attn, i_start, i_end, cls=cls), source_vis))

            if i_start != None and i_end != None: # DECODER
                info_ana["attn"].append(attn.mean(dim=(0, 1))[-1])
                ratio_type, ratio_text = getAttnRatio(attn, start=i_start, end=i_end, cls=cls, enc=enc)
                info_ana["attn_ratio_type"].append(ratio_type)
                info_ana["attn_ratio_text"].append(ratio_text)


                attn_alloc_full = torch.stack([attn.mean(dim=(0, 1))[-1][:i_start].sum(dim=-1), attn.mean(dim=(0, 1))[-1][i_start:i_end].sum(dim=-1), attn.mean(dim=(0, 1))[-1][i_end:i_len - 1].sum(dim=-1), attn.mean(dim=(0, 1))[-1][i_len - 1:].sum(dim=-1)])
                attn_alloc_token = torch.stack([attn.mean(dim=(0, 1))[-1][:i_start].mean(dim=-1), attn.mean(dim=(0, 1))[-1][i_start:i_end].mean(dim=-1), attn.mean(dim=(0, 1))[-1][i_end:i_len - 1].mean(dim=-1), attn.mean(dim=(0, 1))[-1][i_len - 1:].mean(dim=-1)])
                info_ana["eos_attn_alloc"].append(attn_alloc_full)
                info_ana["eos_attn_effi"].append(attn_alloc_token / attn_alloc_token[1])

                info_ana["fastV"].append(getImpFastV(attn, i_start, i_end))
                info_ana["fitPrune"].append(getImpFitprune(attn, i_start, i_end))

            else: # ENCODER
                info_ana["vidTLDR"].append(unPrune(getImpVidTLDR(attn, i_start, i_end), source_vis))

        if feat is not None and feat.shape[1] != 1:
            info_ana["norm2"].append(unPrune(getL2Norm(feat, i_start, i_end), source_vis))
            feat_norm = F.normalize(feat.to(torch.float32), dim=-1)  # ↑ : 단순
            complexity = (1 - (feat_norm @ feat_norm.transpose(-1, -2))).mean(dim=-1)  # ↑ : 복잡
            # complexity = (1 - (feat_norm @ feat_norm.transpose(-1, -2))).mean()  # ↑ : 복잡
            info_ana["complexity"].append(complexity)

            if i_start != None: # ENCODER
                # PART I. Entropy / Logit / PRED
                logits = info_temp["lm_head"](info_temp["norm"](feat[:, -1].detach()))
                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)
                pred = logits.argmax(dim=-1).int()
                info_ana["logit"].append(logits)
                info_ana["entropy"].append(entropy)
                info_ana["pred"].append(pred)


    if info_comp["use"]:
        i_start, i_end, i_len = info_comp["img_idx"]
        cls, importance = info_comp["cls"], None

        if attn is not None and info_comp["info_type"] == 1:    # attn : BASE
            importance = getImpBase(attn, start=i_start, end = i_end, cls=cls)
        elif attn is not None and info_comp["info_type"] == 2:  # attn : vid-TLDR
            importance = getImpVidTLDR(attn, start=i_start, end = i_end)
        elif attn is not None and info_comp["info_type"] == 3:  # attn : fastV
            importance = getImpFastV(attn, start = i_start, end = i_end)
        elif attn is not None and info_comp["info_type"] == 4:  # attn : fitPrune
            importance = getImpFitprune(attn, start = i_start, end = i_end)
        elif feat is not None and info_comp["info_type"] == 5:  # feat : norm2
            importance = getL2Norm(feat, start=i_start, end = i_end)
        elif feat is not None and info_comp["info_type"] == 6:  # feat : redundancy
            importance = getComplexity(feat, start=i_start, end = i_end)
        elif info_comp["info_type"] in [7, 8]:  # attn : pre_propagate
            importance = info_comp["attn"][:, info_comp["preAttn"], i_start:i_end]

        if importance is not None:
            info_comp["importance"] = importance

        if feat is not None and info["efficiency"].activate \
            and layer_idx >= info["efficiency"].start_layer and layer_idx < 20:
            logits = info_temp["lm_head"](info_temp["norm"](feat[:, -1].detach()))
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)
            info_comp["entropy"] = entropy

def resetInfo(info, compression = None, ret=None, need_attn=False):
    info["efficiency"].reset()
    info["analysis"]["attn"] = []
    if info["analysis"]["use"]:
        # PART I. INFORMATION
        info["analysis"]["attn"] = []
        info["analysis"]["attn_ratio_type"]  = []
        info["analysis"]["attn_ratio_text"]  = []

        info["analysis"]["eos_attn_alloc"] = []
        info["analysis"]["eos_attn_effi"]  = []
        info["analysis"]["eos_attn"]       = []
        info["analysis"]["eos_attn_vis"]   = []

        # PART II. VISUALIZATION
        info["analysis"]["base"]     = []
        info["analysis"]["vidTLDR"]  = []
        info["analysis"]["fastV"]    = []
        info["analysis"]["fitPrune"] = []

        info["analysis"]["norm2"]    = []
        info["analysis"]["pred"]     = []
        info["analysis"]["logit"]    = []
        info["analysis"]["entropy"]  = []

        info["analysis"]["white_mask"] = []

        # PART III. DIFFICULTY
        info["analysis"]["complexity"] = []
        info["analysis"]["interpret"] = defaultdict(list)


    info["compression"]["img_idx"] = [None, None, None]
    if compression is not None:
        info["compression"]["use"] = True
        info["compression"]["info_type"]       = compression[0]
        info["compression"]["prune_r_layer"]   = compression[1]
        info["compression"]["prune_r"]         = compression[2]

        info["compression"]["prune_thr_layer"] = compression[3]
        info["compression"]["prune_thr"]       = compression[4]

        info["compression"]["prePrune_layer"]  = compression[5]
        info["compression"]["prePrune_thr"]    = compression[6]

        info["compression"]["diffPrune_type"]        = compression[7]
        info["compression"]["diffPrune_start"]       = compression[8]
        info["compression"]["diffPrune_drop_ratio"]  = compression[9]
        info["compression"]["diffPrune_drop_thr"]    = compression[10]
        info["efficiency"].register_diffPruning(compression[7], compression[8], compression[9], compression[10])

        info["compression"]["preAttn"]               = compression[11]

        info["compression"]["need_naive"] = [needAttn(info, l) if need_attn == 1 else False for l in range(50)] # SELECTIVE FA
        info["compression"]["need_attn"]  = [needAttn(info, l) if need_attn == 2 else False for l in range(50)] # DETOUR    FA

        info["compression"]["tau_sim"]      = 0
        info["compression"]["tau_info"]     = 0
        info["compression"]["tau_size"]     = 0
        info["compression"]["pooling_type"] = 0
        info["compression"]["mass"]         = 0
        info["compression"]["propAttn"]     = 0

    if info["compression"]["use"]:
        info["compression"]["size"] = None
        info["compression"]["source"] = None
        info["compression"]["entropy"] = None


    if ret is not None:
        if ret:
            white = torch.load(f"./temp/white_ret_pix.pt", weights_only=True)
        else:
            white = torch.load(f"./temp/white_qa_pix.pt", weights_only=True)
        info["temp"]["white"] = white


def grouping(x, group_num):
    D = x.shape[-1]
    return x.reshape(-1, group_num, D) if len(x.shape) == 2 else x.reshape(x.shape[0], -1, group_num, D)

def pruning(x, mask, prop=False):
    D = x.shape[-1] # T, D

    remain = x.masked_select(mask.reshape(-1, 1, 1)).view(-1, D)
    if prop:
        remain = torch.cat([remain, x.masked_select(~mask.reshape(-1, 1, 1)).view(-1, D).mean(dim=0, keepdim=True)], dim=0)

    return remain



class DiffDropScheduler:
    def __init__(self, enc):
        self.drop_ratio_avg = None # 레이버 별 드랍 토큰 갯수 (평균)
        self.benchmark = False
        self.Ts = []
        self.Ts_full = []
        self.activate = False
        self.enc = enc

    def getDifficulty(self, layer_idx, data):
        if layer_idx >= self.start_layer:
            if self.diff_type == 1 and len(data) == 3:
                lm_head, norm, feat = data
                logits = lm_head(norm(feat[:, -1].detach()))
                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)
                return entropy
            elif self.diff_type == 2 and len(data) == 2: # L2_Norm
                feat, feat_prev = data
                # delta =
                return
            elif self.diff_type == 3 and len(data) == 2:
                pass
        return False

        return True if True else False
    def benchmark_mode(self):
        self.reset()
        self.benchmark = True
        Ts = np.array(self.Ts_full).mean(axis=0, dtype=int)
        self.drop_ratio_avg = Ts[:-1] - Ts[1:]

    def register_diffPruning(self, diff_type, start_layer, diff_drop_ratio, diff_drop_thr):
        if type(diff_drop_ratio) != list:
            self.activate = False
        else:
            assert len(diff_drop_ratio) != len(diff_drop_thr)
            self.diff_type = diff_type
            self.start_layer = start_layer
            self.diff_drop_ratio = torch.as_tensor(diff_drop_ratio, dtype=torch.float32)
            self.diff_drop_thr = diff_drop_thr
            self.K = len(diff_drop_ratio)
            self.activate = True

    def reset(self):
        if self.activate:
            self.bin_used = torch.zeros(self.K, dtype=torch.bool)
        if len(self.Ts):
            self.Ts_full.append(self.Ts)
        self.Ts = []

    def calculate_flops(self):
        flops = self.calculate_flops_enc() if self.enc else self.calculate_flops_dec()
        return flops / 1e+9

    def add_token(self, T):
        self.Ts.append(T)

    @torch.no_grad()
    def __call__(self, T, diff, layer):
        if self.activate and (layer >= self.start_layer):
            bid = torch.bucketize(torch.tensor(10.0 - float(diff)), self.bins[1:-1], right=False).item()  # 0..K-1
            pending = ~self.bin_used[:bid+1]
            if pending.any():
                keep_factor = (1.0 - self.diff_drop_ratio[:bid+1][pending]).prod().item()
                self.bin_used[:bid+1] = True
            else:
                keep_factor = 1.0

            keep = max(1, int(torch.ceil(torch.tensor(keep_factor * T)).item()))
            return T - keep
        return 0

    def calculate_flops_enc(self):
        D_in, D, D_out = 1176, 1280, 3584
        flops = 0
        for idx, T in enumerate(self.Ts): #
            if idx == 0: # PATCH_EMBED
                flops += T * D_in * D
            elif idx == len(self.Ts) - 1: # MERGER
                flops += 4 * T * D * D + T * D * D_out
            else:
                flops += 4 * T * D * D + 2 * T * T * D
                flops += 8 * T * D * D

        return flops

    def calculate_flops_dec(self):
        D, D_kv, D_mlp = 3584, 512, 18944
        flops = 0
        for T in self.Ts[1:-1]: # 28 Layer
            flops += 2 * T * D * D + 2 * T * D * D_kv + 2 * T * T * D
            flops += 3 * (T * D * D_mlp)
        return flops




# def getDivPrune(feat, r_keep):
#     feat_norm = feat / feat.norm(dim=-1, keepdim=True)
#     feat_sim = 1 - torch.mm(feat_norm, feat_norm.t())
#
#     s = torch.empty(r_keep, dtype=torch.long, device=feat.device)
#     for i in range(r_keep):
#         if i == 0:
#             m2 = feat_sim  # (576, 576)
#         else:
#             m2 = torch.index_select(feat_sim, 0, torch.index_select(s, 0, torch.arange(0, i, device=cosine_matrix.device)))  # (1, 576)
#
#         if i == 0:
#             scores = torch.topk(m2, 2, dim=0, largest=False).values[1, :]  # 576
#         else:
#             scores = torch.min(m2, dim=0).values  # 576
#
#         phrase_to_add_idx = torch.argmax(scores)  # 234
#         s[i] = phrase_to_add_idx
#     return s