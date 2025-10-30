from joonmyung.compression.compression import needAttn
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
import torch

def getDivPrune(feat, r_type, r_prune, i_start=None, i_end=None):
    if len(feat.shape) == 3:
        feat = feat[0]
    if i_start is not None:
        feat = feat[i_start:i_end]
    T_vis, device = feat.shape[0], feat.device

    feat_norm = feat / feat.norm(dim=-1, keepdim=True)
    feat_dist = 1 - (feat_norm @ feat_norm.t()) # (B, T, D)
    r_keep = int(T_vis * (1 - r_prune))

    unprune_idx = torch.empty(T_vis, dtype=torch.long, device=feat.device)
    for i in range(T_vis):
        m2 = feat_dist if i == 0 else torch.index_select(feat_dist, 0, torch.index_select(unprune_idx, 0, torch.arange(0, i, device=feat.device)))  # (1, 576)
        scores = torch.topk(m2, 2, dim=0, largest=False).values[1, :] if i == 0 else torch.min(m2, dim=0).values  # 576
        add_score, add_idx = torch.max(scores, dim=0)
        unprune_idx[i] = add_idx
        if r_type == 0 and i == r_keep:
            break
        if r_type == 1 and add_score < r_prune:
            break
    unprune_idx = unprune_idx[:i]
    mask = torch.zeros(T_vis, dtype=torch.float32, device=device)
    mask[unprune_idx] = 1
    return mask


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
def getRepShift(feat, feat_prev, start = None, end = None):
    return torch.norm(feat - feat_prev, p=2, dim=-1)[:, start:end]

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
    if not enc and N != 1 and end: # DECODER
        prt      = splitAttn(attn_headavg[:, :start],        start = start, end = end)
        vis      = splitAttn(attn_headavg[:, start:end],     start = start, end = end)
        txt      = splitAttn(attn_headavg[:, end:],          start = start, end = end)
        txt_e    = splitAttn(attn_headavg[:, -1:],           start = start, end = end)
        result_full = torch.stack([prt, vis, txt, txt_e], dim=1)
        results_text = attn_headavg[:, end:, end:].sum(dim=1) / torch.arange(N - end, 0, -1, device=attn.device)
        return result_full, results_text
    if not enc and N != 1:  # DECODER : ONLY TEXT
        results_text = attn_headavg.sum(dim=1)
        return None, results_text
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

def l2Norm(A, B = None, d = 2):
    if B == None:
        return torch.round(A.norm(dim=-1), decimals=d)
    else:
        return torch.round((A - B).norm(dim=-1), decimals=d)

def cossim(A, B, d):
    return torch.round(torch.cosine_similarity(A, B, dim=-1), decimals=d)

def getDifficulty(info, input_B, input_B_N, output, output_res, name, input_L = None):
    if input_B.shape[1] != 1:
        input_B, input_B_N, output, output_res = input_B[0, -1].to(torch.float32), input_B_N[0, -1].to(torch.float32), output[0, -1].to(torch.float32), output_res[0, -1].to(torch.float32)

        info["analysis"]["interpret"][f"{name}_delta_l2"].append(l2Norm(input_B,     output, d=2)) # EXPERIMENTS ✓
        info["analysis"]["interpret"][f"{name}_delta_l2_N"].append(l2Norm(input_B_N, output, d=2))
        info["analysis"]["interpret"][f"{name}_delta_l2_R"].append(l2Norm(input_B,   output_res, d=2))

        info["analysis"]["interpret"][f"{name}_cos"].append(cossim(input_B,     output, d=3))
        info["analysis"]["interpret"][f"{name}_cos_N"].append(cossim(input_B_N, output, d=3))
        info["analysis"]["interpret"][f"{name}_cos_R"].append(cossim(input_B,   output_res, d=3))

        info["analysis"]["interpret"][f"{name}_l2"].append(l2Norm(output, d=2))
        info["analysis"]["interpret"][f"{name}_l2_R"].append(l2Norm(output_res, d=2))
        if input_L is not None:
            input_L = input_L[0, -1].to(torch.float32)
            info["analysis"]["interpret"][f"FULL_delta_l2_L"].append(l2Norm(input_L, output_res, d=2))
            info["analysis"]["interpret"][f"FULL_cos_L"].append(cossim(input_L, output_res, d=3))





def getAnalysis(info, attn = None, feat = None, feat_mlp = None, feat_input = None, enc= False, layer_idx = False):
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
            if enc: # ENCODER
                info_ana["vidTLDR"].append(unPrune(getImpVidTLDR(attn, i_start, i_end), source_vis))
            else: # DECODER
                info_ana["attn"].append(attn.mean(dim=(0, 1)))
                ratio_type, ratio_text = getAttnRatio(attn, start=i_start, end=i_end, cls=cls, enc=enc)
                info_ana["attn_ratio_type"].append(ratio_type)
                info_ana["attn_ratio_text"].append(ratio_text)
                if i_start:
                    attn_alloc_full = torch.stack([attn.mean(dim=(0, 1))[-1][:i_start].sum(dim=-1), attn.mean(dim=(0, 1))[-1][i_start:i_end].sum(dim=-1), attn.mean(dim=(0, 1))[-1][i_end:i_len - 1].sum(dim=-1), attn.mean(dim=(0, 1))[-1][i_len - 1:].sum(dim=-1)])
                    attn_alloc_token = torch.stack([attn.mean(dim=(0, 1))[-1][:i_start].mean(dim=-1), attn.mean(dim=(0, 1))[-1][i_start:i_end].mean(dim=-1), attn.mean(dim=(0, 1))[-1][i_end:i_len - 1].mean(dim=-1), attn.mean(dim=(0, 1))[-1][i_len - 1:].mean(dim=-1)])
                    info_ana["eos_attn_alloc"].append(attn_alloc_full)
                    info_ana["eos_attn_effi"].append(attn_alloc_token / attn_alloc_token[1])

                    info_ana["fastV"].append(getImpFastV(attn, i_start, i_end))
                    info_ana["fitPrune"].append(getImpFitprune(attn, i_start, i_end))


        if feat is not None and feat.shape[1] != 1:
            feat = feat.detach()
            info_ana["norm2"].append(unPrune(getL2Norm(feat, i_start, i_end), source_vis))
            if feat_mlp is not None: info_ana["shift"].append(unPrune(getRepShift(feat_mlp, feat_input, i_start, i_end), source_vis))
            # info_ana["feat"].append(feat)
            feat_norm = F.normalize(feat.to(torch.float32), dim=-1)  # ↑ : 단순
            complexity = (1 - (feat_norm @ feat_norm.transpose(-1, -2))).mean(dim=-1)  # ↑ : 복잡
            # complexity = (1 - (feat_norm @ feat_norm.transpose(-1, -2))).mean()  # ↑ : 복잡
            info_ana["img_complexity"].append(complexity)
            if not enc: # DECODER : Entropy / Logit / PRED
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
        elif feat is not None and info_comp["info_type"] == 9 and layer_idx == info_comp["prune_layer"]:
            importance = getDivPrune(feat, info_comp["r_type"], info_comp["prune_r"], i_start, i_end)
        elif feat is not None and info_comp["info_type"] == 10: # RANDOM
            T = i_end - i_start if i_start else feat.shape[1]
            importance = torch.rand((1, T), device=feat.device)

        if importance is not None:
            info_comp["importance"] = importance

        if feat is not None and info["efficiency"].activate:
            info["efficiency"].setDifficulty(info_comp, layer_idx, [0, feat, info_temp["lm_head"], info_temp["norm"]])


def resetInfo(info, info_comp = None, info_ret = None, ret=None, enc=None, need_attn=False, ana = True, device = "cuda"):
    if info["analysis"]["use"] and ana:
        # PART I. INFORMATION
        info["analysis"]["attn_ratio_type"]  = []
        info["analysis"]["attn_ratio_text"]  = []

        info["analysis"]["eos_attn_alloc"] = []
        info["analysis"]["eos_attn_effi"]  = []
        info["analysis"]["eos_attn"]       = []
        info["analysis"]["eos_attn_vis"]   = []

        # PART II. VISUALIZATION
        info["analysis"]["attn"]     = []
        info["analysis"]["feat"]     = []

        info["analysis"]["base"]     = []
        info["analysis"]["vidTLDR"]  = []
        info["analysis"]["fastV"]    = []
        info["analysis"]["fitPrune"] = []

        info["analysis"]["shift"]    = []
        info["analysis"]["norm2"]    = []
        info["analysis"]["pred"]     = []
        info["analysis"]["logit"]    = []
        info["analysis"]["entropy"]  = []

        info["analysis"]["white_mask"] = []

        # PART III. DIFFICULTY
        info["analysis"]["img_complexity"] = []
        info["analysis"]["interpret"] = defaultdict(list)



    if info_ret is not None:
        info["retrieval"]["use"] = True
        info["retrieval"]["ret_type"]  = info_ret[0]
        info["retrieval"]["token_idx"] = info_ret[1]
        info["retrieval"]["layer_idx"] = info_ret[2]

    if info["retrieval"]["use"]:
        info["retrieval"]["importance"] = []


    info["compression"]["img_idx"] = [None, None, None]
    if info_comp is not None:
        info["compression"]["use"] = True
        info["compression"]["info_type"]             = info_comp[0]
        info["compression"]["r_type"]                = info_comp[1]
        info["compression"]["prune_layer"]           = info_comp[2]
        info["compression"]["prune_r"]               = info_comp[3]
        info["compression"]["diffPrune_type"]        = info_comp[4]
        info["compression"]["diffPrune_start"]       = info_comp[5]
        info["compression"]["diffPrune_drop_ratio"]  = info_comp[6]
        info["compression"]["diffPrune_drop_thr"]    = info_comp[7]
        info["efficiency"].register_diffPruning(info_comp[1], info_comp[4], info_comp[5], info_comp[6], info_comp[7], device)

        info["compression"]["preAttn"]               = info_comp[8]
        if info_comp[10] or info_comp[11] or info_comp[12]:
            info["compression"]["prePrune_layer"]        = info_comp[9]  # 토큰 제거할 레이어 넘버
            info["compression"]["prePrune_ratio"]        = info_comp[10] # 흰색 배경 제거 : 흰색 픽색 비율 Threshold
            info["compression"]["prePrune_ret_r"]        = info_comp[11] # 유사도 ↓ 제거 : 제거 토큰 비율
            info["compression"]["prePrune_ret_thr"]      = info_comp[12] # 유사도 ↓ 제거 : 제거 토큰 Threshold
            info["compression"]["prePrune_ret_kernel"]   = info_comp[13] # 유사도 ↓ 제거 : 커널 사이즈
            info["compression"]["prePrune_ret_str"]      = info_comp[14] # 유사도 ↓ 제거 : 커널 강도
            info["compression"]["prePrune_ret_norm"]     = info_comp[15] # 유사도 ↓ 제거 : 정규화


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
        info["compression"]["difficulty"] = None
        # info["compression"]["mask_block"] = True

    info["efficiency"].reset()
    if ret != None:
        info["efficiency"].retrieval = ret
    if enc == True:
        if ret and enc:
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
        self.activate = False
        self.T = 0    # 입력 토큰 갯수
        self.Ts = []  # 레이어 별 샘플 갯수
        self.Ts_full = [] # 전체 데이터 셋 레이어 별 샘플 갯수
        self.difficulty = []
        self.enc = enc
        self.diff_type_input = [[1,3,5,7,9,11,13], [2,4,6,8,10,12,14], [15, 16, 17, 18]]
        self.device = None
        self.retrieval = False



    def setDifficulty(self, info_comp, layer_idx, data):
        if self.activate and layer_idx >= self.start_layer:
            data_from, feat = data[:2]
            if self.diff_type == data_from and feat.shape[1] > 1:
                if data_from == 0: # ENTROPY
                    feat, lm_head, norm = data[1:]
                    logits = lm_head(norm(feat[:, -1].detach()))
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp()
                    difficulty = 10 + (probs * log_probs).sum(dim=-1)
                elif data_from in self.diff_type_input[0]: # Delta L2_Norm
                    feat, feat_prev = data[1:]
                    difficulty = (feat[:, -1] - feat_prev[:, -1]).norm(dim=-1)
                elif data_from in self.diff_type_input[1]: # Cosine Similarity
                    feat, feat_prev = data[1:]
                    difficulty = 1 - torch.cosine_similarity(feat[:, -1], feat_prev[:, -1], dim=-1)
                elif data_from in self.diff_type_input[2]: # L2_Norm
                    feat = data[1]
                    difficulty = feat[:, -1].norm(dim=-1)
                self.difficulty.append(difficulty)

                info_comp["difficulty"] = difficulty

    def benchmark_mode(self):
        self.reset()
        self.benchmark = True
        Ts_full = np.array(self.Ts_full)
        if len(Ts_full.shape) == 3:
            Ts_full = Ts_full[:, :, 1:] - Ts_full[:, :, :-1]
            Ts = Ts_full.mean(axis=(0, 2), dtype=int)
        else:
            Ts = Ts_full.mean(axis=0, dtype=int)


        self.drop_ratio_avg = Ts[:-1] - Ts[1:]

    def register_diffPruning(self, r_type, diff_type, start_layer, diff_drop_thr, diff_drop_ratio, device):
        if type(diff_drop_thr) != list:
            self.activate = False
        else:
            assert len(diff_drop_ratio) == len(diff_drop_thr)
            self.r_type = r_type
            self.device = device
            self.diff_type = diff_type
            self.start_layer = start_layer
            self.diff_drop_ratio = torch.tensor(diff_drop_ratio, device=device)
            self.diff_drop_thr = torch.tensor(diff_drop_thr, device=device)
            self.K = len(diff_drop_ratio)
            self.activate = True


    def reset(self):
        if self.activate:
            self.bin_used = torch.ones(self.K, dtype=torch.bool, device=self.device)
            self.difficulty = []
        if len(self.Ts):
            self.Ts_full.append(self.Ts)
        self.Ts = []

    def calculate_flops(self, Ds):
        if self.retrieval:
            flops = self.calculate_flops_enc_ret(Ds) if self.enc else self.calculate_flops_dec_ret(Ds)
        else:
            flops = self.calculate_flops_enc(Ds) if self.enc else self.calculate_flops_dec(Ds)
        return flops / 1e+9

    def register_T(self, T):
        self.T = T

    def add_token(self, T):
        self.Ts.append(T)

    @torch.no_grad()
    def __call__(self, difficulty):
        if self.activate and difficulty is not None:
            activate = self.bin_used * (self.diff_drop_thr < difficulty)
            if activate.sum():
                if self.r_type == 0:
                    T_prune = 1 - (1 - self.diff_drop_ratio[activate]).prod()
                    self.bin_used = self.bin_used * ~activate
                    return T_prune
                else:
                    bin_idx = (self.bin_used == True).nonzero(as_tuple=True)[0][0].item()
                    thr_prune = self.diff_drop_ratio[bin_idx]
                    self.bin_used[bin_idx] = False
                    return thr_prune
        return 0

    def calculate_flops_enc_ret(self, Ds):
        D_in, D, D_mlp = Ds
        flops = 0

        # 1. PATCH EMBED
        flops += self.Ts[0] * D_in * D

        # 2. Layer
        for idx, T in enumerate(self.Ts[1:-1]):
            flops += 4 * T * D * D + 2 * T * T * D
            flops += 2 * T * D * D_mlp

        return flops

    def calculate_flops_enc(self, Ds):
        D_in, D, D_out = Ds
        flops = 0
        BTs = torch.stack([torch.Tensor(v) for v in self.Ts], dim=-1)
        BTs = (BTs[1:] - BTs[:-1]).tolist()

        # 1. PATCH EMBED
        for Ts in BTs:
            flops += Ts[0] * D_in * D

            # 2. Layer
            for idx, T in enumerate(Ts[1:-1]):
                flops += 4 * T * D * D + 2 * T * T * D
                flops += 8 * T * D * D

            # 3. MERGER / QA
            flops += 4 * Ts[-1] * (D * D + D * D_out)

        return flops

    def calculate_flops_dec_ret(self, Ds):
        D, D_kv, D_mlp, D_lora = Ds

        flops = 0
        # 1. LAYER
        for T in self.Ts[1:-1]: # 28 Layer
            # SA : Q_proj
            flops += T * D * D + 2 * T * D_lora * D
            # SA : KV_proj
            flops += 2 * T * (D * D_kv + D * D_lora + D_lora * D_kv)
            # SA : proj
            flops += T * D * D + 2 * T * D_lora * D
            # SA : ATTN
            flops += T * T * D

            # MLP
            flops += 3 * T * (D * D_mlp + D * D_lora + D_lora * D_mlp)

        return flops

    def calculate_flops_dec(self, Ds):
        D, D_kv, D_mlp = Ds

        flops = 0
        # 1. LAYER
        for T in self.Ts[1:-1]: # 28 Layer
            flops += 2 * T * D * D + 2 * T * D * D_kv + 2 * T * T * D
            flops += 3 * (T * D * D_mlp)


        return flops