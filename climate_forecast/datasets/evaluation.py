# climate_forecast/datasets/evaluation.py

from typing import List, Dict
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from collections import defaultdict

def _denormalize_log(data: torch.Tensor, clip_value: float) -> torch.Tensor:
    if clip_value <= 0: return data.float()
    scale_factor = torch.log1p(torch.tensor(clip_value, device=data.device, dtype=torch.float32))
    if scale_factor == 0: return torch.zeros_like(data, dtype=torch.float32)
    denormalized = torch.expm1(data.float() * scale_factor)
    return torch.clamp(denormalized, 0.0, clip_value)

class ClimateSkillScore(Metric):
    full_state_update = True

    def __init__(
        self,
        config: Dict, # Added config parameter
        seq_len: int,
        layout: str,
        threshold_list: List[float],
        spatial_scales: List[int] = [1, 2, 4, 8],
        denormalize_clip_value: float = 100.0,
        dist_sync_on_step=False
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.config = config # Store config
        self.seq_len = seq_len
        self.layout = layout
        self.threshold_list = sorted(threshold_list)
        self.spatial_scales = sorted(spatial_scales)
        self.denormalize_clip_value = denormalize_clip_value
        
        # summed counts per (lead, scale, threshold): hits, misses, false alarms, FSS MSE, FSS reference
        shape = (seq_len, len(self.spatial_scales), len(self.threshold_list), 5)
        self.add_state("stats", default=torch.zeros(shape, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds_denorm = _denormalize_log(preds, self.denormalize_clip_value)
        target_denorm = _denormalize_log(target, self.denormalize_clip_value)
        for t_idx in range(self.seq_len):                      # layout NTHWC
            p_t = preds_denorm[:, t_idx].permute(0, 3, 1, 2)
            o_t = target_denorm[:, t_idx].permute(0, 3, 1, 2)
            for si, scale in enumerate(self.spatial_scales):
                # categorical scores on upscaled fields (mean rain in scale x scale boxes)
                p_pool = F.avg_pool2d(p_t, kernel_size=scale) if scale > 1 else p_t
                o_pool = F.avg_pool2d(o_t, kernel_size=scale) if scale > 1 else o_t
                for ki, thr in enumerate(self.threshold_list):
                    pb, ob = (p_pool >= thr).float(), (o_pool >= thr).float()
                    # FSS: binarise at native resolution, then neighbourhood fractions (moving window)
                    pf, of = (p_t >= thr).float(), (o_t >= thr).float()
                    if scale > 1:
                        pf, of = F.avg_pool2d(pf, scale, stride=1), F.avg_pool2d(of, scale, stride=1)
                    self.stats[t_idx, si, ki] += torch.stack([
                        (pb * ob).sum(), ((1 - pb) * ob).sum(), (pb * (1 - ob)).sum(),
                        ((pf - of) ** 2).sum(), (pf ** 2).sum() + (of ** 2).sum()]).double()

    def compute(self) -> Dict:
        if not torch.any(self.stats):
            return {}
        zero = torch.tensor(0.0, device=self.stats.device)
        results = {}
        interval = self.config['data'].get("time_interval_minutes", 30)
        for t_idx in range(self.seq_len):
            for si, scale in enumerate(self.spatial_scales):
                for ki, thr in enumerate(self.threshold_list):
                    hits, misses, false_alarms, mse, ref = self.stats[t_idx, si, ki].float()
                    pod = hits / (hits + misses) if hits + misses > 0 else zero
                    csi = hits / (hits + misses + false_alarms) if hits + misses + false_alarms > 0 else zero
                    far = false_alarms / (hits + false_alarms) if hits + false_alarms > 0 else zero
                    sucr = 1.0 - far if hits + false_alarms > 0 else zero
                    bias = (hits + false_alarms) / (hits + misses) if hits + misses > 0 else zero
                    fss = 1.0 - mse / ref if ref > 0 else zero
                    base = f"details/T+{(t_idx + 1) * interval}min/{scale}x{scale}/{thr:.2f}mm"
                    for name, val in (("csi", csi), ("bias", bias), ("fss", fss), ("pod", pod), ("far", far), ("sucr", sucr)):
                        results[f"{base}/{name}"] = val

        # Calculate averages for native scale (1x1) over all lead times
        avg_scores = defaultdict(list)
        for key, value in results.items():
            if 'details/' in key and '/1x1/' in key:
                metric_name = key.split('/')[-1]
                avg_scores[f"{metric_name}_avg_epoch"].append(value)
        
        for key, val_list in avg_scores.items():
            if val_list:
                results[key] = torch.stack(val_list).mean()

        return {f"val/{k}": v for k, v in results.items()}