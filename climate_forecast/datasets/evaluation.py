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
        
        self.add_state("stats", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds_denorm = _denormalize_log(preds, self.denormalize_clip_value)
        target_denorm = _denormalize_log(target, self.denormalize_clip_value)

        # Assuming layout is NTHWC
        for t_idx in range(self.seq_len):
            preds_t = preds_denorm[:, t_idx, ...].permute(0, 3, 1, 2) # N, C, H, W
            target_t = target_denorm[:, t_idx, ...].permute(0, 3, 1, 2)

            for scale in self.spatial_scales:
                p_pool = F.avg_pool2d(preds_t, kernel_size=scale) if scale > 1 else preds_t
                t_pool = F.avg_pool2d(target_t, kernel_size=scale) if scale > 1 else target_t

                for threshold in self.threshold_list:
                    p_binary = (p_pool >= threshold).float()
                    t_binary = (t_pool >= threshold).float()

                    hits = torch.sum(p_binary * t_binary)
                    misses = torch.sum((1 - p_binary) * t_binary)
                    false_alarms = torch.sum(p_binary * (1 - t_binary))
                    
                    # Correct FSS calculation on binarized fields
                    fss_ms_error = F.mse_loss(p_binary, t_binary, reduction='sum')
                    fss_ref_variance = torch.sum(p_binary**2) + torch.sum(t_binary**2)

                    self.stats.append(torch.stack([
                        torch.tensor(t_idx, device=self.device),
                        torch.tensor(scale, device=self.device),
                        torch.tensor(threshold, device=self.device),
                        hits, misses, false_alarms,
                        fss_ms_error, fss_ref_variance
                    ]))

    def compute(self) -> Dict:
        if not self.stats: return {}

        aggregated_stats = defaultdict(lambda: torch.zeros(5, device=self.device))
        for item in self.stats:
            t_idx, scale, threshold = int(item[0].item()), int(item[1].item()), float(item[2].item())
            stats = item[3:]
            key = f"{t_idx}_{scale}_{threshold}"
            aggregated_stats[key] += stats

        results = {}
        for key, value in aggregated_stats.items():
            t_idx_str, scale_str, threshold_str = key.split('_')
            hits, misses, false_alarms, mse, ref_var = value

            pod = hits / (hits + misses) if (hits + misses) > 0 else torch.tensor(0.0, device=self.device)
            csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else torch.tensor(0.0, device=self.device)
            bias = (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else torch.tensor(0.0, device=self.device)
            fss = 1.0 - (mse / ref_var) if ref_var > 0 else torch.tensor(0.0, device=self.device)

            lead_time = (int(t_idx_str) + 1) * self.config['data'].get("time_interval_minutes", 30)
            time_key = f"T+{lead_time}min"
            scale_key = f"{scale_str}x{scale_str}"
            thresh_key = f"{float(threshold_str):.2f}mm"
            
            results[f"details/{time_key}/{scale_key}/{thresh_key}/csi"] = csi
            results[f"details/{time_key}/{scale_key}/{thresh_key}/bias"] = bias
            results[f"details/{time_key}/{scale_key}/{thresh_key}/fss"] = fss
            results[f"details/{time_key}/{scale_key}/{thresh_key}/pod"] = pod

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