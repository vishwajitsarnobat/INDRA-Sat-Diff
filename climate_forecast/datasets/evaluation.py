# climate_forecast/datasets/evaluation.py

import warnings
from typing import Optional, Sequence, Union, Dict, Tuple, Set

import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange
from torchmetrics import Metric
from torch.nn import AvgPool2d

# A default clip value used for denormalization if not specified.
DEFAULT_CLIP_VALUE = 100.0

def _denormalize_log_np(data: np.ndarray, clip_value: float = DEFAULT_CLIP_VALUE) -> np.ndarray:
    """
    Denormalizes log-transformed numpy data back to the physical scale.
    This is the inverse of a `log1p` style normalization.
    """
    if clip_value <= 0:
        warnings.warn(f"Clip value ({clip_value}) is non-positive. Returning original data.")
        return data.astype(np.float32)

    data = data.astype(np.float32)
    scale_factor = np.log1p(clip_value)

    if abs(scale_factor) < 1e-9:
        warnings.warn(f"Scale factor is near zero ({scale_factor}). Returning original data.")
        return data

    denormalized = np.expm1(data * scale_factor)
    # Handle potential infinities and NaNs that can arise from the exponential function.
    denormalized = np.nan_to_num(denormalized, nan=0.0, posinf=clip_value, neginf=0.0)
    denormalized = np.clip(denormalized, 0.0, clip_value)

    return denormalized.astype(np.float32)

def _threshold_and_clean_np(
    target_denorm: np.ndarray,
    pred_denorm: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Thresholds denormalized NumPy arrays to create binary event maps."""
    t_bin = (target_denorm >= threshold).astype(np.float32)
    p_bin = (pred_denorm >= threshold).astype(np.float32)
    # Ensure that any NaN values in the original data do not contribute to events.
    is_nan = np.logical_or(np.isnan(target_denorm), np.isnan(pred_denorm))
    t_bin[is_nan] = 0.0
    p_bin[is_nan] = 0.0
    return t_bin, p_bin

def _calc_contingency_np(
    t_bin: np.ndarray,
    p_bin: np.ndarray,
    aggregation_axes: Optional[Tuple[int, ...]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates contingency table elements (Hits, Misses, False Alarms) using NumPy."""
    hits = t_bin * p_bin
    misses = t_bin * (1.0 - p_bin)
    fas = (1.0 - t_bin) * p_bin

    # Sum over the specified axes to get total counts.
    hits = np.sum(hits, axis=aggregation_axes)
    misses = np.sum(misses, axis=aggregation_axes)
    fas = np.sum(fas, axis=aggregation_axes)

    return hits, misses, fas

def _calculate_fss_components_pytorch(
    t_bin: torch.Tensor,
    p_bin: torch.Tensor,
    scale: int,
    aggregation_dims: Optional[Tuple[int, ...]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates FSS components (MSE and Reference MSE) using PyTorch's AvgPool2d.
    This function contains the critical fix for the FSS calculation.
    """
    if scale < 1: raise ValueError(f"FSS scale must be >= 1, but got {scale}")

    # The input tensor is guaranteed to be 4D (N, T, H, W) by the update method.
    input_shape = t_bin.shape
    orig_spatial_dims = input_shape[-2:]
    leading_dims = input_shape[:-2]
    num_total_frames = int(np.prod(leading_dims))

    # --- THIS IS THE CORE FSS FIX ---
    # Reshape the tensor from (N, T, H, W) to (N*T, 1, H, W).
    # This treats each frame as a separate item in a batch with a single channel,
    # which is the format required by AvgPool2d for independent spatial pooling.
    t_bin_reshaped = t_bin.reshape(num_total_frames, 1, *orig_spatial_dims)
    p_bin_reshaped = p_bin.reshape(num_total_frames, 1, *orig_spatial_dims)

    if scale > 1:
        padding = scale // 2
        pool = AvgPool2d(kernel_size=scale, stride=1, padding=padding, count_include_pad=False).to(t_bin.device)
        S_t_pooled, S_p_pooled = pool(t_bin_reshaped), pool(p_bin_reshaped)
        # Reshape back to original leading dims (e.g., N, T, H, W)
        S_t = S_t_pooled.reshape(*leading_dims, *S_t_pooled.shape[-2:])
        S_p = S_p_pooled.reshape(*leading_dims, *S_p_pooled.shape[-2:])
    else: # If scale is 1, no pooling is needed.
        S_t, S_p = t_bin, p_bin

    term_mse = (S_p - S_t)**2
    term_ref = S_p**2 + S_t**2

    fss_mse_sum = torch.sum(term_mse, dim=aggregation_dims) if aggregation_dims is not None else torch.sum(term_mse)
    fss_ref_sum = torch.sum(term_ref, dim=aggregation_dims) if aggregation_dims is not None else torch.sum(term_ref)

    return fss_mse_sum, fss_ref_sum


# ====================================================================
# Main Metric Class
# ====================================================================

class ClimateSkillScore(Metric):
    """
    A comprehensive torchmetrics-based skill score calculator for precipitation nowcasting.

    This metric computes several standard verification scores (CSI, POD, SUCR, BIAS, FSS)
    for both a predictive model and a persistence baseline. It handles data denormalization,
    thresholding, and aggregation internally. It is robust to different input tensor layouts
    (e.g., NTHWC, NCTHW) thanks to the use of `einops`.

    Args:
        layout (str): The layout of the input tensors (e.g., "NTHWC").
        mode (str): Defines aggregation behavior.
                    '0' or '2': Aggregate over all dimensions (N, T, H, W).
                    '1': Aggregate over N, H, W, preserving the T dimension for per-timestep scores.
        seq_len (int): Required if mode='1', specifies the length of the time dimension.
        threshold_list (Sequence[float]): A list of physical thresholds (e.g., in mm/hr) to evaluate.
        metrics_list (Sequence[str]): A list of metrics to compute from {"csi", "pod", "sucr", "bias", "fss"}
                                      and their persistence versions like {"persistence_csi"}.
        fss_scale (int): The neighborhood size (in pixels) for the Fractions Skill Score (FSS).
        eps (float): A small epsilon value to prevent division by zero in metric calculations.
        denormalize_clip_value (float): The clip value used during normalization, needed for denormalization.
        dist_sync_on_step (bool): For distributed training.
    """
    full_state_update: bool = False
    higher_is_better: Optional[bool] = None

    def __init__(self,
                 layout: str = "NTHWC",
                 mode: str = "0",
                 seq_len: Optional[int] = None,
                 threshold_list: Sequence[float] = (2.5, 7.6, 16, 50),
                 metrics_list: Sequence[str] = ("csi", "pod", "sucr", "bias", "fss", "persistence_csi"),
                 fss_scale: int = 10,
                 eps: float = 1e-6,
                 denormalize_clip_value: float = DEFAULT_CLIP_VALUE,
                 dist_sync_on_step: bool = False,
                 ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if not all(c in layout.upper() for c in "NTHW"): raise ValueError(f"Layout '{layout}' must contain N, T, H, W")
        if fss_scale < 1: raise ValueError(f"fss_scale must be >= 1, got {fss_scale}")

        self.layout, self.has_channel = layout.upper(), 'C' in layout.upper()
        self.threshold_list, self.fss_scale, self.eps = sorted(list(threshold_list)), fss_scale, eps
        self.mode, self.seq_len, self.denormalize_clip_value = str(mode), seq_len, denormalize_clip_value

        self.model_metrics, self.persistence_metrics = set(), set()
        for metric in set(m.lower() for m in metrics_list):
            if metric.startswith("persistence_"):
                self.persistence_metrics.add(metric.replace("persistence_", ""))
            else:
                self.model_metrics.add(metric)
        self.calc_persistence = len(self.persistence_metrics) > 0

        # Aggregation axes are now based on the standardized 'N T H W' internal layout
        if self.mode in ("0", "2"): # Aggregate everything
            state_shape, self.agg_axes = (len(self.threshold_list),), (0, 1, 2, 3)
        elif self.mode == "1": # Keep T dimension
            if seq_len is None: raise ValueError("seq_len must be provided for mode '1'")
            state_shape = (len(self.threshold_list), seq_len)
            self.agg_axes = (0, 2, 3) # Aggregate N, H, W
        else:
            raise ValueError(f"Mode '{mode}' not supported. Use '0', '1', or '2'.")

        # Initialize states based on requested metrics
        contingency_keys = {"pod", "csi", "sucr", "bias"}
        if self.model_metrics.intersection(contingency_keys):
            self.add_state("hits", default=torch.zeros(state_shape), dist_reduce_fx="sum")
            self.add_state("misses", default=torch.zeros(state_shape), dist_reduce_fx="sum")
            self.add_state("fas", default=torch.zeros(state_shape), dist_reduce_fx="sum")
        if "fss" in self.model_metrics:
            self.add_state("fss_mse", default=torch.zeros(state_shape), dist_reduce_fx="sum")
            self.add_state("fss_ref", default=torch.zeros(state_shape), dist_reduce_fx="sum")
        if self.persistence_metrics.intersection(contingency_keys):
            self.add_state("hits_pers", default=torch.zeros(state_shape), dist_reduce_fx="sum")
            self.add_state("misses_pers", default=torch.zeros(state_shape), dist_reduce_fx="sum")
            self.add_state("fas_pers", default=torch.zeros(state_shape), dist_reduce_fx="sum")
        if "fss" in self.persistence_metrics:
            self.add_state("fss_mse_pers", default=torch.zeros(state_shape), dist_reduce_fx="sum")
            self.add_state("fss_ref_pers", default=torch.zeros(state_shape), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.shape != target.shape: raise ValueError("Pred and target shapes must match")
        if pred.ndim != len(self.layout): raise ValueError("Input ndim must match layout length")

        # --- Robust Layout Handling ---
        # Standardize any input layout to a consistent internal format.
        input_pattern = " ".join(self.layout.lower())
        output_pattern = "n t h w c" if self.has_channel else "n t h w"
        target_rearranged = rearrange(target, f"{input_pattern} -> {output_pattern}")
        pred_rearranged = rearrange(pred, f"{input_pattern} -> {output_pattern}")
        # Squeeze channel dim if it exists and is singular, resulting in a clean 4D (N,T,H,W) tensor.
        if self.has_channel:
            target_rearranged = target_rearranged.squeeze(-1)
            pred_rearranged = pred_rearranged.squeeze(-1)
        # All subsequent operations can now safely assume a 4D NTHW layout.

        pred_np = pred_rearranged.detach().cpu().numpy()
        target_np = target_rearranged.detach().cpu().numpy()

        pred_denorm = _denormalize_log_np(pred_np, self.denormalize_clip_value)
        target_denorm = _denormalize_log_np(target_np, self.denormalize_clip_value)
        to_torch = lambda val: torch.from_numpy(np.asanyarray(val)).to(self.device).float()

        # --- Main Model Metrics Calculation ---
        for i, T in enumerate(self.threshold_list):
            t_bin_np, p_bin_np = _threshold_and_clean_np(target_denorm, pred_denorm, T)
            
            if hasattr(self, "hits"):
                h, m, f = _calc_contingency_np(t_bin_np, p_bin_np, self.agg_axes)
                self.hits[i] += to_torch(h)
                self.misses[i] += to_torch(m)
                self.fas[i] += to_torch(f)
            
            if hasattr(self, "fss_mse"):
                t_bin = torch.from_numpy(t_bin_np).to(self.device)
                p_bin = torch.from_numpy(p_bin_np).to(self.device)
                mse, ref = _calculate_fss_components_pytorch(t_bin, p_bin, self.fss_scale, self.agg_axes)
                self.fss_mse[i] += mse
                self.fss_ref[i] += ref

        # --- Persistence Baseline Metrics Calculation ---
        if self.calc_persistence and target_np.shape[1] > 1: # time dim is axis 1
            pred_pers_denorm = target_denorm[:, :-1]
            target_pers_denorm = target_denorm[:, 1:]
            
            for i, T in enumerate(self.threshold_list):
                t_bin_p_np, p_bin_p_np = _threshold_and_clean_np(target_pers_denorm, pred_pers_denorm, T)
                
                if hasattr(self, "hits_pers"):
                    h_p, m_p, f_p = _calc_contingency_np(t_bin_p_np, p_bin_p_np, self.agg_axes)
                    if self.mode == "1": # Pad time dimension for alignment if needed
                        h_p, m_p, f_p = [np.pad(arr, ((1, 0),), 'constant') for arr in (h_p, m_p, f_p)]
                    self.hits_pers[i] += to_torch(h_p)
                    self.misses_pers[i] += to_torch(m_p)
                    self.fas_pers[i] += to_torch(f_p)
                
                if hasattr(self, "fss_mse_pers"):
                    t_p = torch.from_numpy(t_bin_p_np).to(self.device)
                    p_p = torch.from_numpy(p_bin_p_np).to(self.device)
                    mse_p, ref_p = _calculate_fss_components_pytorch(t_p, p_p, self.fss_scale, self.agg_axes)
                    if self.mode == "1":
                        mse_p, ref_p = [F.pad(val, (1, 0)) for val in (mse_p, ref_p)]
                    self.fss_mse_pers[i] += mse_p
                    self.fss_ref_pers[i] += ref_p
    
    def _calculate_scores(self, H, M, F, mse, ref, metrics_to_calc: Set[str]) -> Dict[str, torch.Tensor]:
        """Helper to calculate final scores from accumulated contingency and FSS components."""
        scores = {}
        if metrics_to_calc.intersection({"pod", "csi", "sucr", "bias"}):
            pod = H / (H + M + self.eps)
            sucr = H / (H + F + self.eps)
            csi = H / (H + M + F + self.eps)
            bias = (H + F) / (H + M + self.eps)
            if "pod" in metrics_to_calc: scores["pod"] = pod
            if "sucr" in metrics_to_calc: scores["sucr"] = sucr
            if "csi" in metrics_to_calc: scores["csi"] = csi
            if "bias" in metrics_to_calc: scores["bias"] = bias
        if "fss" in metrics_to_calc and mse is not None and ref is not None:
            fss = torch.clamp(1.0 - (mse / (ref + self.eps)), 0.0, 1.0)
            scores["fss"] = fss
        return scores

    def compute(self) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Computes the final metrics and returns them in a nested dictionary.
        - Top-level keys: Thresholds as strings (e.g., '2.5') and an 'avg' key.
        - Second-level keys: Metric names (e.g., 'csi', 'persistence_fss').
        """
        results = {str(T): {} for T in self.threshold_list}
        res_func = (lambda x: x.detach().cpu().numpy()) if self.mode == "1" else (lambda x: x.item())

        model_scores = self._calculate_scores(
            H=getattr(self, "hits", None),
            M=getattr(self, "misses", None),
            F=getattr(self, "fas", None),
            mse=getattr(self, "fss_mse", None),
            ref=getattr(self, "fss_ref", None),
            metrics_to_calc=self.model_metrics
        )
        for name, values in model_scores.items():
            for i, T in enumerate(self.threshold_list):
                results[str(T)][name] = res_func(values[i])

        pers_scores = self._calculate_scores(
            H=getattr(self, "hits_pers", None),
            M=getattr(self, "misses_pers", None),
            F=getattr(self, "fas_pers", None),
            mse=getattr(self, "fss_mse_pers", None),
            ref=getattr(self, "fss_ref_pers", None),
            metrics_to_calc=self.persistence_metrics
        )
        for name, values in pers_scores.items():
            for i, T in enumerate(self.threshold_list):
                results[str(T)][f"persistence_{name}"] = res_func(values[i])
        
        # Calculate the average over thresholds (and time if applicable)
        avg_results = {}
        all_scores = {**model_scores, **{f"persistence_{k}": v for k, v in pers_scores.items()}}
        for name, values in all_scores.items():
            avg_results[name] = torch.mean(values).item()
        if avg_results:
            results["avg"] = avg_results
            
        return results