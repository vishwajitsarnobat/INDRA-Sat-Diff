# climate_forecast/diffusion/knowledge_alignment/alignment_guides.py

from typing import Dict, Any
import torch

from .alignment_pl import get_sample_align_fn
from .models import NoisyCuboidTransformerEncoder

class AverageIntensityAlignment(): # RENAMED
    """
    A knowledge alignment guide for steering diffusion models towards a target
    average intensity.
    """
    def __init__(self,
                 alignment_type: str = "avg_x",
                 guide_scale: float = 1.0,
                 model_type: str = "cuboid",
                 model_args: Dict[str, Any] = None,
                 model_ckpt_path: str = None,
                 ):
        super().__init__()
        # The internal logic of this method is fine.
        assert alignment_type == "avg_x", f"Only 'avg_x' alignment_type is currently supported."
        self.alignment_type = alignment_type
        self.guide_scale = guide_scale
        if model_args is None:
            model_args = {}
        if model_type == "cuboid":
            self.model = NoisyCuboidTransformerEncoder(**model_args)
        else:
            raise NotImplementedError(f"model_type={model_type} is not implemented")
        if model_ckpt_path is not None:
            self.model.load_state_dict(torch.load(model_ckpt_path, map_location="cpu"))

    @staticmethod
    def model_objective(model, x_t, t, **kwargs) -> torch.Tensor:
        """
        The objective function that the alignment model learns.
        It should predict the average intensity of the final, clean data (x_0),
        given the noisy latent data (x_t) and the timestep (t).
        """
        # This calls the forward pass of the actual torch module (NoisyCuboidTransformerEncoder)
        return model(x_t, t)

    @staticmethod
    def calculate_ground_truth_target(x_0: torch.Tensor) -> torch.Tensor:
        """
        Calculates the ground truth target value from the clean data x_0.
        
        This function is now separate, more explicit, and robust.
        It calculates the mean over all dimensions except the batch dimension.

        Parameters
        ----------
        x_0:  torch.Tensor
            The clean, ground truth data sequence (e.g., shape N, T, H, W, C).

        Returns
        -------
        avg: torch.Tensor
            The average intensity for each item in the batch (shape N,).
        """
        batch_size = x_0.shape[0]
        # Reshape to (batch_size, -1) and calculate the mean over all other dimensions
        return torch.mean(x_0.view(batch_size, -1), dim=1)

    def alignment_fn(self, zt, t, y=None, zc=None, **kwargs):
        """
        Defines the loss function for the guidance gradient.
        It's the L2 norm between the model's prediction and the target value.
        """
        # Predict the average intensity from the noisy latent zt
        predicted_target = self.model(zt, t, zc=zc, y=y, **kwargs)
        
        # The ground truth target is passed in via kwargs during sampling
        ground_truth_target = kwargs.get("avg_x_gt")
        if ground_truth_target is None:
            raise ValueError("'avg_x_gt' must be provided in kwargs for alignment guidance.")

        # Squeeze to ensure shapes are compatible for loss calculation
        predicted_target = predicted_target.squeeze()
        ground_truth_target = ground_truth_target.squeeze()
        
        ret = torch.linalg.vector_norm(predicted_target - ground_truth_target, ord=2)
        return ret

    def get_mean_shift(self, zt, t, y=None, zc=None, **kwargs):
        """

        Calculates the gradient of the alignment loss w.r.t. the noisy latent zt.
        This gradient is used to "shift" the diffusion sampling process.
        """
        grad_fn = get_sample_align_fn(self.alignment_fn)
        grad = grad_fn(self, zt, t, y=y, zc=zc, **kwargs) # Pass self to the grad_fn
        return self.guide_scale * grad