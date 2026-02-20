"""
Gradient-weighted Class Activation Mapping (Grad-CAM) for mangrove models.

Provides interpretability by highlighting which spatial regions in
a 244×244 AlphaEarth embedding tile most influence the model's
coverage classification decision.

Supports both CNN-based and ViT-based architectures.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class GradCAM:
    """
    Grad-CAM for CNN or ViT classifiers.

    Usage (CNN):
        model = CNNClassifier(...)
        gradcam = GradCAM(model, target_layer=model.features[-3])
        heatmap = gradcam(input_tensor, target_class=2)

    Usage (ViT):
        model = ViTClassifier(...)
        gradcam = GradCAM(model, target_layer=model.blocks[-1].norm1)
        heatmap = gradcam(input_tensor, target_class=2)
    """
    def __init__(self, model: torch.nn.Module,
                 target_layer: torch.nn.Module):
        """
        Args:
            model: trained classification model.
            target_layer: the layer whose activations will be used
                          to compute the Grad-CAM heatmap.
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    @torch.enable_grad()
    def __call__(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None,
                 img_size: Tuple[int, int] = (244, 244)) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Args:
            input_tensor: (1, C, H, W) embedding tensor (single sample).
            target_class: class index to explain. If None, uses predicted class.
            img_size: (H, W) to resize the heatmap to.

        Returns:
            heatmap: (H, W) numpy array in [0, 1], same spatial size as input.
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Zero gradients and backward on target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Pool gradients over spatial / token dimensions
        gradients = self._gradients    # (1, N, E) or (1, C, H', W')
        activations = self._activations

        if gradients.dim() == 3:
            # ViT: (1, N_tokens, embed_dim) → average over token dim
            weights = gradients.mean(dim=1, keepdim=True)    # (1, 1, E)
            cam = (weights * activations).sum(dim=-1)        # (1, N)
            cam = cam[:, 1:]  # drop [CLS] token
            # Reshape to spatial grid
            n_patches = cam.shape[1]
            h = w = int(n_patches ** 0.5)
            cam = cam.reshape(1, 1, h, w)
        else:
            # CNN: (1, C, H', W')
            weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
            cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')

        # ReLU + normalize
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=img_size, mode='bilinear',
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        """Remove forward/backward hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass
