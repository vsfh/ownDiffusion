# Copyright 2022 NVIDIA and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
import sys
sys.path.append('..')
sys.path.append('.')

class KarrasVeScheduler():
    """
    Stochastic sampling from Karras et al. [1] tailored to the Variance-Expanding (VE) models [2]. Use Algorithm 2 and
    the VE column of Table 1 from [1] for reference.

    [1] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models."
    https://arxiv.org/abs/2206.00364 [2] Song, Yang, et al. "Score-based generative modeling through stochastic
    differential equations." https://arxiv.org/abs/2011.13456

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functions.

    For more details on the parameters, see the original paper's Appendix E.: "Elucidating the Design Space of
    Diffusion-Based Generative Models." https://arxiv.org/abs/2206.00364. The grid search values used to find the
    optimal {s_noise, s_churn, s_min, s_max} for a specific model are described in Table 5 of the paper.

    Args:
        sigma_min (`float`): minimum noise magnitude
        sigma_max (`float`): maximum noise magnitude
        s_noise (`float`): the amount of additional noise to counteract loss of detail during sampling.
            A reasonable range is [1.000, 1.011].
        s_churn (`float`): the parameter controlling the overall amount of stochasticity.
            A reasonable range is [0, 100].
        s_min (`float`): the start value of the sigma range where we add noise (enable stochasticity).
            A reasonable range is [0, 10].
        s_max (`float`): the end value of the sigma range where we add noise.
            A reasonable range is [0.2, 80].

    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        s_noise: float = 1.003,
        s_churn: float = 80,
        s_min: float = 0.05,
        s_max: float = 50,
    ):
        self.__dict__ = autoargs(self, locals())
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = sigma_max

        # setable values
        self.num_inference_steps: int = None
        self.timesteps: np.IntTensor = None
        self.schedule: torch.FloatTensor = None  # sigma(t_i)
        

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the continuous timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.

        """
        self.num_inference_steps = num_inference_steps
        timesteps = np.arange(0, self.num_inference_steps).copy()
        self.timesteps = torch.from_numpy(timesteps).cuda()
        rho = 7
        schedule = [
            (
                self.sigma_max**(1/rho) +
                 (self.sigma_min**(1/rho) - self.sigma_max**(1/rho)) * (i / (num_inference_steps - 1))
            )**(rho)
            for i in self.timesteps
        ]
        self.schedule = torch.tensor(schedule, dtype=torch.float32).cuda()

    def prepare(self, original_input, noise, timestep):

        sigma = (torch.randn((original_input.shape[0],), device=original_input.device, dtype=original_input.dtype) * 1.2 -1.2).exp()
        sigma_data = 0.5
        c_skip = sigma_data**2/(sigma**2+sigma_data**2)
        c_out = sigma*sigma_data/(sigma**2+sigma_data**2)**(1/2)
        c_in = (sigma**2+sigma_data**2)**(-1/2)
        c_noise = sigma
        # print(sigma, noise.shape, original_input.shape)
        while len(sigma.shape) < len(noise.shape):
            sigma = sigma.unsqueeze(-1)
            c_noise = c_noise.unsqueeze(-1)
            c_in = c_in.unsqueeze(-1)
            c_skip = c_skip.unsqueeze(-1)
            c_out = c_out.unsqueeze(-1)
        
            
        input = c_in * (original_input + sigma * noise)
        input_sigma = c_noise.squeeze()
        label = (original_input- c_skip*(original_input + sigma * noise))/c_out
        eff = c_out**2 * (sigma**2+sigma_data**2)/(sigma*sigma_data)**2
        # print(eff)
        return input, input_sigma, label, eff
        
    def add_noise_to_input(
        self, sample: torch.FloatTensor, sigma: float, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.FloatTensor, float]:
        """
        Explicit Langevin-like "churn" step of adding noise to the sample according to a factor gamma_i ≥ 0 to reach a
        higher noise level sigma_hat = sigma_i + gamma_i*sigma_i.

        TODO Args:
        """
        if self.s_min <= sigma <= self.s_max:
            gamma = min(self.s_churn / self.num_inference_steps, 2**0.5 - 1)
        else:
            gamma = 0

        # sample eps ~ N(0, S_noise^2 * I)
        eps = self.s_noise * torch.randn(sample.shape, generator=generator).to(sample.device)
        sigma_hat = sigma + gamma * sigma
        sample_hat = sample + ((sigma_hat**2 - sigma**2) ** 0.5 * eps)

        return sample_hat, sigma_hat

    def step(
        self,
        model_output: torch.FloatTensor,
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: torch.FloatTensor,
        return_dict: bool = True,
    ):
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.FloatTensor`): TODO
            return_dict (`bool`): option for returning tuple rather than KarrasVeOutput class

            KarrasVeOutput: updated sample in the diffusion chain and derivative (TODO double check).
        Returns:
            [`~schedulers.scheduling_karras_ve.KarrasVeOutput`] or `tuple`:
            [`~schedulers.scheduling_karras_ve.KarrasVeOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """

        pred_original_sample = sample_hat + sigma_hat * model_output
        derivative = (sample_hat - pred_original_sample) / sigma_hat
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * derivative

        if not return_dict:
            return (sample_prev, derivative)

        return {'prev_sample':sample_prev, 'derivative':derivative, 'pred_original_sample':pred_original_sample}
            
        

    def step_correct(
        self,
        model_output: torch.FloatTensor,
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: torch.FloatTensor,
        sample_prev: torch.FloatTensor,
        derivative: torch.FloatTensor,
        return_dict: bool = True,
    ):
        """
        Correct the predicted sample based on the output model_output of the network. TODO complete description

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.FloatTensor`): TODO
            sample_prev (`torch.FloatTensor`): TODO
            derivative (`torch.FloatTensor`): TODO
            return_dict (`bool`): option for returning tuple rather than KarrasVeOutput class

        Returns:
            prev_sample (TODO): updated sample in the diffusion chain. derivative (TODO): TODO

        """
        pred_original_sample = sample_prev + sigma_prev * model_output
        derivative_corr = (sample_prev - pred_original_sample) / sigma_prev
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * (0.5 * derivative + 0.5 * derivative_corr)

        if not return_dict:
            return (sample_prev, derivative)

        return {
            'prev_sample':sample_prev, 'derivative':derivative, 'pred_original_sample':pred_original_sample
        }

    def add_noise(self, original_samples, noise, timesteps):
        raise NotImplementedError()
