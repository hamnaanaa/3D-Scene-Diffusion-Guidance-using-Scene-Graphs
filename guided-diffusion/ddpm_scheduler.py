from collections import namedtuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce

from tqdm.auto import tqdm

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class DDPMScheduler(nn.Module):
    def __init__(
        self,
        model,
        *,
        N,
        D,
        range_matrix,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 1.,
        min_snr_loss_weight = False,
        min_snr_gamma = 5
    ):
        super(DDPMScheduler, self).__init__()
        #assert not (type(self) == DDPM) #and model.channels != model.out_dim)
        #assert not model.random_or_learned_sinusoidal_cond
        self.range_matrix = range_matrix

        self.model = model
        #self.channels = self.model.channels

        self.N = N
        self.D = D
        self.range_matrix = range_matrix

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'
        
        # Defines vector of betas based on noise scheduler
        # --> one beta for each timestep
        
        if beta_schedule == 'linear':
            betas = DDPMUtils.linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = DDPMUtils.cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # alpha_hat_t for every timestep
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.) # alpha_hat_(t-1): for the PREVIOUS timestep

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = DDPMUtils.default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod)) # sqrt(alpha_hat)
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod)) # sqrt(1-alpha_hat)
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod)) # log(1-alpha_hat)
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod)) # 1/sqrt(alpha_hat)
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1)) # sqrt(1/alpha_hat -1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # beta_t_schlange

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20))) # log(beta_t)
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)) # first coefficient to calculate mü_t_schlange
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)) # second coefficient to calculate mü_t_schlange

        # loss weight ???????????

        snr = alphas_cumprod / (1 - alphas_cumprod) # signal-to-noise ratio of q(x_t | x_0)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    def predict_start_from_noise(self, x_t, t, noise): # x_0 = 1/sqrt(alpha_hat)*x_t - sqrt((1-alpha_hat)/alpha_hat)eps
        return (
            DDPMUtils.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            DDPMUtils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0): # eps = (1/sqrt(alpha_hat)*x_t-x_0)/sqrt(1/alpha_hat-1)
        return (
            (DDPMUtils.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            DDPMUtils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            DDPMUtils.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            DDPMUtils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            DDPMUtils.extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            DDPMUtils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t): # q(x_t-1 | x_t, x_0)
        posterior_mean = (
            DDPMUtils.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            DDPMUtils.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = DDPMUtils.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = DDPMUtils.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, obj_cond, edge_cond, relation_cond, cond_scale = 3., clip_x_start = False):
        '''
        Given x_t [BxNxD], t [BxNxD] and graph_cond (obj_cond [BxNxC], edge_cond [Bx2xE], relation_cond [BxE])
        --> forward pass through NN
        --> outputs eps_theta(x_t, t) and x0
        '''
        model_output = self.model.forward_with_cond_scale(x, t, obj_cond, edge_cond, relation_cond, cond_scale = cond_scale) # forward block im RGCN
        
        #maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        # dynamic thresholding instead
        

        if self.objective == 'pred_noise': # I assume we use this one
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            #x_start = maybe_clip(x_start)
            x_start = DDPMUtils.dynamic_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            #x_start = maybe_clip(x_start)
            x_start = DDPMUtils.dynamic_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            #x_start = maybe_clip(x_start)
            x_start = DDPMUtils.dynamic_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start) # shaping output as: ('ModelPrediction', ['pred_noise', 'pred_x_start'])

    def p_mean_variance(self, x, t, obj_cond, edge_cond, relation_cond, cond_scale, clip_denoised = True):
        '''
        Given x_t, t and graph_cond (obj_cond [BxNxC], edge_cond [Bx2xE], relation_cond [BxE])
        --> gets x_0 from 'model_predictions'
        --> sends x_0, x_t, t to 'y_posterior' to recover mean and variance of p(x_t-1 | x_0, x_t, t)
        --> outputs mü, var, log_var, x_0
        '''
        preds = self.model_predictions(x, t, obj_cond, edge_cond, relation_cond, cond_scale) # preds = ('ModelPrediction', ['pred_noise', 'pred_x_start'])
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, obj_cond, edge_cond, relation_cond, cond_scale = 3., clip_denoised = True):
        '''
        Given x_t, t and graph_cond (obj_cond [BxNxC], edge_cond [Bx2xE], relation_cond [BxE])
        --> gets mü, log_var, x_0 from 'p_mean_variance'
        --> predicts x_t-1
        --> outputs x_t-1, x_0
        '''
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long) # vector of length dim0(x) filled with t
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, obj_cond = obj_cond, edge_cond = edge_cond, relation_cond = relation_cond, cond_scale = cond_scale, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, obj_cond, edge_cond, relation_cond, shape, cond_scale = 3., return_all_samples = False):
        '''
        given an input shape: BxNxD (where B = batch size) and a graph condition (obj_cond [BxNxC], edge_cond [2xnum(edges)], relation_cond [num(edges)])
        --> creates Gaussian noise x_T of shape BxNxD
        --> for every timestep t from T to 0:
            sends x_t, t, graph_cond to 'p_sample' to get x_t-1, x_0
            sets x_t = x_t-1
        --> outputs x_0 OR all samples [(T, x_T), (T-1, x_T-1), ..., (0, x_0)] if return_all_samples == True
        '''
        batch, device = shape[0], self.betas.device

        data = torch.randn(shape, device=device)

        x_start = None
        
        if return_all_samples:
            all_samples = [(self.num_timesteps, DDPMUtils.unnormalize_to_original(data, self.range_matrix))]

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            data, x_start = self.p_sample(data, t, obj_cond, edge_cond, relation_cond, cond_scale)
            if return_all_samples:
                all_samples.append((t, DDPMUtils.unnormalize_to_original(data, self.range_matrix)))

        data = DDPMUtils.unnormalize_to_original(data, self.range_matrix) 
        return data if not return_all_samples else all_samples

    @torch.no_grad()
    def ddim_sample(self, obj_cond, edge_cond, relation_cond, shape, cond_scale = 3., clip_denoised = True, return_all_samples = False):
        '''
        given an input shape: BxNxD (where B = batch size)
        --> creates Gaussian noise x_T of shape BxNxD
        --> create time pairs [(T-1, T-2), (T-2, T-3),..., (1, 0), (0, -1)]
        --> for every time pair (t1, t2)
            - create a matrix time_cond of size x filled with t1
            - send x_t, time_cond to 'model_predictions' to compute eps_theta(x_t, t) and x_0
            - get a_hat(t1) and a_hat(t2)
            - compute x_t-1 based on the DDIM sampler
            - set x_t = x_t-1
        --> output x_0 
            - WARNING: TODO: implement logic for return_all_samples <- currently not used
        '''
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        data = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(data, time_cond, obj_cond, edge_cond, relation_cond, cond_scale = cond_scale, clip_x_start = clip_denoised)

            if time_next < 0:
                data = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(data)

            data = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        data = DDPMUtils.unnormalize_to_original(data, self.range_matrix) 
        return data

    @torch.no_grad()
    def sample(self, obj_cond, edge_cond, relation_cond, cond_scale = 3., return_all_samples = False):
        '''
        - gets barch size from graph_cond (obj_cond [BxNxC], edge_cond [Bx2xE], relation_cond [BxE])
        - sends the required shape BxNxD to one of the sample functions
        - outputs x_0 [BxNxD] OR all samples [(T, x_T), (T-1, x_T-1), ..., (0, x_0)] if return_all_samples == True
        '''
        batch_size, N, D = obj_cond.shape[0] // self.N, self.N, self.D
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(obj_cond, edge_cond, relation_cond, (batch_size, N, D), cond_scale, return_all_samples = return_all_samples)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        '''
        given x1_0, x2_0
        --> compute x1_t and x2_t
        --> interpolate data_t between x1_t and x2_t
        --> compute data_0 using 'p_sample'
        --> output data_0
        '''
        b, *_, device = *x1.shape, x1.device
        t = DDPMUtils.default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        data = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            data = self.p_sample(data, torch.full((b,), i, device=device, dtype=torch.long))

        return data

    def q_sample(self, x_start, t, noise=None):
        '''
        Given x_0, t
        --> sample x_t
        '''
        noise = DDPMUtils.default(noise, lambda: torch.randn_like(x_start))

        return (
            # x_start + noise
            DDPMUtils.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            DDPMUtils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        '''
        return the right loss function
        '''
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, obj_cond, edge_cond, relation_cond, noise = None):
        '''
        Given x_0, t
        --> sample noise
        --> sample x_t based on noise using 'q_sample'
        --> predicting the noise eps_theta(t) using the RGCN
        --> calculate loss by comparing the true noise and eps_theta(t)
        --> output mean of loss over batch
        
        '''
        #b, c, h, w = x_start.shape
        B, N, D = x_start.shape
        noise = DDPMUtils.default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step

        model_out = self.model.forward(x, t, obj_cond, edge_cond, relation_cond) # self.model

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * DDPMUtils.extract(self.loss_weight, t, loss.shape)

        return loss.mean()
    
    # tbd image_size
    def forward(self, data, obj_cond, edge_cond, relation_cond, noise=None, *args, **kwargs):
        '''
        Given Data of shape [BxNxD]
        --> create a vector t with B random numbers between 0 and num_timesteps
        --> returns the loss of the current NN on the batch using 'p_losses'
        '''
        #b, c, h, w, device, data_size, = *data.shape, data.device, self.data_size
        #assert h == data_size and w == data_size, f'height and width of image must be {data_size}'
        B, N, D, device = *data.shape, data.device
        assert N == self.N and D == self.D
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long() 
        
        data = DDPMUtils.normalize_to_neg_one_to_one(data, self.range_matrix) 

        return self.p_losses(data, t, obj_cond, edge_cond, relation_cond, noise=noise, *args, **kwargs)


class DDPMUtils:
    """
    Utility functions for DDPM
    """
    @staticmethod
    # TODO: what should be the default value of p?
    def dynamic_clip(A, p=0.8): 
        s = torch.quantile(torch.flatten(torch.absolute(A), start_dim=1), p, dim=1)
        s = torch.clamp(s, max=1)
        expanded_s = s.view(A.size(dim=0), 1, 1).expand(A.shape)
        A_out = A.clamp(min=-expanded_s, max=expanded_s)/expanded_s
        return A_out

    @staticmethod
    def default(val, d):
        if val is not None:
            return val
        return d() if callable(d) else d

    # --- Time scheduling functions
    
    @staticmethod
    def extract(a, t, x_shape):
        """
        extract values from a at indices t
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    @staticmethod
    def linear_beta_schedule(timesteps):
        """
        linear schedule
        """
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
    
    @staticmethod
    def cosine_beta_schedule(timesteps, s = 0.008):
        """
        cosine schedule 
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    @staticmethod
    def normalize_to_neg_one_to_one(data,range_matrix):
        '''
        Args:
            data: (B, 20, 315)
            range: (2, 315) --> max, min

        Returns:
            data: (B, 20, 315)
        ''' 
        # Shift the data so that the min value is at 0
        shifted_data = data - range_matrix[1]

        # Normalize the data [0, 1]
        normalized_data = shifted_data / (range_matrix[0]-range_matrix[1])

        # Modify to [-1, 1]
        modified_data = 2*normalized_data - 1

        return modified_data
    
    @staticmethod
    def unnormalize_to_original(data, range_matrix):
        '''
        Args:
            data: (B, 20, 315)
            range: (2, 315) --> max, min

        Returns:
            data: (B, 20, 315)
        '''
        # Shift the data so that the min value is at 0
        shifted_data = data + 1

        # Normalize the data [0, 1]
        normalized_data = shifted_data / 2

        # Scale data to original range
        scaled_data = normalized_data * (range_matrix[0]-range_matrix[1])

        # Shift data to original position
        modified_data = scaled_data + range_matrix[1]

        return modified_data