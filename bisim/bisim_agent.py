import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os

from bisim.encoder import PixelEncoder
from bisim.transistion_model import TransistionModel

class BisimAgent(object):
    def __init__(self, obs_shape, n_actions, discount=0.99, device='cuda', bisim_coef=0.5,
                encoder_lr=1e-3, encoder_weight_decay=0., decoder_lr=1e-3, decoder_weight_decay=0., 
                encoder_feature_dim=256, encoder_n_layers=2, 
                t_model_layer_width=5, 
                decoder_layer_size=512):
        self.encoder = PixelEncoder(obs_shape, feature_dim=encoder_feature_dim, num_layers=encoder_n_layers).to(device)
        self.transition_model = TransistionModel(n_actions=n_actions, encoder_feature_dim=encoder_feature_dim, layer_width=t_model_layer_width).to(device)
        self.reward_decoder = nn.Sequential(
            nn.Linear(encoder_feature_dim, decoder_layer_size),
            nn.LayerNorm(decoder_layer_size),
            nn.ReLU(),
            nn.Linear(decoder_layer_size, 1)).to(device)

        self.discount = discount
        self.bisim_coef = bisim_coef
        self.n_actions = n_actions

        self.device = device

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr, weight_decay=encoder_weight_decay)
        self.reward_decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()), 
            lr=decoder_lr, weight_decay=decoder_weight_decay)

    def __call__(self, state):
        state = state.to(self.device)
        return self.encoder(state)

    def calTransistionModelLoss(self, state, action, next_state, reward):

        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)

        h = self.encoder(state)
        
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h = self.encoder(next_state)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))

        pred_next_latent = self.transition_model.sample_prediction(torch.cat([h, action], dim=1))
        pred_next_reward = self.reward_decoder(pred_next_latent)
        reward_loss = F.mse_loss(pred_next_reward, reward)
        total_loss = loss + reward_loss
        return total_loss

    def calEncoderLoss(self, state, action, reward):
        '''
        '''
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)

        h = self.encoder(state)            

        # Sample random states across episodes at random
        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)
        h2 = h[perm]

        with torch.no_grad():
            # action, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([h, action], dim=1))
            # reward = self.reward_decoder(pred_next_latent_mu1)
            reward2 = reward[perm]
        if pred_next_latent_sigma1 is None:
            pred_next_latent_sigma1 = torch.zeros_like(pred_next_latent_mu1)

        if pred_next_latent_mu1.ndim == 2:  # shape (B, Z), no ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[perm]

        elif pred_next_latent_mu1.ndim == 3:  # shape (B, E, Z), using an ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[:, perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[:, perm]
        else:
            raise NotImplementedError
        
        z_dist = F.smooth_l1_loss(h, h2, reduction='none')
        r_dist = F.smooth_l1_loss(reward, reward2, reduction='none').unsqueeze(1)

        transition_dist = torch.sqrt(
            (pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) +
            (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2)
        )

        # print("R_DIST: ", r_dist.size())
        # print("TR_DIST: ", transition_dist.size())
        bisimilarity = r_dist + self.discount * transition_dist
        loss = (z_dist - bisimilarity).pow(2).mean()

        return loss

    def update(self, state, action, reward, next_state):

        # TODO: here, action may be converted to onehot (done).
        action_onehot = torch.zeros([action.size(0), self.n_actions]).to(self.device).scatter_(1, action, 1.)

        transistionModelLoss = self.calTransistionModelLoss(state, action_onehot, next_state, reward)
        encoderLoss = self.calEncoderLoss(state, action_onehot, reward)
        totalLoss = self.bisim_coef * encoderLoss + transistionModelLoss

        self.encoder_optimizer.zero_grad()
        self.reward_decoder_optimizer.zero_grad()

        totalLoss.backward()

        self.encoder_optimizer.step()
        self.reward_decoder_optimizer.step()
    
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.encoder, os.path.join(save_dir, "encoder"))
        torch.save(self.transition_model, os.path.join(save_dir, "transistion_model"))
        torch.save(self.reward_decoder, os.path.join(save_dir, "decoder"))
        print("@Info: bisim model saved")

    def load(self, save_dir):
        encoder_exists = os.path.exists(os.path.join(save_dir, "encoder"))
        transistion_model_exists = os.path.exists(os.path.join(save_dir, "transistion_model"))
        decoder_exists = os.path.exists(os.path.join(save_dir, "decoder"))

        if encoder_exists and transistion_model_exists and decoder_exists:
            print("@Info: bisim load path check passed")
            self.encoder = torch.load(os.path.join(save_dir, "encoder"))
            self.transition_model = torch.load(os.path.join(save_dir, "transistion_model"))
            self.reward_decoder = torch.load(os.path.join(save_dir, "decoder"))
            print("@Info: bisim model loaded")
        
        else:
            raise FileNotFoundError("bisim model not found at {}".format(save_dir))