# curriculum_ssl_selftaught.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
import random
import os
import networks
from layers import *

class SILogLoss(torch.nn.Module):
    """Same loss used in your fine-tuning script."""
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = F.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]

        g = torch.log(input) - torch.log(target)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)

class CurriculumLearnerSupervised:
    def __init__(self, opt, model, dataloader, dataset, model_path ,pacing_function="linear", device="cuda"):
        """
        model: SPIdepth model (without loading weights)
        dataloader: full training dataloader
        pacing_function: linear or quadratic pacing function
        model_path: path to the checkpoint to use for scoring
        opt: options file to load the model
        """
        self.models = {} if model=='SPIdepth' else ''
        self.dataloader = dataloader
        self.dataset= dataset
        self.device = device
        self.pacing_function = pacing_function
        self.sample_scores = []
        self.sorted_indices = []
        self.opt= opt
        self.batch_size=1
        
        
        if model=='SPIdepth' and not os.path.exists("/home/jturriatellallire/scores_mid_sl_transfer.npy"):
            self.models["encoder"] = networks.Unet(pretrained=False, backbone="convnextv2_huge.fcmae_ft_in22k_in1k_384", in_channels=3, num_classes=32, decoder_channels=(1024,512,256,128))
            self.models["depth"] = networks.Depth_Decoder_QueryTr(in_channels=32, patch_size=32, dim_out=64, embedding_dim=32, 
                                                                    query_nums=64, num_heads=4, min_val=0.001, max_val=80.0)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Checkpoint not found at {model_path}")
            
            encoder_path = os.path.join(model_path, "encoder.pth")
            decoder_path = os.path.join(model_path, "depth.pth")


            loaded_dict_enc = torch.load(encoder_path, map_location=self.device)
            loaded_dict_enc = self.remove_module_prefix(loaded_dict_enc)
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models["encoder"].state_dict()}
            self.models["encoder"].load_state_dict(filtered_dict_enc)

            loaded_dict_enc = torch.load(decoder_path, map_location=self.device)
            loaded_dict_enc = self.remove_module_prefix(loaded_dict_enc)
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models["depth"].state_dict()}
            self.models["depth"].load_state_dict(filtered_dict_enc)


            for model in self.models.values():
                model.to(device)

            for m in self.models.values():
                m.eval()

        
    def pacing(self, step, total_steps, total_samples):
        """
        Define how many samples to use at this stage of training.
        Supports linear, quadratic, exponential, logarithmic, and step pacing functions.
        """
        Nb = int(total_samples * self.opt.b)  # fraction of full training data
        aT = self.opt.a * total_steps  # parameter a times total epochs
        t =step + 1  # current step (1-based index)
        pacing_result=0

        if self.pacing_function == "linear":
            pacing_result= int(Nb + ((1 - self.opt.b) * total_samples / aT) * t)
        elif self.pacing_function == "quadratic":
            pacing_result= int(Nb + (total_samples * (1 - self.opt.b) / ((aT)**(self.opt.p))) * (t ** self.opt.p))
        elif self.pacing_function == "exponential":
            pacing_result= int(Nb + (total_samples * (1 - self.opt.b) / (np.exp(10) - 1)) * (np.exp(10 * t / aT) - 1))
        elif self.pacing_function == "logarithmic":
            pacing_result= int(Nb + total_samples * (1 - self.opt.b) * (1 + (1 / 10) * np.log(t / aT + np.exp(-10))))
        elif self.pacing_function == "step":
            pacing_result= int(Nb + total_samples * (0 if (t / aT)< 1 else 1))
        else:
            raise NotImplementedError(f"Pacing function '{self.pacing_function}' not implemented")
        
        return min(pacing_result, total_samples)

    def get_curriculum_batches(self, step, total_steps, batch_size, score_path="sample_scores.npy"):
        """
        Return a DataLoader for the current step based on the pacing function and stored scores
        """
        if len(self.sample_scores) == 0:
            self.load_scores(score_path)
        
        # Determine the number of samples to use at this stage of training
        selected_size = self.pacing(step, total_steps, len(self.sample_scores))
        selected_indices = self.sorted_indices[:selected_size]

        # Create a subset of the dataset based on selected indices
        selected_subset = torch.utils.data.Subset(self.dataloader.dataset, selected_indices)
        # Create a DataLoader for the selected subset
        selected_loader = torch.utils.data.DataLoader(
            selected_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
        )

        return selected_loader
    
    def score_and_save_losses(self, score_path="scores_midair_sl.npy"):
        """
        Computes and stores difficulty scores (losses) of the dataset.
        Only needs to be run once before curriculum training begins.
        """
        sample_losses = []
        criterion_ueff = SILogLoss().to(self.device)
        new_dataloader= torch.utils.data.DataLoader(
            self.dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
        )

        self.models["encoder"].eval()
        self.models["depth"].eval()

        print("Computing supervised difficulty scores...")
        with torch.no_grad():
            for inputs in new_dataloader:
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)

                outputs, loss_val = self.process_batch_supervised(inputs, criterion_ueff)
                sample_losses.append(loss_val)

        sample_losses = np.array(sample_losses)
        np.save(score_path, sample_losses)
        print(f"Saved supervised difficulty scores to: {score_path}")

        for model in self.models.values():
            del model
        torch.cuda.empty_cache()


    def load_scores(self, score_path="sample_scores.npy"):
        if not os.path.exists(score_path):
            raise FileNotFoundError(f"Score file {score_path} not found. Run score_and_save_losses() first.")
        self.sample_scores = np.load(score_path)
        self.sorted_indices = np.argsort(self.sample_scores)  # easiest samples first


    def process_batch_supervised(self, inputs, criterion_ueff):
        """
        Processes a batch using the SAME logic as your supervised fine-tuning,
        while ensuring the input resolution matches the pretrained SPIdepth model.
        """

        # Raw MidAir tensors: [B,3,384,384]
        img = inputs["image"].to(self.device)
        gt_depth = inputs["depth"].to(self.device)

        # === 1. Resize image to pretrained resolution ===
        pretrained_h, pretrained_w = 320, 1024
        if img.shape[-2:] != (pretrained_h, pretrained_w):
            img_resized = F.interpolate(img, size=(pretrained_h, pretrained_w),
                                        mode='bilinear', align_corners=False)
        else:
            img_resized = img

        # === 2. Forward pass ===
        features = self.models["encoder"](img_resized)
        pred_dict = self.models["depth"](features)

        # === Extract depth tensor (decoder returns a dict) ===
        if "depth" in pred_dict:
            pred_depth = pred_dict["depth"]
        elif "disp" in pred_dict:
            pred_depth = 1.0 / (pred_dict["disp"] + 1e-6)
        else:
            raise ValueError("Decoder output does not contain 'depth' or 'disp'")

        # === 3. Resize prediction back to GT resolution (384×384) ===
        if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
            pred_depth = F.interpolate(pred_depth, size=gt_depth.shape[-2:],
                                    mode='bilinear', align_corners=True)

        # === 4. Scale recovery exactly like your training loop ===
        pred_np = pred_depth[0].squeeze().detach().cpu().numpy()
        depth_np = gt_depth[0].squeeze().detach().cpu().numpy()

        valid_mask = np.logical_and(
            depth_np > self.opt.min_depth_eval,
            depth_np < self.opt.max_depth_eval
        )

        if np.isnan(np.median(depth_np)) or np.isnan(np.median(pred_np)):
            ratio = 1
        else:
            ratio = np.median(depth_np[valid_mask]) / np.median(pred_np[valid_mask])

        pred_depth *= ratio

        # === 5. Compute supervised SILog loss ===
        mask = gt_depth > self.opt.min_depth
        l_dense = criterion_ueff(pred_depth, gt_depth, mask=mask.bool(), interpolate=False)

        return pred_depth, l_dense.item()

    def remove_module_prefix(self, state_dict):
        return {k.replace("module.", ""): v for k, v in state_dict.items()}