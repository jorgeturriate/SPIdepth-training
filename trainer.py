# pyright: reportGeneralTypeIssues=warning
from __future__ import absolute_import, division, print_function

import numpy as np
import time
import uuid
from datetime import datetime as dt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
import wandb #To log in wandb
# from datetime import datetime as dt
# import uuid
from collections import OrderedDict



PROJECT = "SPIdepth_Midair_full"
experiment_name="resnet18lite"

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        # assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        # assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales) # default=[0], we only perform single scale training
        self.num_input_frames = len(self.opt.frame_ids) # default=[0, -1, 1]
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames # default=2 

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0]) # default=True

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # self.models["encoder"] = networks.BaseEncoder.build(num_features=self.opt.num_features, model_dim=self.opt.model_dim)
        # self.models["encoder"] = networks.ResnetEncoderDecoder(num_layers=self.opt.num_layers, num_features=self.opt.num_features, model_dim=self.opt.model_dim)
        if self.opt.backbone in ["resnet", "resnet_lite"]:
            self.models["encoder"] = networks.ResnetEncoderDecoder(num_layers=self.opt.num_layers, num_features=self.opt.num_features, model_dim=self.opt.model_dim)
        elif self.opt.backbone == "resnet18_lite":
            self.models["encoder"] = networks.LiteResnetEncoderDecoder(model_dim=self.opt.model_dim)
        elif self.opt.backbone == "eff_b5":
            self.models["encoder"] = networks.BaseEncoder.build(num_features=self.opt.num_features, model_dim=self.opt.model_dim)
        else: 
            self.models["encoder"] = networks.Unet(pretrained=(not self.opt.load_pretrained_model), backbone=self.opt.backbone, in_channels=3, num_classes=self.opt.model_dim, decoder_channels=self.opt.dec_channels)

        if self.opt.load_pretrained_model:
            print("-> Loading pretrained encoder from ", self.opt.load_pt_folder)
            encoder_path = os.path.join(self.opt.load_pt_folder, "encoder.pth")
            loaded_dict_enc = torch.load(encoder_path, map_location=self.device)
            loaded_dict_enc = {k.replace("module.", ""): v for k, v in loaded_dict_enc.items()}
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models["encoder"].state_dict()}
            self.models["encoder"].load_state_dict(filtered_dict_enc)

        #self.models["encoder"] = self.models["encoder"].cuda() #GPU
        #self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"]) #GPU
        self.models["encoder"] = self.models["encoder"].to(self.device) #CPU
        if torch.cuda.is_available() and not self.opt.no_cuda:
            self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"])
        
        
        if self.opt.backbone.endswith("_lite"):
            self.models["depth"] = networks.Lite_Depth_Decoder_QueryTr(in_channels=self.opt.model_dim, patch_size=self.opt.patch_size, dim_out=self.opt.dim_out, embedding_dim=self.opt.model_dim, 
                                                                    query_nums=self.opt.query_nums, num_heads=4, min_val=self.opt.min_depth, max_val=self.opt.max_depth)
        else:
            self.models["depth"] = networks.Depth_Decoder_QueryTr(in_channels=self.opt.model_dim, patch_size=self.opt.patch_size, dim_out=self.opt.dim_out, embedding_dim=self.opt.model_dim, 
                                                                    query_nums=self.opt.query_nums, num_heads=4, min_val=self.opt.min_depth, max_val=self.opt.max_depth)

        if self.opt.load_pretrained_model:
            print("-> Loading pretrained depth decoder from ", self.opt.load_pt_folder)
            depth_decoder_path = os.path.join(self.opt.load_pt_folder, "depth.pth")
            loaded_dict_enc = torch.load(depth_decoder_path, map_location=self.device)
            loaded_dict_enc = {k.replace("module.", ""): v for k, v in loaded_dict_enc.items()}
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models["depth"].state_dict()}
            self.models["depth"].load_state_dict(filtered_dict_enc)

        #self.models["depth"] = self.models["depth"].cuda() #GPU
#        self.models["depth"] = torch.nn.DataParallel(self.models["depth"])
        self.models["depth"] = self.models["depth"].to(self.device) #added for CPU
        self.parameters_to_train += list(self.models["depth"].parameters())


        self.models["pose"] = networks.PoseCNN(
            self.num_input_frames if self.opt.pose_model_input == "all" else 2) # default=2
        if self.opt.pretrained_pose :
            print(f'loaded pose from {self.opt.pose_net_path}')
            pose_net_path = os.path.join(self.opt.pose_net_path, 'pose.pth')
            state_dict = OrderedDict([
                (k.replace("module.", ""), v) for (k, v) in torch.load(pose_net_path).items()])
            self.models["pose"].load_state_dict(state_dict)
            print("-> Loading pretrained depth decoder from ", self.opt.pose_net_path)
            depth_decoder_path = os.path.join(self.opt.pose_net_path, "depth.pth")
            loaded_dict_enc = torch.load(depth_decoder_path, map_location=self.device)
            loaded_dict_enc = {k.replace("module.", ""): v for k, v in loaded_dict_enc.items()}
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models["depth"].state_dict()}
            self.models["depth"].load_state_dict(filtered_dict_enc)
        
        if torch.cuda.is_available() and not self.opt.no_cuda: #Added if there is GPU
            self.models["depth"] = torch.nn.DataParallel(self.models["depth"])

        self.models["pose"] = self.models["pose"].to(self.device)
        #self.models["pose"] = self.models["pose"].cuda() #GPU


        #self.models["pose"] = torch.nn.DataParallel(self.models["pose"])
        if self.opt.diff_lr :
            print("using diff lr for depth-net and pose-net")
            self.pose_params = []
            self.pose_params += list(self.models["encoder"].parameters())
        else :
            self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["pose"].parameters())

        # if self.opt.predictive_mask:
        #     assert self.opt.disable_automasking, \
        #         "When using predictive_mask, please disable automasking with --disable_automasking"

        #     # Our implementation of the predictive masking baseline has the the same architecture
        #     # as our depth decoder. We predict a separate mask for each source frame.
        #     self.models["predictive_mask"] = networks.DepthDecoder(
        #         self.models["encoder"].num_ch_enc, self.opt.scales,
        #         num_output_channels=(len(self.opt.frame_ids) - 1))
        #     self.models["predictive_mask"].to(self.device)
        #     self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        if self.opt.diff_lr :
            df_params = [{"params": self.pose_params, "lr": self.opt.learning_rate / 10},
                      {"params": self.parameters_to_train, "lr": self.opt.learning_rate}]
            self.model_optimizer = optim.Adam(df_params, lr=self.opt.learning_rate)
        else : 
            self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate) # default=1e-4
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1) # default=15

        #if self.opt.load_weights_folder is not None:
        #     self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir) # default to ~/tmp/mdp/train
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                         "midair": datasets.MidAirDataset}
        self.dataset = datasets_dict[self.opt.dataset] # default="kitti"

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        """train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 1, is_train=True, img_ext=img_ext) # num_scales = 1"""
        #Edited the train dataset to crop the car the image selecting the car
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 1, is_train=True, img_ext=img_ext, crop_box=self.opt.crop_box)

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        """val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 1, is_train=False, img_ext=img_ext) # num_scales = 1"""
        #Edited the val dataset to crop the car the image selecting the car
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 1, is_train=False, img_ext=img_ext, crop_box=self.opt.crop_box)


        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{self.opt.batch_size}-tep{self.opt.num_epochs}-lr{self.opt.learning_rate}--{uuid.uuid4()}"
        name = f"{experiment_name}_{run_id}"
        #wandb.init(project=PROJECT, name=name, config=self.opt, dir='.')
        wandb.init(project=PROJECT, name=name, config=self.opt, dir=self.opt.log_dir)
        self.save_model()
        for self.epoch in range(self.opt.num_epochs):
            wandb.log({"Epoch": self.epoch}, step=self.step) #Added to log in wandb
            self.run_epoch()
            self.model_lr_scheduler.step()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
        self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            #Wandb loging
            should_log = True
            if should_log and self.step % 10 == 0:
              #wandb.log({f"Train/reprojection_loss": losses["loss"].item()}, step=self.step)
              wandb.log({
                    "Train/reprojection_loss": losses["loss"].item(),
                    "Train/smoothness_loss": losses["reproj_loss"].item(),  
                    "Train/total_loss": losses["smooth_loss"].item()
                }, step=self.step)

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 1000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                #if batch_idx % (self.opt.log_frequency*2 )== 0: 
                #    self.val() #Evaluate after the double of log_frequency
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared": # default no
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])

            outputs = self.models["depth"](features)

        if self.opt.predictive_mask: # default no
            outputs["predictive_mask"] = self.models["predictive_mask"](features)
        # self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        if self.use_pose_net: # default=True
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}

        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    # print(axisangle.shape)
                    # axisangle:[12, 1, 1, 3]  translation:[12, 1, 1, 3]
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    # outputs[("cam_T_cam", 0, f_i)]: [12, 4, 4]

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            # inputs = self.val_iter.next() # for old pytorch
            inputs = next(self.val_iter) # for new pytorch
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            # inputs = self.val_iter.next()
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            # Log to W&B
            wandb.log({
                "Val/total_loss": losses["loss"].item(),
                "Val/reprojection_loss": losses["reproj_loss"].item() ,
                "Val/smoothness_loss": losses["smooth_loss"].item()
            }, step=self.step)

            del inputs, outputs, losses

        self.set_train()

        """Validate the model on the entire validation set (used because of the small val test)"""
        """self.set_eval()

        val_losses = []
        smooth_losses = []
        reproj_losses = []

        with torch.no_grad():
            val_total_count= len(self.val_loader.dataset)
            count=0
            for inputs in self.val_loader:
                count+=1
                outputs, losses = self.process_batch(inputs)

                val_losses.append(losses["loss"].item())
                reproj_losses.append(losses["reproj_loss"].item())
                smooth_losses.append(losses["smooth_loss"].item())

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                if count==val_total_count:
                    self.log("val", inputs, outputs, losses) #Log the last image from validation test
                del inputs, outputs, losses

        # Average over validation set
        avg_loss = sum(val_losses) / len(val_losses)
        avg_reproj = sum(reproj_losses) / len(reproj_losses)
        avg_smooth = sum(smooth_losses) / len(smooth_losses)

        # Log to W&B
        wandb.log({
            "Val/total_loss": avg_loss,
            "Val/reprojection_loss": avg_reproj,
            "Val/smoothness_loss": avg_smooth
        }, step=self.step)

        self.set_train()"""

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

                depth = disp
            # _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn" and not self.opt.use_stereo:

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                # pix_coords: [bs, h, w, 2]

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                #weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).to(self.device))
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            #Save raw reprojection loss (mean over batch)
            losses["reproj_loss"] = reprojection_loss.mean()

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001 #.cuda() was replaced with to(self.device)

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            if color.shape[-2:] != disp.shape[-2:]:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            # if GPU memory is not enough, you can downsample color instead
            # color = F.interpolate(color, [self.opt.height // 2, self.opt.width // 2], mode="bilinear", align_corners=False)
            smooth_loss = 0
            smooth_loss = get_smooth_loss(norm_disp, color)
            # smooth_loss

             # Save smoothness loss per scale
            losses["smooth_loss"] = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        _, _, h_gt, w_gt = inputs["depth_gt"].shape #Added
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [h_gt, w_gt], mode="bilinear", align_corners=False), 1e-3, 80)
        #depth_pred = torch.clamp(F.interpolate(
        #    depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        if self.opt.dataset=="kitti":
            crop_mask = torch.zeros_like(mask)
            crop_mask[:, :, 153:371, 44:1197] = 1
            mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk, default /home/Process3/tmp/mdp/models/
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            # for nn.DataParallel models, you must use model.module.state_dict() instead of model.state_dict()
            #if model_name == 'pose':
            #   to_save = model.state_dict()
            #else:
            #    to_save = model.module.state_dict()
            to_save = model.state_dict() #Considering that all models are not Dataparallel (CPU models)
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


