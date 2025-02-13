# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
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

import os
import math
import argparse
import shutil
import datetime
import logging
from omegaconf import OmegaConf

from tqdm.auto import tqdm
from einops import rearrange

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.logging import get_logger
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed

from latentsync.data.unet_dataset import UNetDataset
from latentsync.models.unet import UNet3DConditionModel
from latentsync.models.syncnet import SyncNet
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.utils.util import (
    init_dist,
    cosine_loss,
    reversed_forward,
)
from latentsync.utils.util import plot_loss_chart, gather_loss
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.trepa import TREPALoss
from eval.syncnet import SyncNetEval
from eval.syncnet_detect import SyncNetDetector
from eval.eval_sync_conf import syncnet_eval
import lpips
from peft import LoraConfig, get_peft_model, PeftModel

logger = get_logger(__name__)


def main(config):
    # Initialize distributed training
    local_rank = init_dist()
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = config.run.seed + global_rank
    set_seed(seed)

    # Logging folder
    folder_name = "train" + datetime.datetime.now().strftime(f"-%Y_%m_%d-%H:%M:%S")
    output_dir = os.path.join(config.data.train_output_dir, folder_name)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Handle the output folder creation
    if is_main_process:
        diffusers.utils.logging.set_verbosity_info()
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/val_videos", exist_ok=True)
        os.makedirs(f"{output_dir}/loss_charts", exist_ok=True)
        shutil.copy(config.unet_config_path, output_dir)
        shutil.copy(config.data.syncnet_config_path, output_dir)

    device = torch.device(local_rank)

    noise_scheduler = DDIMScheduler.from_pretrained("configs")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae.requires_grad_(False)
    vae.to(device)

    syncnet_eval_model = SyncNetEval(device=device)
    syncnet_eval_model.loadParameters("checkpoints/auxiliary/syncnet_v2.model")

    syncnet_detector = SyncNetDetector(device=device, detect_results_dir="detect_results")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device=device,
        audio_embeds_cache_dir=config.data.audio_embeds_cache_dir,
        num_frames=config.data.num_frames,
    )

    unet, resume_global_step = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        config.ckpt.resume_ckpt_path,  # load checkpoint
        device=device,
    )

    if config.model.add_audio_layer and config.run.use_syncnet:
        syncnet_config = OmegaConf.load(config.data.syncnet_config_path)
        if syncnet_config.ckpt.inference_ckpt_path == "":
            raise ValueError("SyncNet path is not provided")
        syncnet = SyncNet(OmegaConf.to_container(syncnet_config.model)).to(device=device, dtype=torch.float16)
        syncnet_checkpoint = torch.load(syncnet_config.ckpt.inference_ckpt_path, map_location=device)
        syncnet.load_state_dict(syncnet_checkpoint["state_dict"])
        syncnet.requires_grad_(False)

    # Injecting LoRA layers into unet
    if config.lora.use_lora:
        if config.lora.resume_lora_path:
            unet = PeftModel.from_pretrained(unet, config.lora.resume_lora_path, is_trainable=True)
            resume_global_step = torch.load(os.path.join(config.lora.resume_lora_path, "state_dict.pt"), map_location=device)["global_step"]
        else:
            lora_config = LoraConfig(
                r=config.lora.r,                      
                lora_alpha=config.lora.lora_alpha,            
                target_modules=config.lora.target_modules,  
                bias=config.lora.bias,
                lora_dropout=config.lora.lora_dropout,
                task_type=config.lora.task_type,      
            )
            unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
        trainable_params = [p for p in unet.parameters() if p.requires_grad]
    else:
        unet.requires_grad_(True)
        trainable_params = list(unet.parameters())

    if config.optimizer.scale_lr:
        config.optimizer.lr = config.optimizer.lr * num_processes

    optimizer = torch.optim.AdamW(trainable_params, lr=config.optimizer.lr)

    if is_main_process:
        logger.info(f"trainable params number: {len(trainable_params)}")
        logger.info(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable xformers
    if config.run.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if config.run.enable_gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Get the training dataset
    train_dataset = UNetDataset(config.data.train_data_dir, config)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=config.run.seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=config.data.num_workers,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=train_dataset.worker_init_fn,
    )

    # Get the training iteration
    if config.run.max_train_steps == -1:
        assert config.run.max_train_epochs != -1
        config.run.max_train_steps = config.run.max_train_epochs * len(train_dataloader)

    # Scheduler
    lr_scheduler = get_scheduler(
        config.optimizer.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.optimizer.lr_warmup_steps,
        num_training_steps=config.run.max_train_steps,
    )

    if config.run.perceptual_loss_weight != 0 and config.run.pixel_space_supervise:
        lpips_loss_func = lpips.LPIPS(net="vgg").to(device)

    if config.run.trepa_loss_weight != 0 and config.run.pixel_space_supervise:
        trepa_loss_func = TREPALoss(device=device)

    # Validation pipeline
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=noise_scheduler,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    # DDP warpper
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(config.run.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = config.data.batch_size * num_processes

    if is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {config.data.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps = {config.run.max_train_steps}")
    global_step = resume_global_step
    first_epoch = resume_global_step // num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(0, config.run.max_train_steps),
        initial=resume_global_step,
        desc="Steps",
        disable=not is_main_process,
    )

    train_step_list = []
    sync_loss_list = []
    recon_loss_list = []

    val_step_list = []
    sync_conf_list = []

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if config.run.mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()

        for step, batch in enumerate(train_dataloader):
            ### >>>> Training >>>> ###

            if config.model.add_audio_layer:
                if batch["mel"] != []:
                    mel = batch["mel"].to(device, dtype=torch.float16)

                audio_embeds_list = []
                try:
                    for idx in range(len(batch["video_path"])):
                        video_path = batch["video_path"][idx]
                        start_idx = batch["start_idx"][idx]

                        with torch.no_grad():
                            audio_feat = audio_encoder.audio2feat(video_path)
                        audio_embeds = audio_encoder.crop_overlap_audio_window(audio_feat, start_idx)
                        audio_embeds_list.append(audio_embeds)
                except Exception as e:
                    logger.info(f"{type(e).__name__} - {e} - {video_path}")
                    continue
                audio_embeds = torch.stack(audio_embeds_list)  # (B, 16, 50, 384)
                audio_embeds = audio_embeds.to(device, dtype=torch.float16)
            else:
                audio_embeds = None

            # Convert videos to latent space
            gt_images = batch["gt"].to(device, dtype=torch.float16)
            gt_masked_images = batch["masked_gt"].to(device, dtype=torch.float16)
            mask = batch["mask"].to(device, dtype=torch.float16)
            ref_images = batch["ref"].to(device, dtype=torch.float16)

            gt_images = rearrange(gt_images, "b f c h w -> (b f) c h w")
            gt_masked_images = rearrange(gt_masked_images, "b f c h w -> (b f) c h w")
            mask = rearrange(mask, "b f c h w -> (b f) c h w")
            ref_images = rearrange(ref_images, "b f c h w -> (b f) c h w")

            with torch.no_grad():
                gt_latents = vae.encode(gt_images).latent_dist.sample()
                gt_masked_images = vae.encode(gt_masked_images).latent_dist.sample()
                ref_images = vae.encode(ref_images).latent_dist.sample()

            mask = torch.nn.functional.interpolate(mask, size=config.data.resolution // vae_scale_factor)

            gt_latents = (
                rearrange(gt_latents, "(b f) c h w -> b c f h w", f=config.data.num_frames) - vae.config.shift_factor
            ) * vae.config.scaling_factor
            gt_masked_images = (
                rearrange(gt_masked_images, "(b f) c h w -> b c f h w", f=config.data.num_frames)
                - vae.config.shift_factor
            ) * vae.config.scaling_factor
            ref_images = (
                rearrange(ref_images, "(b f) c h w -> b c f h w", f=config.data.num_frames) - vae.config.shift_factor
            ) * vae.config.scaling_factor
            mask = rearrange(mask, "(b f) c h w -> b c f h w", f=config.data.num_frames)

            # Sample noise that we'll add to the latents
            if config.run.use_mixed_noise:
                # Refer to the paper: https://arxiv.org/abs/2305.10474
                noise_shared_std_dev = (config.run.mixed_noise_alpha**2 / (1 + config.run.mixed_noise_alpha**2)) ** 0.5
                noise_shared = torch.randn_like(gt_latents) * noise_shared_std_dev
                noise_shared = noise_shared[:, :, 0:1].repeat(1, 1, config.data.num_frames, 1, 1)

                noise_ind_std_dev = (1 / (1 + config.run.mixed_noise_alpha**2)) ** 0.5
                noise_ind = torch.randn_like(gt_latents) * noise_ind_std_dev
                noise = noise_ind + noise_shared
            else:
                noise = torch.randn_like(gt_latents)
                noise = noise[:, :, 0:1].repeat(
                    1, 1, config.data.num_frames, 1, 1
                )  # Using the same noise for all frames, refer to the paper: https://arxiv.org/abs/2308.09716

            bsz = gt_latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=gt_latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_tensor = noise_scheduler.add_noise(gt_latents, noise, timesteps)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            unet_input = torch.cat([noisy_tensor, mask, gt_masked_images, ref_images], dim=1)

            # Predict the noise and compute loss
            # Mixed-precision training
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.run.mixed_precision_training):
                pred_noise = unet(unet_input, timesteps, encoder_hidden_states=audio_embeds).sample

            if config.run.recon_loss_weight != 0:
                recon_loss = F.mse_loss(pred_noise.float(), target.float(), reduction="mean")
            else:
                recon_loss = 0

            pred_latents = reversed_forward(noise_scheduler, pred_noise, timesteps, noisy_tensor)

            if config.run.pixel_space_supervise:
                pred_images = vae.decode(
                    rearrange(pred_latents, "b c f h w -> (b f) c h w") / vae.config.scaling_factor
                    + vae.config.shift_factor
                ).sample

            if config.run.perceptual_loss_weight != 0 and config.run.pixel_space_supervise:
                pred_images_perceptual = pred_images[:, :, pred_images.shape[2] // 2 :, :]
                gt_images_perceptual = gt_images[:, :, gt_images.shape[2] // 2 :, :]
                lpips_loss = lpips_loss_func(pred_images_perceptual.float(), gt_images_perceptual.float()).mean()
            else:
                lpips_loss = 0

            if config.run.trepa_loss_weight != 0 and config.run.pixel_space_supervise:
                trepa_pred_images = rearrange(pred_images, "(b f) c h w -> b c f h w", f=config.data.num_frames)
                trepa_gt_images = rearrange(gt_images, "(b f) c h w -> b c f h w", f=config.data.num_frames)
                trepa_loss = trepa_loss_func(trepa_pred_images, trepa_gt_images)
            else:
                trepa_loss = 0

            if config.model.add_audio_layer and config.run.use_syncnet:
                if config.run.pixel_space_supervise:
                    syncnet_input = rearrange(pred_images, "(b f) c h w -> b (f c) h w", f=config.data.num_frames)
                else:
                    syncnet_input = rearrange(pred_latents, "b c f h w -> b (f c) h w")

                if syncnet_config.data.lower_half:
                    height = syncnet_input.shape[2]
                    syncnet_input = syncnet_input[:, :, height // 2 :, :]
                ones_tensor = torch.ones((config.data.batch_size, 1)).float().to(device=device)
                vision_embeds, audio_embeds = syncnet(syncnet_input, mel)
                sync_loss = cosine_loss(vision_embeds.float(), audio_embeds.float(), ones_tensor).mean()
                sync_loss_list.append(gather_loss(sync_loss, device))
            else:
                sync_loss = 0

            loss = (
                recon_loss * config.run.recon_loss_weight
                + sync_loss * config.run.sync_loss_weight
                + lpips_loss * config.run.perceptual_loss_weight
                + trepa_loss * config.run.trepa_loss_weight
            )

            train_step_list.append(global_step)
            if config.run.recon_loss_weight != 0:
                recon_loss_list.append(gather_loss(recon_loss, device))

            optimizer.zero_grad()

            # Backpropagate
            if config.run.mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), config.optimizer.max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(unet.parameters(), config.optimizer.max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            # Check the grad of attn blocks for debugging
            # print(unet.module.up_blocks[3].attentions[2].transformer_blocks[0].audio_cross_attn.attn.to_q.weight.grad)

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            ### <<<< Training <<<< ###

            # Save checkpoint and conduct validation
            if is_main_process and (global_step % config.ckpt.save_ckpt_steps == 0):
                if config.run.recon_loss_weight != 0:
                    plot_loss_chart(
                        os.path.join(output_dir, f"loss_charts/recon_loss_chart-{global_step}.png"),
                        ("Reconstruction loss", train_step_list, recon_loss_list),
                    )
                if config.model.add_audio_layer:
                    if sync_loss_list != []:
                        plot_loss_chart(
                            os.path.join(output_dir, f"loss_charts/sync_loss_chart-{global_step}.png"),
                            ("Sync loss", train_step_list, sync_loss_list),
                        )
                model_save_path = os.path.join(output_dir, f"checkpoints/checkpoint-{global_step}.pt")
                
                if config.lora.use_lora:
                    lora_save_path = os.path.join(output_dir, f"checkpoints/checkpoint-{global_step}")
                    state_dict = {
                        "global_step": global_step,
                    }
                    try:
                        unet.module.save_pretrained(lora_save_path, save_adapter=True)
                        torch.save(state_dict, os.path.join(lora_save_path, "state_dict.pt"))
                        logger.info(f"Saved lora checkpoint to {lora_save_path}")
                    except Exception as e:
                        logger.error(f"Error saving model: {e}")
                else:
                    state_dict = {
                        "global_step": global_step,
                        "state_dict": unet.module.state_dict(),  # to unwrap DDP
                    }
                    try:
                        torch.save(state_dict, model_save_path)
                        logger.info(f"Saved checkpoint to {model_save_path}")
                    except Exception as e:
                        logger.error(f"Error saving model: {e}")

                # Validation
                logger.info("Running validation... ")

                validation_video_out_path = os.path.join(output_dir, f"val_videos/val_video_{global_step}.mp4")
                validation_video_mask_path = os.path.join(output_dir, f"val_videos/val_video_mask.mp4")

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pipeline(
                        config.data.val_video_path,
                        config.data.val_audio_path,
                        validation_video_out_path,
                        validation_video_mask_path,
                        num_frames=config.data.num_frames,
                        num_inference_steps=config.run.inference_steps,
                        guidance_scale=config.run.guidance_scale,
                        weight_dtype=torch.float16,
                        width=config.data.resolution,
                        height=config.data.resolution,
                        mask=config.data.mask,
                    )

                logger.info(f"Saved validation video output to {validation_video_out_path}")

                val_step_list.append(global_step)

                if config.model.add_audio_layer:
                    try:
                        _, conf = syncnet_eval(syncnet_eval_model, syncnet_detector, validation_video_out_path, "temp")
                    except Exception as e:
                        logger.info(e)
                        conf = 0
                    sync_conf_list.append(conf)
                    plot_loss_chart(
                        os.path.join(output_dir, f"loss_charts/sync_conf_chart-{global_step}.png"),
                        ("Sync confidence", val_step_list, sync_conf_list),
                    )

            logs = {"step_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= config.run.max_train_steps:
                break

    progress_bar.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Config file path
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")

    args = parser.parse_args()
    config = OmegaConf.load(args.unet_config_path)
    config.unet_config_path = args.unet_config_path

    main(config)
