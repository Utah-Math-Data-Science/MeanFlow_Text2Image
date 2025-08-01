# Copyright © 2025
# FlowTok‑MeanFlow‑Sob — **Distributed Data Parallel (DDP) edition**
# ================================================================
# This script is a drop‑in replacement for the original
# `flowtok_mean_flow_sob.py`, refactored to use native PyTorch
# Distributed Data Parallel instead of 🤗 Accelerate.  Launch with:
#
#     torchrun --nproc_per_node=<GPUs> flowtok_mean_flow_sob_ddp.py \
#         train --dataset <path> [... other args ...]
#
# The CLI, model, loss and sampling logic are unchanged; only the
# infrastructure around distributed training and checkpointing differs.

from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from typing import Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd.functional import jvp
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image

from datasets import load_from_disk
from diffusers.models import AutoencoderKL
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    AutoTokenizer,
    AutoModel,
)

from models.model import FlowTokLite  # local codebase
from models.EMA import EMA           # local codebase
from functools import partial
from PIL import Image, ImageDraw, ImageFont
import wandb

################################################################################
# Helper utilities
################################################################################

def setup_distributed(local_rank: int, port: str | None = None):
    """Initialise NCCL process‑group for multi‑GPU training."""
    if dist.is_initialized():
        return  # already set up (e.g. by torchrun)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if port is not None:
        os.environ.setdefault("MASTER_PORT", port)
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

################################################################################
# Flow‑matching helper targets (unchanged)
################################################################################

def make_targets(
    txt_tokens: torch.Tensor,
    img_tokens: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_t = (1 - t)[:, None, None, None] * img_tokens + t[:, None, None, None] * txt_tokens
    v = txt_tokens - img_tokens
    return z_t, v, torch.ones_like(txt_tokens)

def adaptive_l2_loss(error: torch.Tensor, gamma: float = 1.0, c: float = 1e-3):
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    return (w.detach() * delta_sq).mean()

################################################################################
# Parameter counting (helper)
################################################################################

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

################################################################################
# Main training loop (DDP)
################################################################################

def train(args):
    local_rank = args.local_rank
    setup_distributed(local_rank, port=args.dist_port)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    # Seed – make sure each worker gets a different but deterministic seed
    torch.manual_seed(42 + rank)

    # ---------------------------------------------------------------------
    # Config / model / optimiser
    # ---------------------------------------------------------------------
    class CFG:
        n_layers = 3
        d_model = 256
        n_heads = 4
        seq_len = 20
        img_size = args.img_size
        frozen_text_proj = args.frozen_text_proj
        model = args.model

    cfg = CFG()
    model = FlowTokLite(cfg).to(device)

    args.ckpt_out = args.ckpt_out + str(int(1000*args.noise_scale)) + '_sob_model' + str(args.model) + '_txtreg_{}'.format(int(100000*args.txt_reg)) + '.pt'

    # Wrap in DistributedDataParallel (find_unused_parameters handles jvp path)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    # ---------------------------------------------------------------------
    # Dataset / DataLoader (DistributedSampler)
    # ---------------------------------------------------------------------
    ds = load_from_disk(os.path.join(args.dataset, "train"))
    img_trans = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size), antialias=True),
        transforms.ToTensor(),
    ])

    def collate(batch):
        imgs, captions = zip(*[(img_trans(b["image"]), b["text"]) for b in batch])
        return torch.stack(imgs), list(captions)

    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
    )

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer = optim,
        num_warmup_steps = len(loader) * args.epochs * 0.02,
        num_training_steps = len(loader) * args.epochs,
        lr_end = 1e-8,
        power = 0.7,
    )

    # ---------------------------------------------------------------------
    # Externals: VAE + text encoder (frozen)
    # ---------------------------------------------------------------------
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    pre_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base")
    pre_model = AutoModel.from_pretrained("intfloat/e5-base").to(device).eval()

    ema = EMA(model.module, decay=0.9995)  # Track original (unwrapped) model
    jvp_fn = partial(torch.autograd.functional.jvp, create_graph=True)
    scale_ = 0.18215
    noise_scale = args.noise_scale

    # ---------------------------------------------------------------------
    # W&B (log only from rank‑0)
    # ---------------------------------------------------------------------
    if rank == 0:
        wandb.init(
            entity="utah-math-data-science",
            project="Flow_Matching_Text2Image",
            mode=args.wandb,
            name='Text2Image_MFlow_{}_TXTnoise{}_lr{}_frzn{}_TSample{}_flowR{}_gamma{}_txtReg{}'.format(args.run_name, 
                                                                                                        args.noise_scale, 
                                                                                                        args.lr, 
                                                                                                        args.frozen_text_proj, 
                                                                                                        args.t_sample, 
                                                                                                        args.flow_ratio, 
                                                                                                        args.gamma, 
                                                                                                        args.txt_reg),
            config=vars(args),
            settings=wandb.Settings(_disable_stats=True),
            reinit=True,
        )
        wandb.save("*.txt")

    # ---------------------------------------------------------------------
    # Training epochs
    # ---------------------------------------------------------------------
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_1loss = 0.0
        epoch_2loss = 0.0
        epoch_embeddings_std = 0.0
        epoch_tix_tok_std = 0.0
        epoch_derror = 0.0
        start_time = time.time()

        for step, (imgs, captions) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            # ---------------------------------------------------------
            # Sample (t, r)
            # ---------------------------------------------------------
            if args.t_sample == 'log':
                normal_samples = torch.randn((imgs.size(0), 2), device=device) * 1.0 - 0.4
                samples = 1 / (1 + torch.exp(-normal_samples))  # sigmoid to map to (0,1)
                # t is max
                t = torch.max(samples[:, 0], samples[:, 1])  # ensure t >= r
                # r is min
                r_ = torch.min(samples[:, 0], samples[:, 1])  # ensure r <= t
            elif args.t_sample == 'uniform_1':
                samples = torch.rand((imgs.size(0), 2), device=device)
                t = torch.max(samples[:, 0], samples[:, 1])  # ensure t >= r
                r_ = torch.min(samples[:, 0], samples[:, 1])
            else:
                raise ValueError(f"Unknown t_sample method: {args.t_sample}")
            
            select = torch.rand(imgs.size(0), device=device) < args.flow_ratio
            r_[select] = t[select]

            # ---------------------------------------------------------
            # Pre‑process image / text tokens
            # ---------------------------------------------------------
            with torch.no_grad():
                img_tok = vae.encode(imgs).latent_dist.mode() * scale_ 
                img_tok = img_tok + 0.05 * torch.randn_like(img_tok)
                # text to latent
                tokens = pre_tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(device)
                output = pre_model(**tokens)
                embeddings = output.last_hidden_state[:, 0]  # [CLS]-like 

            txt_tok = model.module.text_to_latent(embeddings)
            reg = txt_tok.norm(p=2, dim=(1, 2, 3)).mean() * args.txt_reg
            txt_tok = txt_tok + noise_scale * torch.randn_like(txt_tok)

            # ---------------------------------------------------------
            # Loss computation (unchanged)
            # ---------------------------------------------------------
            def u_fn(x, r_, t):
                return model(x, t - r_, t)

            v = None
            d_v_d_txt = None

            def loss_fn(img_tok, txt_tok, r_, t):
                nonlocal v, d_v_d_txt
                z_t, v, d_v_d_txt = make_targets(txt_tok, img_tok, t)
                v_pred, dvdt = jvp_fn(u_fn, (z_t, r_, t), 
                                      (v.detach(), torch.zeros_like(r_), torch.ones_like(t)))
                return v_pred, dvdt

            eps = torch.randn_like(txt_tok)
            (primal_pair, tangent_pair) = jvp_fn(
                                                loss_fn,
                                                (img_tok, txt_tok, r_, t),
                                                (
                                                torch.zeros_like(img_tok),
                                                eps,
                                                torch.zeros_like(r_),
                                                torch.zeros_like(t),
                                                ),
                                                )
            v_pred, dvdt = primal_pair
            d_v_pred_d_txt, d_dvdt_d_txt = tangent_pair

            v_trgt = v - (t - r_)[:, None, None, None] * dvdt.detach()
            d_v_d_txt_trgt = d_v_d_txt * eps - (t - r_)[:, None, None, None] * d_dvdt_d_txt.detach()

            error1 = v_pred - v_trgt
            error2 = d_v_pred_d_txt - d_v_d_txt_trgt

            loss1 = adaptive_l2_loss(error1, gamma=args.gamma, c=1e-3)
            loss2 = adaptive_l2_loss(error2, gamma=args.gamma, c=1e-3)
            loss = loss1 + args.sob_lambda * loss2 + reg

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()
            ema.update()

            # record
            epoch_1loss += loss1.item()
            epoch_2loss += loss2.item()
            epoch_loss += loss.item()
            epoch_embeddings_std += embeddings.std().item()
            epoch_tix_tok_std += txt_tok.std().item()
            epoch_derror += torch.mean(torch.square(dvdt.detach()))

        # -----------------------------------------------------------------
        # Metrics aggregation (average across all GPUs)
        # -----------------------------------------------------------------
        epoch_loss = torch.tensor(epoch_loss / len(loader), device=device)
        epoch_1loss = torch.tensor(epoch_1loss / len(loader), device=device)
        epoch_2loss = torch.tensor(epoch_2loss / len(loader), device=device)
        epoch_embeddings_std = torch.tensor(epoch_embeddings_std / len(loader), device=device)
        epoch_tix_tok_std = torch.tensor(epoch_tix_tok_std / len(loader), device=device)
        epoch_derror = torch.tensor(epoch_derror / len(loader), device=device)
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_1loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_2loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_embeddings_std, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_tix_tok_std, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_derror, op=dist.ReduceOp.SUM)
        epoch_loss /= world_size
        epoch_1loss /= world_size
        epoch_2loss /= world_size
        epoch_embeddings_std /= world_size
        epoch_tix_tok_std /= world_size
        epoch_derror /= world_size

        elapsed = time.time() - start_time

        if rank == 0:

            ############################# sample start #############################
            if epoch % 50 == 0:
                dt = 0.2
                ema.apply_shadow()
                with torch.no_grad():
                    img_tok_recon = txt_tok.detach() - dt * u_fn(txt_tok.detach(), (1-dt) * torch.ones_like(t), torch.ones_like(t))
                    img_tok_recon = img_tok_recon - (1-dt) * u_fn(img_tok_recon, torch.zeros_like(t), (1-dt) * torch.ones_like(t))
                    img_recon = vae.decode(img_tok_recon / scale_).sample
                    img_recon_gd = vae.decode(img_tok / scale_).sample
                ema.restore()

                dir_path = '/mntc/yuhaoh/programme/MeanFlow_Text2Image/' + 'meanflow_imgs_{}_{}'.format(int(1000*args.noise_scale), args.model)
                Path(dir_path).mkdir(parents=True, exist_ok=True)

                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 24)
                except IOError:
                    font = ImageFont.load_default()
                padding = 20
                line_height = font.getbbox("Hg")[3] + 10        # approximate per-line height
                image_width = 800
                image_height = padding * 2 + line_height * 12
                # Create blank white image
                img = Image.new("RGB", (image_width, image_height), color="white")
                draw = ImageDraw.Draw(img)
                # Render each caption line
                for i, caption in enumerate(captions):
                    y = padding + i * line_height
                    caption = '({}) '.format(i+1) + caption
                    draw.text((padding, y), caption, font=font, fill="black")
                    if i == 11:
                        break
                img.save(f"{dir_path}/captions_epoch_{epoch}.png")

                # Save images
                save_image(img_recon[:12], '{}/recon_batch_epoch_{}.png'.format(dir_path, epoch), nrow=6)
                save_image(img_recon_gd[:12], '{}/recon_batch_epoch_{}_gd.png'.format(dir_path, epoch), nrow=6)
                wandb.log({
                    "recon_batch": wandb.Image(f"{dir_path}/recon_batch_epoch_{epoch}.png"),
                    "recon_batch_gd": wandb.Image(f"{dir_path}/recon_batch_epoch_{epoch}_gd.png"),
                    "captions": wandb.Image(f"{dir_path}/captions_epoch_{epoch}.png")
                })
            ############################# sample end #############################

            lr = scheduler.get_last_lr()[0]
            print(
                f"[Epoch {epoch+1}/{args.epochs}] loss={epoch_loss:.4f} "
                f"(v={epoch_1loss:.4f}, sob={epoch_2loss:.4f}) lr={lr:.6f} "
                f"time={elapsed:.2f}s",
                flush=True,
            )
            wandb.log({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "learning_rate": lr,
                "time": elapsed,
                "epoch_derror": epoch_derror,
                "embeddings_std": epoch_embeddings_std,
                "txt_tok_std": epoch_tix_tok_std,
                "epoch_1loss": epoch_1loss,
                "epoch_2loss": epoch_2loss
            })

            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                ckpt = {
                    "model": model.module.state_dict(),
                    "ema": ema.shadow,
                    "optimizer": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                }
                ckpt_path = f"{args.ckpt_out}_epoch{epoch+1}.pt"
                torch.save(ckpt, ckpt_path)
                print(f"✓ Saved checkpoint to {ckpt_path}")

    dist.destroy_process_group()

################################################################################
# CLI
################################################################################

def build_parser():
    p = argparse.ArgumentParser(description="FlowTok‑Lite DDP: train or sample")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---------------- train ----------------
    p_train = sub.add_parser("train")
    p_train.add_argument("--dataset", type=str, default="flowers_blip_splits", help="Path to the dataset directory (Hugging Face format)")
    p_train.add_argument("--img_size", type=int, default=256)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--epochs", type=int, default=2000)
    p_train.add_argument("--ckpt_out", type=str, default="flowtok_mean_flow_")
    p_train.add_argument("--run_name", type=str, default="FlowTokLite")
    p_train.add_argument("--wandb", type=str, default="disabled")

    p_train.add_argument("--frozen_text_proj", action="store_true")
    p_train.add_argument("--noise_scale", type=float, default=0.1)
    p_train.add_argument("--model", type=str, default="mfunet")
    p_train.add_argument("--alpha", type=float, default=0.0)
    p_train.add_argument("--flow_ratio", type=float, default=0.75)
    p_train.add_argument("--gamma", type=float, default=0.5)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--t_sample", type=str, default='uniform_1',)
    p_train.add_argument("--sob_lambda", type=float, default=1e-2)
    p_train.add_argument("--txt_reg", type=float, default=1e-4)
    p_train.add_argument("--save_every", type=int, default=200)

    # DDP‑specific
    p_train.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    p_train.add_argument("--dist_port", type=str, default=os.environ.get("PORT", "29500"))

    # ---------------- sample (unchanged) ----------------
    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--ckpt", type=str, required=True)
    p_sample.add_argument("--prompt", type=str, required=True)
    p_sample.add_argument("--out", type=str, default="out.png")
    p_sample.add_argument("--steps", type=int, default=25)
    p_sample.add_argument("--sampler", choices=["euler", "rk38"], default="euler")

    return p

################################################################################
# Entry‑point
################################################################################

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    else:
        raise NotImplementedError("Sampling under DDP is not yet implemented in this refactor. Train the model first, then run a single‑GPU sampling script.")
