from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from tkinter import font
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp
# from torch.autograd.functional import jvp
from torch.nn.attention import sdpa_kernel, SDPBackend   # PyTorch ≥2.3
from torch.utils.data import DataLoader
from torchvision import transforms, utils as tvu
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator
from diffusers.models import AutoencoderKL
from models.model import FlowTokLite  # Assuming FlowTokLite is defined in model.py
from transformers import get_polynomial_decay_schedule_with_warmup, AutoTokenizer, AutoModel
from torch.autograd.functional import jvp
import wandb
import time
from torchvision.utils import save_image
from models.EMA import EMA  # Assuming EMA is defined in EMA.py
from functools import partial
from PIL import Image, ImageDraw, ImageFont
os.environ["TOKENIZERS_PARALLELISM"] = "false"

################################################################################
# Helper: make FM training targets
################################################################################
def make_targets(txt_tokens: torch.Tensor, img_tokens: torch.Tensor, t: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    # z_t = t[:, None, None, None] * img_tokens + (1 - t)[:, None, None, None] * txt_tokens
    z_t = (1-t)[:, None, None, None] * img_tokens + t[:, None, None, None] * txt_tokens
    v = (txt_tokens - img_tokens)  # Δ = 1
    return z_t, v

def sample_t_power(batch_size: int,
                   device=None,
                   alpha: float = 0.2,        # 0 ≤ α < 1
                   ) -> torch.Tensor:
    """
    Draw t ~ Beta(1, 1‑α), i.e.  p(t) ∝ (1‑t)^(-α)  on (0,1).

    Returns values in (eps, 1‑eps) so nothing ever hits the exact
    endpoints 0 or 1 (which avoids log/grad overflow).
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must be in [0, 1)")
    u = torch.rand(batch_size, device=device)            # U(0,1)
    t = 1.0 - u.pow(1.0 / (1.0 - alpha))                 # inverse‑CDF
    return t                                            # (eps, 1‑eps)

def stopgrad(x):
    return x.detach()

def adaptive_l2_loss(error, gamma=1.0, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  
    return (stopgrad(w) * loss).mean()

################################################################################
# Training loop
################################################################################

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable

jvp_fn = partial(torch.autograd.functional.jvp, create_graph=True)
def train(args):

    # seed
    torch.manual_seed(42)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    # Config
    class CFG:
        n_layers = 3
        # n_layers = 12
        # d_model = 768
        # n_heads = 12
        d_model = 256
        n_heads = 4
        seq_len = 20
        img_size = args.img_size
        frozen_text_proj = args.frozen_text_proj
        model = args.model
    cfg = CFG()

    model = FlowTokLite(cfg).to(device)

    # teacher_model = FlowTokLite(cfg).to(device)
    total_model, trainable_model = count_parameters(model)
    print('Number of Parameters in FlowTokLite:', total_model)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0)


    # ------------ dataset --------------
    # ds = load_dataset(args.dataset, split="train", token=True if args.hf_token else None)
    ds = load_from_disk(os.path.join(args.dataset, "train"))
    img_trans = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size), antialias=True),
        transforms.ToTensor(),
    ])

    def collate(batch):
        imgs, captions = zip(*[(img_trans(b["image"]), b["text"]) for b in batch])
        return torch.stack(imgs), list(captions)

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
                        num_workers=4, pin_memory=True)

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=len(loader) * args.epochs * 0.02,  
        num_training_steps=len(loader) * args.epochs,
        lr_end=1e-8,                # final learning rate
        power=0.7                  # 1.0 = linear, 2.0 = quadratic, etc.
    )
    

    model, optim, loader, scheduler = accelerator.prepare(model, optim, loader, scheduler)
    if args.sob_eps > 0:
        args.ckpt_out = args.ckpt_out + str(int(1000*args.noise_scale)) + '_sob_model' + str(args.model) + '_txtreg_{}'.format(int(100000*args.txt_reg)) + '.pt'
    else:
        args.ckpt_out = args.ckpt_out + str(int(1000*args.noise_scale)) + '_model' + str(args.model) + '_txtreg_{}'.format(int(100000*args.txt_reg)) + '.pt'

    # pretrained_chkpt = '/home/sci/yuhao/Programme/FlowTok/flowtok_mean_flow_100_modelmfunet_epoch795.pt'
    # if os.path.exists(pretrained_chkpt):
    #     print(f"Loading pre-trained model from {pretrained_chkpt}")
    #     checkpoint = torch.load(pretrained_chkpt, map_location=device)
    #     model.load_state_dict(checkpoint)
    
    # EMA model
    ema = EMA(model, decay=0.9999)
    
  
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval()
    vae.to(accelerator.device).eval()  # pre-trained VAE

    # count vae parameters
    total_vae, trainable_vae = count_parameters(vae)

    pre_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base")
    pre_model = AutoModel.from_pretrained("intfloat/e5-base")
    pre_model.to(accelerator.device).eval()  # pre-trained text encoder
    # count pre-trained model parameters
    total_pre, trainable_pre = count_parameters(pre_model)
    # report model parameters for these three models and sum them up

    # print(f"Total parameters: {total_model + total_vae + total_pre:,}")
    # print(f"Trainable parameters: {trainable_model + trainable_vae + trainable_pre:,}")


    model.train()
    best_loss = float("inf")
    scale_ =  0.18215  # scale to match VAE training
    noise_scale = args.noise_scale

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        if args.sob_eps > 0:
            epoch_1loss = 0.0
            epoch_2loss = 0.0
        epoch_embeddings_std = 0.0
        epoch_tix_tok_std = 0.0
        start_time = time.time()
        epoch_derror = 0.0

        for step, (imgs, captions) in enumerate(loader):

            if args.t_sample == 'log':
                normal_samples = torch.randn((imgs.size(0), 2), device=device) * 1.0 - 0.4
                samples = 1 / (1 + torch.exp(-normal_samples))  # sigmoid to map to (0,1)
                # t is max
                t = torch.max(samples[:, 0], samples[:, 1])  # ensure t >= r
                # r is min
                r_ = torch.min(samples[:, 0], samples[:, 1])  # ensure r <= t
            elif args.t_sample == 'uniform_0':
                t = torch.rand(imgs.size(0), device=device)
                # uniform (0,t)
                r_ = torch.rand(imgs.size(0), device=device) * t  # ensure r <= t
            elif args.t_sample == 'uniform_1':
                samples = torch.rand((imgs.size(0), 2), device=device)
                t = torch.max(samples[:, 0], samples[:, 1])  # ensure t >= r
                r_ = torch.min(samples[:, 0], samples[:, 1])
            
            # randomly select flow ratio
            select = torch.rand(imgs.size(0), device=device) < args.flow_ratio
            r_[select] = t[select]

            with torch.no_grad():
                # img to latent
                img_tok = vae.encode(imgs).latent_dist.mode() * scale_  # scale to match VAE training
                img_tok = img_tok + 0.05 * torch.randn_like(img_tok)  # add noise to image tokens

                # text to latent
                tokens = pre_tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(device)
                output = pre_model(**tokens)
                embeddings = output.last_hidden_state[:, 0]  # [CLS]-like 
                # txt_tok = teacher_model.text_to_latent(embeddings) * 1e-1

            txt_tok = model.text_to_latent(embeddings)  # text to latent
            reg = txt_tok.norm(p=2, dim=(1, 2, 3)).mean() * args.txt_reg
            txt_tok = txt_tok + noise_scale * torch.randn_like(txt_tok)  # add noise to text tokens

            def fn(x, r_, t):
                return model(x, t-r_, t)

            ############################# sample start #############################
            if epoch % 50 == 0 and step == 1:
                dt = 0.2
                ema.apply_shadow()
                with torch.no_grad():
                    img_tok_recon = txt_tok.detach() - dt * fn(txt_tok.detach(), (1-dt) * torch.ones_like(t), torch.ones_like(t))
                    img_tok_recon = img_tok_recon - (1-dt) * fn(img_tok_recon, torch.zeros_like(t), (1-dt) * torch.ones_like(t))
                    img_recon = vae.decode(img_tok_recon / scale_).sample
                    img_recon_gd = vae.decode(img_tok / scale_).sample
                ema.restore()

                dir_path = '/home/sci/yuhao/Programme/FlowTok/' + 'meanflow_imgs_{}_{}'.format(int(1000*args.noise_scale), args.model)
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
                img.save(f"{dir_path}/captions_epoch_{epoch+args.add_epoch}.png")

                # Save images
                save_image(img_recon[:12], '{}/recon_batch_epoch_{}.png'.format(dir_path, epoch+args.add_epoch), nrow=6)
                save_image(img_recon_gd[:12], '{}/recon_batch_epoch_{}_gd.png'.format(dir_path, epoch+args.add_epoch), nrow=6)
                wandb.log({
                    "recon_batch": wandb.Image(f"{dir_path}/recon_batch_epoch_{epoch+args.add_epoch}.png"),
                    "recon_batch_gd": wandb.Image(f"{dir_path}/recon_batch_epoch_{epoch+args.add_epoch}_gd.png"),
                    "captions": wandb.Image(f"{dir_path}/captions_epoch_{epoch+args.add_epoch}.png")
                })
            ############################# sample end #############################

            # ensure txt_tok and img_tok have the same shape
            z_t, v = make_targets(txt_tok, img_tok, t)
                

            v_pred, dvdt = jvp_fn(fn,
                                 (z_t, r_, t),
                                 (v, torch.zeros_like(r_), torch.ones_like(t)))

            v_tgt = v - ((t-r_)[:, None, None, None] * dvdt).detach()  # target velocity         
            
            err = v_pred - v_tgt
            loss = adaptive_l2_loss(err, gamma=args.gamma, c=1e-3)  # use adaptive L2 loss

            # l2 regularization
            loss += reg

            accelerator.backward(loss)
            optim.step()
            scheduler.step() 
            ema.update()
            optim.zero_grad()

            # record
            epoch_loss += loss.item()
            epoch_embeddings_std += embeddings.std().item()
            epoch_tix_tok_std += txt_tok.std().item()
            epoch_derror += torch.mean(torch.square(dvdt)).item()

        epoch_loss /= (step+1)
        epoch_embeddings_std /= (step+1)
        epoch_tix_tok_std /= (step+1)
        epoch_derror /= (step+1)

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss:.4f}")
        lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {lr:.6f}")
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        wandb.log({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "learning_rate": lr,
            "time": end_time - start_time,
            "epoch_derror": epoch_derror,
            "embeddings_std": epoch_embeddings_std,
            "txt_tok_std": epoch_tix_tok_std,
        })
        if best_loss > epoch_loss and epoch > 500:
            best_loss = epoch_loss

            accelerator.wait_for_everyone()
            # Save the model checkpoint
            if accelerator.is_main_process:
                torch.save({'model': model.state_dict(),
                            'ema': ema.shadow,
                            'optimizer': optim.state_dict(),
                            'scheduler': scheduler.state_dict()},
                            args.ckpt_out)
                print(f"Model saved to {args.ckpt_out}")



################################################################################
# CLI
################################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FlowTok‑Lite: train or sample")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---------------- train ----------------
    p_train = sub.add_parser("train")
    p_train.add_argument("--dataset", type=str, required=True,
                         help="HuggingFace dataset id with image & text fields 'image'/'text'.")
    p_train.add_argument("--img_size", type=int, default=256)
    p_train.add_argument("--batch", type=int, default=24)
    p_train.add_argument("--epochs", type=int, default=2000)
    p_train.add_argument("--ckpt_out", type=str, default="flowtok_mean_flow_")
    p_train.add_argument("--hf_token", type=str, default=None,
                         help="HF token for gated datasets (optional)")
    p_train.add_argument("--run_name", type=str, default="FlowTokLite",
                         help="W&B run name for tracking training progress")
    p_train.add_argument("--wandb", type=str, default='disabled',
                         help="Enable W&B logging (default: False)")

    p_train.add_argument("--frozen_text_proj", type=bool, default=False,
                         help="Use frozen text projection layer (default: False)")
    p_train.add_argument("--add_epoch", type=int, default=0,
                         help="Number of epochs to add for continued training (default: None)")
    
    p_train.add_argument("--noise_scale", type=float, default = 0.1)  # noise scale for text tokens
    
    p_train.add_argument("--model", type=str, default='mfunet',
                         help="select model type: 'unet' or 'flowtok_lite' (default: 'unet')")
    p_train.add_argument("--alpha", type=float, default=0.0,
                         help="Alpha value for sampling t in (0,1) (default: 0.2)")
    p_train.add_argument("--teacher", type=bool, default=False,
                         help="Use teacher forcing during training (default: False)")
    p_train.add_argument("--flow_ratio", type=float, default=0.75,
                         help="Flow ratio for training (default: 0.0)")
    p_train.add_argument("--gamma", type=float, default=0.5,
                         help="Gamma value for training (default: 0.5)")
    p_train.add_argument("--lr", type=float, default=1e-4,
                         help="Learning rate for the optimizer (default: 1e-5)")
    p_train.add_argument("--t_sample", type=str, default='uniform_1',)
    p_train.add_argument("--sob_eps", type=float, default=1e-6,)
    p_train.add_argument("--sob_lambda", type=float, default=5e-3)
    p_train.add_argument("--txt_reg", type=float, default=1e-3,)

    # ---------------- sample ----------------
    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--ckpt", type=str, required=True)
    p_sample.add_argument("--prompt", type=str, required=True)
    p_sample.add_argument("--out", type=str, default="out.png")
    p_sample.add_argument("--steps", type=int, default=25,
                         help="ODE steps (10‑40 is typical)")
    p_sample.add_argument("--sampler", choices=["euler", "rk38"], default="euler")

    args = p.parse_args()

    kwargs = {
        'entity': 'utah-math-data-science', 
        'project': 'Flow_Matching_Text2Image',
        'mode': args.wandb,
        'name': 'Text2Image_MFlow_{}_{}_lr{}_frzn{}_teach{}_TSample{}_flowR{}_gamma{}_txtReg{}'.format(args.run_name, args.noise_scale, 
                                                                                args.lr, args.frozen_text_proj, args.teacher, 
                                                                                args.t_sample, args.flow_ratio, args.gamma, args.txt_reg),
        'config': args,
        'settings': wandb.Settings(_disable_stats=True), 'reinit': True
        }
    wandb.init(**kwargs)
    wandb.save('*.txt')

    if args.cmd == "train":
        train(args)
