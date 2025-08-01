from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils as tvu
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator
from diffusers.models import AutoencoderKL
from models.model import FlowTokLite  # Assuming FlowTokLite is defined in model.py
from transformers import get_polynomial_decay_schedule_with_warmup, AutoTokenizer, AutoModel
import wandb
import time
from torchvision.utils import save_image
from sampler.ODEsolver import euler_sampler, rk38_sampler, dopri5_sampler
os.environ["TOKENIZERS_PARALLELISM"] = "false"

################################################################################
# Helper: make FM training targets
################################################################################
def make_targets(txt_tokens: torch.Tensor, img_tokens: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    z_t = t[:, None, None, None] * img_tokens + (1 - t)[:, None, None, None] * txt_tokens
    v = (img_tokens - txt_tokens)  # Δ = 1
    return z_t, v
    

################################################################################
# Training loop
################################################################################


def train(args):

    torch.manual_seed(123)
    torch.backends.cudnn.benchmark = True  # for faster training

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    # Config
    class CFG:
        n_layers = 3
        # d_model = 384
        # n_heads = 8
        d_model = 256
        n_heads = 4
        seq_len = 20
        img_size = args.img_size
        frozen_text_proj = args.frozen_text_proj
        unet = args.unet
    cfg = CFG()

    model = FlowTokLite(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0)


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
        power=1                  # 1.0 = linear, 2.0 = quadratic, etc.
    )
    

    #  model = torch.compile(model)
    model, optim, loader, scheduler = accelerator.prepare(model, optim, loader, scheduler)

    # load model from checkpoint if exists
    # rcvr_ckpt = '/home/sci/yuhao/Programme/FlowTok/flowtok_lite_100.pt'
    # if Path(rcvr_ckpt).exists():
    #     print(f"Loading model from {rcvr_ckpt}")
    #     sd = torch.load(rcvr_ckpt, map_location=device)
    #     model.load_state_dict(sd, strict=True)

    args.ckpt_out = args.ckpt_out + str(int(1000*args.noise_scale)) + '_unet' + str(args.unet) + '.pt'

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval()
    vae.to(accelerator.device)

    pre_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base")
    pre_model = AutoModel.from_pretrained("intfloat/e5-base")
    pre_model.to(accelerator.device)

    model.train()
    best_loss = float("inf")
    scale_ =  0.18215  # scale to match VAE training
    noise_scale = args.noise_scale

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_embeddings_std = 0.0
        epoch_tix_tok_std = 0.0
        start_time = time.time()

        for step, (imgs, captions) in enumerate(loader, start=1):
            t = torch.rand(imgs.size(0), device=device)

            with torch.no_grad():
                # img to latent
                img_tok = vae.encode(imgs).latent_dist.mode() * scale_  # scale to match VAE training
                img_tok = img_tok + 0.05 * torch.randn_like(img_tok)  # add noise to image tokens

                # text to latent
                tokens = pre_tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(device)
                output = pre_model(**tokens)
                embeddings = output.last_hidden_state[:, 0]  # [CLS]-like 

            txt_tok = model.text_to_latent(embeddings) 
            txt_tok = txt_tok + noise_scale * torch.randn_like(txt_tok)  # add noise to text tokens

            z_t, v = make_targets(txt_tok, img_tok, t)
            v_pred = model(z_t, t)
            loss = F.mse_loss(v_pred, v)
            accelerator.backward(loss)
            optim.step()
            scheduler.step() 
            optim.zero_grad()

            # record
            epoch_loss += loss.item()
            epoch_embeddings_std += embeddings.std().item()
            epoch_tix_tok_std += txt_tok.std().item()

        if epoch % 50 == 0 and epoch > 0:
            with torch.no_grad():
                img_tok_recon = rk38_sampler(model, txt_tok, 64)
                img_recon = vae.decode(img_tok_recon / scale_).sample
                img_recon_gd = vae.decode(img_tok / scale_).sample

            dir_path = '/home/sci/yuhao/Programme/FlowTok/' + 'imgs_{}_unet{}'.format(int(1000*args.noise_scale), args.unet)
            Path(dir_path).mkdir(parents=True, exist_ok=True)

            save_image(img_recon[:16], '{}/recon_batch_epoch_{}.png'.format(dir_path, epoch+args.add_epoch), nrow=8)
            save_image(img_recon_gd[:16], '{}/recon_batch_epoch_{}_gd.png'.format(dir_path, epoch+args.add_epoch), nrow=8)
            wandb.log({
                "recon_batch": wandb.Image('{}/recon_batch_epoch_{}.png'.format(dir_path, epoch+args.add_epoch)),
                "recon_batch_gd": wandb.Image('{}/recon_batch_epoch_{}_gd.png'.format(dir_path, epoch+args.add_epoch)),
            })

        epoch_loss /= step
        epoch_embeddings_std /= step
        epoch_tix_tok_std /= step

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
            "embeddings_std": epoch_embeddings_std,
            "txt_tok_std": epoch_tix_tok_std,
        })
        if best_loss > epoch_loss:
            best_loss = epoch_loss

            accelerator.wait_for_everyone()
            # Save the model checkpoint
            if accelerator.is_main_process:
                torch.save(model.state_dict(), args.ckpt_out)
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
    p_train.add_argument("--batch", type=int, default=32)
    p_train.add_argument("--epochs", type=int, default=2000)
    p_train.add_argument("--ckpt_out", type=str, default="flowtok_lite_")
    p_train.add_argument("--hf_token", type=str, default=None,
                         help="HF token for gated datasets (optional)")
    p_train.add_argument("--run_name", type=str, default="FlowTokLite",
                         help="W&B run name for tracking training progress")
    p_train.add_argument("--wandb", type=str, default='disabled',
                         help="Enable W&B logging (default: False)")
    p_train.add_argument("--noise_scale", type=float, default = 0.05)  # noise scale for text tokens
    p_train.add_argument("--frozen_text_proj", type=bool, default=False,
                         help="Use frozen text projection layer (default: False)")
    p_train.add_argument("--add_epoch", type=int, default=2000,
                         help="Number of epochs to add for continued training (default: None)")
    p_train.add_argument("--unet", type=bool, default=True,
                         help="Use UNet backbone instead of FlowTokBackbone (default: False)")
    args = p.parse_args()

    kwargs = {
        'entity': 'utah-math-data-science', 
        'project': 'Flow_Matching_Text2Image',
        'mode': args.wandb,
        'name': 'Text2Image_{}_{}_frzn{}'.format(args.run_name, args.noise_scale, args.frozen_text_proj),
        'config': args,
        'settings': wandb.Settings(_disable_stats=True), 'reinit': True
        }
    wandb.init(**kwargs)
    wandb.save('*.txt')

    # data_argmentation(args)

    if args.cmd == "train":
        train(args)
