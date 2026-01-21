import os
import math
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_scheduler
from tqdm import tqdm
from audiocraft.dataset import SoundtrackDataset
from audiocraft.models import VideoMusicGen
from audiocraft.base_models import MusicGen
from audiocraft.models.loaders import load_compression_model, load_lm_model

def dist_is_on():
    return dist.is_available() and dist.is_initialized()

def rank():
    return dist.get_rank() if dist_is_on() else 0

def world():
    return dist.get_world_size() if dist_is_on() else 1

def is_main():
    return rank() == 0

def allreduce_mean_scalar(x: float, device: torch.device) -> float:
    t = torch.tensor([x], device=device, dtype=torch.float32)
    if dist_is_on():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world()
    return float(t.item())

<<<<<<< HEAD
=======

# Patch para ViViT 768 -> 1024
# (resolve o bug 3137x1536 vs 1024x1024)

>>>>>>> 4d64b92d42718d846bb4a2b8efcf4388c822cc7a
def patch_video_projection_to_1024(lm_model: nn.Module, device: torch.device) -> int:

    if not hasattr(lm_model, "transformer"):
        raise RuntimeError("LM não tem atributo .transformer; não consigo aplicar patch.")

    tr = lm_model.transformer
    if not hasattr(tr, "layers"):
        raise RuntimeError("Transformer não tem .layers; não consigo aplicar patch.")

    patched = 0
    for i, layer in enumerate(tr.layers):
        if hasattr(layer, "video_projection_layer"):
            vpl = layer.video_projection_layer
            in_dim = getattr(vpl, "in_features", 768)
            out_dim = getattr(vpl, "out_features", None)

            if out_dim != 1024:
                new_vpl = nn.Linear(in_dim, 1024, bias=(getattr(vpl, "bias", None) is not None))
                new_vpl = new_vpl.to(device).float()
                layer.video_projection_layer = new_vpl
                patched += 1

                if is_main():
                    print(f"[PATCH] layer {i}: video_projection_layer {in_dim}->{out_dim}  ==>  {in_dim}->1024")

    if is_main():
        print(f"[PATCH] total patched = {patched}")
    return patched

class Trainer:
    def __init__(
        self,
        dataset_path: str,
        output_dir: str,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        grad_acc: int = 8,
        checkpoint_interval: int = 1,
        num_workers: int = 0,
        master_addr: str = "localhost",
        master_port: str = "12355",
        max_patience: int = 3,
    ):
        self.dataset_path = dataset_path
        self.output_dir = Path("/datasets/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = num_epochs
        self.grad_acc = max(1, grad_acc)
        self.ckpt_every = checkpoint_interval
        self.num_workers = num_workers
        self.master_addr = master_addr
        self.master_port = master_port
        self.max_patience = max_patience

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.criterion = nn.CrossEntropyLoss()

    def setup(self, local_rank: int, world_size: int):
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = self.master_port

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    def cleanup(self):
        if dist_is_on():
            dist.destroy_process_group()

    def wav_to_codes(self, wav: torch.Tensor, model: VideoMusicGen, device: torch.device) -> torch.Tensor:
        """
        item['audio'] no seu dataset veio como [1, 960000].
        Aqui aceitamos [T], [1,T], ou [B,T] (mas no loop usamos por sample).
        Retorna codes como encode devolve: tipicamente [1, Q, T'].
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1, T]
        # -> [1, 1, T]
        wav = wav.to(device, non_blocking=True).unsqueeze(1)

        with torch.no_grad():
            codes, scale = model.compression_model.encode(wav)

        return codes

    def build_model(self, device: torch.device) -> VideoMusicGen:
        base = MusicGen.get_pretrained("facebook/musicgen-medium")
        lm = load_lm_model(base.lm.state_dict(), "facebook/musicgen-medium", device="cuda")
        compression = load_compression_model("facebook/musicgen-medium", device="cuda")

        model = VideoMusicGen(
            "medium",
            compression_model=compression,
            lm=lm,
            max_duration=30,
        )
        
        model.lm.to(device)
        model.lm.float()
       
        #patch_video_projection_to_1024(model.lm, device)

        model.lm.requires_grad_(False)
        for n, p in model.lm.named_parameters():
            if "video" in n: 
                p.requires_grad = True

        return model

    def make_loaders(self, local_rank: int, world_size: int):
        train_ds = SoundtrackDataset(path=self.dataset_path, train_or_valid="train")
        valid_ds = SoundtrackDataset(path=self.dataset_path, train_or_valid="valid")

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
        valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=local_rank, shuffle=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, valid_loader, train_sampler, valid_sampler

    def compute_loss_from_output(self, logits, mask, codes) -> torch.Tensor:

        if logits.dim() != 4:
            raise RuntimeError(f"logits shape inesperado: {tuple(logits.shape)} (esperado [B,Q,T,C])")
        if logits.size(-1) != 2048:
            raise RuntimeError(f"logits C inesperado: {logits.size(-1)} (esperado 2048)")

        if mask.dim() != 3:
            raise RuntimeError(f"mask shape inesperado: {tuple(mask.shape)} (esperado [B,Q,T])")
        mask = mask.bool()

        if codes.dim() != 3:
            raise RuntimeError(f"codes shape inesperado: {tuple(codes.shape)} (esperado [B,Q,T])")
        codes = codes.long()

        masked_logits = logits[mask]  
        masked_targets = codes[mask]   

        if masked_logits.numel() == 0:

            return masked_logits.sum() * 0.0

        return self.criterion(masked_logits, masked_targets)

    def run_one_epoch_train(self, model: VideoMusicGen, ddp_lm: DDP, loader: DataLoader, sampler: DistributedSampler,
                            optimizer, scheduler, scaler, device: torch.device, epoch: int):
        sampler.set_epoch(epoch)
        ddp_lm.train()
        optimizer.zero_grad(set_to_none=True)

        step_loss_acc = 0.0
        micro_count = 0
        global_steps = 0

        pbar = tqdm(loader, disable=not is_main(), desc="Train", total=len(loader))
        for batch_idx, item in enumerate(pbar):
            prompts = item["prompt"]
            audios = item["audio"]
            videos = item["video"]

            if torch.is_tensor(videos):
                videos = videos.to(device, non_blocking=True).float()
            else:
                raise RuntimeError(f"item['video'] não é tensor: {type(videos)}")

            codes_list = []
            for i in range(len(prompts)):
                wav_i = audios[i]
                codes_i = self.wav_to_codes(wav_i, model, device)  # [1, Q, T]
                codes_list.append(codes_i)

            codes = torch.cat(codes_list, dim=0)  # [B, Q, T]

            attrs, _ = model._prepare_tokens_and_attributes(prompts, None)
            tokenized = ddp_lm.module.condition_provider.tokenize(attrs)
            condition_tensors = ddp_lm.module.condition_provider(tokenized)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = ddp_lm.module.compute_predictions(
                    codes=codes,
                    video_features=videos,
                    conditions=[],
                    condition_tensors=condition_tensors,
                )
                loss = self.compute_loss_from_output(out.logits, out.mask, codes)


            loss_scaled = loss / self.grad_acc
            scaler.scale(loss_scaled).backward()

            step_loss_acc += float(loss.detach().item())
            micro_count += 1

            if (batch_idx + 1) % self.grad_acc == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_lm.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                global_steps += 1

                if is_main():
                    avg = step_loss_acc / max(1, micro_count)
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                step_loss_acc = 0.0
                micro_count = 0

        if micro_count > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddp_lm.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        return

    @torch.no_grad()
    def run_validation(self, model: VideoMusicGen, ddp_lm: DDP, loader: DataLoader, sampler: DistributedSampler, device: torch.device, epoch: int) -> float:
        ddp_lm.eval()
        sampler.set_epoch(epoch)

        losses = []
        for item in tqdm(loader, disable=not is_main(), desc="Valid", total=len(loader)):
            prompts = item["prompt"]
            audios = item["audio"]
            videos = item["video"]

            if torch.is_tensor(videos):
                videos = videos.to(device, non_blocking=True).float()
            else:
                raise RuntimeError(f"[VAL] item['video'] não é tensor: {type(videos)}")

            codes_list = []
            for i in range(len(prompts)):
                wav_i = audios[i]
                codes_i = self.wav_to_codes(wav_i, model, device)
                codes_list.append(codes_i)

            codes = torch.cat(codes_list, dim=0)  # [B,Q,T]

            attrs, _ = model._prepare_tokens_and_attributes(prompts, None)
            tokenized = ddp_lm.module.condition_provider.tokenize(attrs)
            condition_tensors = ddp_lm.module.condition_provider(tokenized)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = ddp_lm.module.compute_predictions(
                    codes=codes,
                    video_features=videos,
                    conditions=[],
                    condition_tensors=condition_tensors,
                )
                loss = self.compute_loss_from_output(out.logits, out.mask, codes)

            losses.append(float(loss.detach().item()))

        local_avg = sum(losses) / max(1, len(losses))
        global_avg = allreduce_mean_scalar(local_avg, device)
        return global_avg

    def save_checkpoint(self, ddp_lm: DDP, optimizer, scheduler, epoch: int, val_loss: float):
        if not is_main():
            return
        ckpt = {
            "epoch": epoch,
            "model_state_dict": ddp_lm.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
        }
        path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(ckpt, path)
        print(f"[CKPT] saved: {path}")

    def save_best(self, ddp_lm: DDP, val_loss: float):
        if not is_main():
            return
        path = self.output_dir / "best.pt"
        torch.save({"model_state_dict": ddp_lm.module.state_dict(), "val_loss": val_loss}, path)
        print(f"[BEST] saved: {path} (val_loss={val_loss:.4f})")

    def train_process(self, local_rank: int, world_size: int):
        self.setup(local_rank, world_size)
        device = torch.device(f"cuda:{local_rank}")

        # modelo
        model = self.build_model(device)

        # DDP no LM
        ddp_lm = DDP(model.lm, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        # dados
        train_loader, valid_loader, train_sampler, valid_sampler = self.make_loaders(local_rank, world_size)

        # optimizer só com params treináveis
        params = [p for p in ddp_lm.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError("Nenhum parâmetro treinável (verifique o filtro 'video' nos nomes).")
        optimizer = torch.optim.AdamW(params, lr=self.lr)

        # scheduler
        steps_per_epoch = math.ceil(len(train_loader) / self.grad_acc)
        total_steps = max(1, self.epochs * steps_per_epoch)
        warmup = min(100, max(10, total_steps // 20))
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
        )

        scaler = torch.cuda.amp.GradScaler()

        best = float("inf")
        patience = 0

        if is_main():
            print(f"GPUs={world_size} | steps/epoch={steps_per_epoch} | total_steps={total_steps} | warmup={warmup}")
            print(f"Training params: {sum(p.numel() for p in params):,}")

        for epoch in range(1, self.epochs + 1):
            self.run_one_epoch_train(model, ddp_lm, train_loader, train_sampler, optimizer, scheduler, scaler, device, epoch)
            val_loss = self.run_validation(model, ddp_lm, valid_loader, valid_sampler, device, epoch)

            if is_main():
                print(f"[EPOCH {epoch}] val_loss = {val_loss:.4f}")

            # ckpt periódico
            if epoch % self.ckpt_every == 0:
                self.save_checkpoint(ddp_lm, optimizer, scheduler, epoch, val_loss)

            # early stopping
            if val_loss < best:
               best = val_loss
               patience = 0
               self.save_best(ddp_lm, best)
            else:
               patience += 1
               if is_main():
                 print(f"No improvement. patience={patience}/{self.max_patience}")
#
            stop = torch.tensor([1 if patience >= self.max_patience else 0], device=device, dtype=torch.int32)
            if dist_is_on():
                dist.broadcast(stop, src=0)
            if int(stop.item()) == 1:
                if is_main():
                    print(f"Early stopping. Best val_loss = {best:.4f}")
                break

        self.cleanup()

    def train(self):
        world_size = torch.cuda.device_count()
        if world_size < 1:
            raise RuntimeError("Nenhuma GPU detectada.")

        if world_size > 1:
            mp.spawn(self.train_process, args=(world_size,), nprocs=world_size, join=True)
        else:
            self.train_process(0, 1)


if __name__ == "__main__":
    trainer = Trainer(
        dataset_path="./OpenScreenSoundLibrary-v1/",
        output_dir="/datasets/output2",
        batch_size=1,
        learning_rate=1e-4,
        num_epochs=100,
        grad_acc=8,
        checkpoint_interval=1,
        num_workers=0,
        master_addr="localhost",
        master_port="12355",
        max_patience=3,
    )
    trainer.train()
