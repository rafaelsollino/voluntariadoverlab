import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from audiocraft.dataset import SoundtrackDataset
from audiocraft.models import VideoMusicGen
from audiocraft.base_models import MusicGen
from audiocraft.models.loaders import load_compression_model, load_lm_model
from pathlib import Path
import torch.nn as nn
from transformers import get_scheduler
from tqdm import tqdm
criterion = nn.CrossEntropyLoss()
grad_acc = 8


class DistributedMusicGenTrainer:
    def __init__(
        self,
        dataset_path: str,
        output_dir: str,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        num_epochs: int = 200,
        gradient_accumulation_steps: int = 1,
        checkpoint_interval: int = 10,
    ):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_interval = checkpoint_interval
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_process(self, rank: int, world_size: int):
        """Initialize distributed training process group"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        torch.cuda.set_device(rank)
        torch.backends.cudnn.benchmark = True

    def preprocess_audio(self, wav, model):
        assert wav.shape[0] == 1

        wav = wav.cuda()
        wav = wav.unsqueeze(1)

        with torch.no_grad():
            gen_audio = model.compression_model.encode(wav)

        codes, scale = gen_audio

        assert scale is None

        return codes

    def one_hot_encode(self, codes, num_classes):
        return torch.nn.functional.one_hot(codes, num_classes).float()
    def cleanup_process(self):
        dist.destroy_process_group()

    def train_process(self, rank: int, world_size: int):
        self.setup_process(rank, world_size)
        
        base_model = MusicGen.get_pretrained('facebook/musicgen-small')
        lm_dict = base_model.lm.state_dict()

        lm = load_lm_model(lm_dict, "facebook/musicgen-small", device="cuda")
        
        compression_model = load_compression_model("facebook/musicgen-small", device="cuda")
        model = VideoMusicGen('small',
                           compression_model=compression_model,
                           lm=lm,
                           max_duration=30)
        model.lm.requires_grad_(False)

        for name, param in model.lm.named_parameters():
            if 'video' in name:
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                    param.data = param.data.half()
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
                    param.data = param.data.half()
        for name, param in model.lm.named_parameters():
            if 'video' in name:
                param.requires_grad = True
            if 'video' in name:
                param.requires_grad = True

        train_dataset = SoundtrackDataset(
            path=self.dataset_path,
            train_or_valid='train'
        )
        
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True
        )
        valid_dataset = SoundtrackDataset(
            path=self.dataset_path,
            train_or_valid='valid'
        )
        
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=0,
            pin_memory=True
        )

        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in model.lm.named_parameters() if 'video' in n], 'lr': 1e-4},
            {'params': [p for n, p in model.lm.named_parameters() if 'video' not in n], 'lr': 1e-5}
        ])

        warmup_steps = 10
        scheduler = get_scheduler(
        "cosine",
        optimizer,
        warmup_steps,
        int(self.num_epochs * len(train_dataloader) / grad_acc),
        )
        

        scaler = torch.cuda.amp.GradScaler()

        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(self.num_epochs):
            train_sampler.set_epoch(epoch)
            model.lm.float()
            model.lm.train()
            
            for batch_idx, item in tqdm(enumerate(train_dataloader)):
                prompts = item['prompt']
                audios = item['audio'].to(rank)
                videos = item['video'].to(rank)
                
                all_codes = []
                for prompt, audio, _ in zip(prompts, audios, videos):
                    codes = self.preprocess_audio(audio, model)
                    all_codes.append(codes)
                    
                if len(all_codes) == 0:
                    continue
                    
                attributes, _ = model._prepare_tokens_and_attributes(prompts, None)
                conditions = attributes
                tokenized = model.lm.condition_provider.tokenize(conditions)
                cfg_conditions = model.lm.condition_provider(tokenized)
                condition_tensors = cfg_conditions
                
                codes = torch.cat(all_codes, dim=0)
                
                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    lm_output = model.lm.compute_predictions(
                        codes=codes,
                        video_features=videos,
                        conditions=[],
                        condition_tensors=condition_tensors
                    )
                    
                    codes = codes[0]
                    logits = lm_output.logits[0]
                    mask = lm_output.mask[0]
                    
                    codes = self.one_hot_encode(codes, num_classes=2048)
                    codes = codes.cuda()
                    logits = logits.cuda()
                    mask = mask.cuda()
                    
                    mask = mask.view(-1)
                    masked_logits = logits.view(-1, 2048)[mask]
                    masked_codes = codes.view(-1, 2048)[mask]
                    
                    loss = criterion(masked_logits, masked_codes)
                    
                
                scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 1)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                if batch_idx % 100 == 0:
                    print(
                        f"Epoch: {epoch}/{self.num_epochs}, "
                        f"Batch: {batch_idx}/{len(train_dataloader)}, "
                        f"Loss: {loss.item():.4f}, "
                    )
            
            model.lm.eval()
            val_losses = []
            
            with torch.no_grad():
                for val_batch_idx, val_item in tqdm(enumerate(valid_dataloader), desc="Validation"):
                    val_prompts = val_item['prompt']
                    val_audios = val_item['audio'].to(rank)
                    val_videos = val_item['video'].to(rank)
                    
                    val_all_codes = []
                    for val_prompt, val_audio, val_video in zip(val_prompts, val_audios, val_videos):
                        val_codes = self.preprocess_audio(val_audio, model)
                        val_all_codes.append(val_codes)
                        
                    if len(val_all_codes) == 0:
                        continue
                        
                    val_attributes, _ = model._prepare_tokens_and_attributes(val_prompts, None)
                    val_conditions = val_attributes
                    val_tokenized = model.lm.condition_provider.tokenize(val_conditions)
                    val_cfg_conditions = model.lm.condition_provider(val_tokenized)
                    val_condition_tensors = val_cfg_conditions
                    
                    val_codes = torch.cat(val_all_codes, dim=0)
                    
                    with torch.autocast(device_type="cuda", dtype=torch.float32):
                        val_lm_output = model.lm.compute_predictions(
                            codes=val_codes,
                            video_features=val_videos,
                            conditions=[],
                            condition_tensors=val_condition_tensors
                        )
                        
                        val_codes = val_codes[0]
                        val_logits = val_lm_output.logits[0]
                        val_mask = val_lm_output.mask[0]
                        
                        val_codes = self.one_hot_encode(val_codes, num_classes=2048)
                        val_codes = val_codes.cuda()
                        val_logits = val_logits.cuda()
                        val_mask = val_mask.cuda()
                        
                        val_mask = val_mask.view(-1)
                        val_masked_logits = val_logits.view(-1, 2048)[val_mask]
                        val_masked_codes = val_codes.view(-1, 2048)[val_mask]
                        
                        val_loss = criterion(val_masked_logits, val_masked_codes)
                        val_losses.append(val_loss.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"Epoch {epoch}: Average Validation Loss = {avg_val_loss:.4f}")
            

            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.lm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, f'multi{epoch}.pt')
                print(f"Saved best model with validation loss: {avg_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Validation loss did not improve. Patience: {patience_counter}/3")
                
            if patience_counter >= 3:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"Best validation loss achieved: {best_val_loss:.4f}")
                exit()
            
    def train(self):
        """Main training function"""
        world_size = torch.cuda.device_count()
        
        if world_size > 1:
            mp.spawn(
                self.train_process,
                args=(world_size,),
                nprocs=world_size,
                join=True
            )
        else:
            self.train_process(0, 1)

def train_musicgen(
    dataset_path: str,
    output_dir: str,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    num_epochs: int = 200,
    gradient_accumulation_steps: int = 1,
    checkpoint_interval: int = 10,
):
    
    trainer = DistributedMusicGenTrainer(
        dataset_path=dataset_path,
        output_dir=output_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        checkpoint_interval=checkpoint_interval,
    )
    trainer.train()

if __name__ == "__main__":
    train_musicgen(
        dataset_path="./OpenScreenSoundLibrary-v1/",
        output_dir="./",
        batch_size=1, 
        learning_rate=1e-6, 
        num_epochs=100,
        gradient_accumulation_steps=2,
        checkpoint_interval=10,
    )