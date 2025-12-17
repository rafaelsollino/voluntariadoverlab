import argparse
import math
import torch
from scipy.io import wavfile
from audiocraft.models.loaders import load_compression_model, load_lm_model
from audiocraft.models import VideoMusicGen
from audiocraft.base_models import MusicGen
from audiocraft.dataset import OESCom
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_bip_bip(bip_duration=0.125, frequency=440, duration=0.5, sample_rate=32000, device="cuda"):
    """Generates a series of bip bip at the given frequency."""
    t = torch.arange(int(duration * sample_rate), device=device, dtype=torch.float) / sample_rate
    wav = torch.cos(2 * math.pi * frequency * t)[None]
    tp = (t % (2 * bip_duration)) / (2 * bip_duration)
    envelope = (tp >= 0.5).float()
    return wav * envelope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["s-base", "m-base", "s-text", "m-text", "s-multi", "m-multi"],
                        help="Specify which model configuration to use.")
    parser.add_argument("--checkpoint", type=str, default="model.pt",
                        help="Path to the checkpoint for the text or multi models.")
    args = parser.parse_args()

    model_name = args.model_name
    CHECKPOINT_PATH = args.checkpoint

    if model_name == 's-base':
        model = MusicGen.get_pretrained('small')
    elif model_name == 'm-base':
        model = MusicGen.get_pretrained('medium')
    elif model_name == 's-text':
        model = MusicGen.get_pretrained('small')
        model.lm.load_state_dict(torch.load(CHECKPOINT_PATH), strict=False)
    elif model_name == 'm-text':
        model = MusicGen.get_pretrained('medium')
        model.lm.load_state_dict(torch.load(CHECKPOINT_PATH), strict=False)
    elif model_name == 's-multi':
        base_model = MusicGen.get_pretrained("small")
        lm = load_lm_model(base_model.lm.state_dict(), "facebook/musicgen-small", device="cuda")
        compression_model = load_compression_model("facebook/musicgen-small", device="cuda")
        model = VideoMusicGen('small', compression_model=compression_model, lm=lm, max_duration=30)
        model.lm.load_state_dict(torch.load(CHECKPOINT_PATH), strict=False)
        model.compression_model.load_state_dict(base_model.compression_model.state_dict(), strict=False)
    elif model_name == 'm-multi':
        base_model = MusicGen.get_pretrained("medium")
        lm = load_lm_model(base_model.lm.state_dict(), "facebook/musicgen-medium", device="cuda")
        compression_model = load_compression_model("facebook/musicgen-medium", device="cuda")
        model = VideoMusicGen('medium', compression_model=compression_model, lm=lm, max_duration=30)
        model.lm.load_state_dict(torch.load(CHECKPOINT_PATH), strict=False)
        model.compression_model.load_state_dict(base_model.compression_model.state_dict(), strict=False)

    dataset = OESCom()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)

    for batch_idx, item in tqdm(enumerate(dataloader)):
        print(batch_idx, item['file_name'], item['prompt'])
        bip = get_bip_bip(0.125).expand(1, -1, -1)
        if model_name in ['s-base', 'm-base', 's-text', 'm-text']:
            res = model.generate_continuation(bip, 32000, item['prompt'], progress=True)
        else:
            res = model.generate_continuation(bip, 32000, item['prompt'], item['video'].to("cuda"), progress=True)
        wavfile.write(f"{batch_idx}.wav", 32000, res.cpu().numpy())


if __name__ == "__main__":
    main()
