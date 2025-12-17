import torch
import torch.nn as nn

from audiocraft.dataset import SoundtrackDataset
from audiocraft.models import VideoMusicGen
from audiocraft.base_models import MusicGen
from audiocraft.models.loaders import load_compression_model, load_lm_model


# ===== Dataset =====
dataset = SoundtrackDataset(
    path="./OpenScreenSoundLibrary-v1/",
    train_or_valid="train"
)

item = dataset[0]

print(len(dataset))

print("\n=== DATASET ===")
print("prompt type:", type(item["prompt"]))
print("audio shape:", item["audio"].shape)
print("video type:", type(item["video"]))
if torch.is_tensor(item["video"]):
    print("video shape:", item["video"].shape, "dtype:", item["video"].dtype)
else:
    print("video value:", item["video"])


# ===== Modelo =====
device = torch.device("cuda")

model_name = "facebook/musicgen-small"
base = MusicGen.get_pretrained(model_name)
lm = load_lm_model(base.lm.state_dict(), model_name, device=device)
compression = load_compression_model(model_name, device=device)

model = VideoMusicGen(
    "small",
    compression_model=compression,
    lm=lm,
    max_duration=30,
)

# Importante: VideoMusicGen não tem .to(); mova só o que é Module
model.lm.to(device)
model.lm.float()
model.lm.eval()

print("\n=== MODEL ===")
print("LM type:", type(model.lm))
print("Has transformer:", hasattr(model.lm, "transformer"))


# ===== Patch: forçar projeção de vídeo para 1024 =====
def patch_video_projection_to_1024(lm_model):
    if not hasattr(lm_model, "transformer"):
        raise RuntimeError("lm_model não tem atributo .transformer; não consigo patchar as camadas.")

    tr = lm_model.transformer
    patched = 0
    inspected = 0

    # adapted_transformer normalmente tem .layers (lista de blocks)
    layers = getattr(tr, "layers", None)
    if layers is None:
        raise RuntimeError("transformer não tem .layers; não consigo patchar automaticamente.")

    for i, layer in enumerate(layers):
        if hasattr(layer, "video_projection_layer"):
            inspected += 1
            vpl = layer.video_projection_layer

            # tenta inferir in/out
            in_dim = getattr(vpl, "in_features", None)
            out_dim = getattr(vpl, "out_features", None)

            # se não for Linear padrão, pelo menos tenta manter in_dim=768
            if in_dim is None:
                in_dim = 768

            # se já é 1024, não mexe
            if out_dim == 1024:
                continue

            # substitui por Linear(in_dim -> 1024)
            new_vpl = nn.Linear(in_dim, 1024, bias=(getattr(vpl, "bias", None) is not None))
            new_vpl = new_vpl.to(device).float()

            layer.video_projection_layer = new_vpl
            patched += 1
            print(f"[PATCH] layer {i}: video_projection_layer {in_dim}->{out_dim}  ==>  {in_dim}->1024")

    print(f"[PATCH] inspected={inspected}, patched={patched}")
    return patched

patched = patch_video_projection_to_1024(model.lm)
if patched == 0:
    print("[PATCH] Nenhuma camada foi patchada. Isso é suspeito (ou já estava 1024).")


# ===== Audio -> codes =====
wav = item["audio"].to(device)
if wav.dim() == 1:
    wav = wav.unsqueeze(0)
wav = wav.unsqueeze(1)  # [1, 1, T]

with torch.no_grad():
    codes, scale = model.compression_model.encode(wav)

print("\n=== ENCODE AUDIO ===")
print("codes shape:", codes.shape)
print("codes dtype:", codes.dtype)
print("scale:", scale)


# ===== Video forward =====
video = item["video"]
if torch.is_tensor(video):
    video = video.float().unsqueeze(0).to(device)  # [1, Tv, 768]

print("\nVIDEO BEFORE:", video.shape, "dtype:", video.dtype, "device:", video.device)
print("CODES BEFORE:", codes.shape, "dtype:", codes.dtype, "device:", codes.device)

with torch.no_grad():
    attrs, _ = model._prepare_tokens_and_attributes([item["prompt"]], None)
    tokenized = model.lm.condition_provider.tokenize(attrs)
    cond_tensors = model.lm.condition_provider(tokenized)

    out = model.lm.compute_predictions(
        codes=codes,
        video_features=video,
        conditions=[],
        condition_tensors=cond_tensors,
    )

print("\n=== LM OUTPUT ===")
print("logits shape:", out.logits.shape)
print("mask shape:", out.mask.shape)
