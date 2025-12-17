import os
import av
import torch
from tqdm import tqdm
from transformers import VivitImageProcessor, VivitModel

##############################################
# CONFIGURAÇÕES
##############################################

# Pasta do novo dataset (OESPUB) dentro do repo
VIDEO_DIR = os.path.abspath("./OESPUB")

if not os.path.isdir(VIDEO_DIR):
    raise RuntimeError(f"Diretório de vídeos não existe: {VIDEO_DIR}")

##############################################
# MODELO VIVIT
##############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "google/vivit-b-16x2-kinetics400"
print("Carregando modelo ViViT:", model_name)

processor = VivitImageProcessor.from_pretrained(model_name)
vivit = VivitModel.from_pretrained(model_name).to(device)
vivit.eval()

##############################################
# FUNÇÕES AUXILIARES
##############################################

def sample_frame_indices(clip_len, seg_len):
    seg_size = float(seg_len) / clip_len
    return [int(seg_size / 2.0 + seg_size * i) for i in range(clip_len)]

def extrair_features_video(video_path):
    container = av.open(video_path)
    stream = container.streams.video[0]

    fps = float(stream.base_rate)
    total_frames = stream.frames

    max_frames = min(int(150 * fps), total_frames)
    indices = sample_frame_indices(32, max_frames)

    frames = []
    for idx in indices:
        container.seek(int(idx / fps * av.time_base))
        frame = next(container.decode(stream))
        frames.append(frame.to_rgb().to_ndarray())

    inputs = processor(frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = vivit(**inputs)

    # [1, 32, 1536] -> [32, 1536]
    feats = outputs.last_hidden_state.squeeze(0).cpu()
    # corta de 1536 para 1024 dimensões (para bater com o modelo)
    feats = feats[:, :1024]   # [32, 1024]

    return feats

##############################################
# LOOP PRINCIPAL NOS .MP4
##############################################

video_files = sorted(f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(".mp4"))

if not video_files:
    raise RuntimeError(f"Nenhum .mp4 encontrado em {VIDEO_DIR}")

print(f"Encontrados {len(video_files)} vídeos em {VIDEO_DIR}. Gerando .pt...\n")

for fname in tqdm(video_files):
    base, _ = os.path.splitext(fname)
    video_path = os.path.join(VIDEO_DIR, fname)
    output_path = os.path.join(VIDEO_DIR, base + ".pt")

    # se já existe, pula
    if os.path.exists(output_path):
        continue

    try:
        feats = extrair_features_video(video_path)
        torch.save(feats, output_path)
    except Exception as e:
        print(f"\n[ERRO] Falha em {video_path}: {e}")
        continue

print("\n✔ Finalizado! Features salvas como *.pt dentro de OESPUB.")
