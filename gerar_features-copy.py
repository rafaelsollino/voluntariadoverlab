import os
import av
import torch
import pandas as pd
from tqdm import tqdm
from transformers import VivitImageProcessor, VivitModel

##############################################
# CONFIGURAÇÕES
##############################################

# Caminho onde está o CSV
CSV_DIR = "/datasets/"

# Caminho onde estão os vídeos
VIDEO_DIR = "/datasets/OESPUB/"

# Localizar o CSV automaticamente dentro de /my_datasets
def localizar_csv(path):
    for f in os.listdir(path):
        if f.endswith(".csv") or f.endswith(".tsv"):
            return os.path.join(path, f)
    raise FileNotFoundError("Nenhum .csv/.tsv encontrado em /my_datasets/. "
                            "Coloque o CSV lá ou especifique manualmente.")

META_PATH = localizar_csv(CSV_DIR)
print("Usando metadata:", META_PATH)

##############################################
# CARREGAR MODELO VIVIT
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

def extrair_features(video_path):
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

    return torch.save(outputs.last_hidden_state.squeeze(0).cpu()[:, :1024], output_path)

##############################################
# LOOP PRINCIPAL
##############################################

df = pd.read_csv(META_PATH)

if "film_id" not in df.columns or "clip_id" not in df.columns:
    raise RuntimeError("CSV precisa ter colunas film_id e clip_id.")

print(f"Processando {len(df)} entradas…\n")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    base_name = f"{row['film_id']}_{row['clip_id']}"
    video_path = os.path.join(VIDEO_DIR, base_name + ".mp4")
    output_path = os.path.join(VIDEO_DIR, base_name + ".pt")

    # Se já existe, pula
    if os.path.exists(output_path):
        continue

    if not os.path.exists(video_path):
        print("Vídeo ausente:", video_path)
        continue

    try:
        feats = extrair_features(video_path)
        torch.save(feats, output_path)
    except Exception as e:
        print(f"[ERRO] Falhou para {video_path}: {e}")
        continue

print("\n✔ FINALIZADO: arquivos .pt gerados!")
