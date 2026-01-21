import os
import re
import av
import torch
import pandas as pd
from tqdm import tqdm
from transformers import VivitImageProcessor, VivitModel

CSV_DIR = "/ossl-v1/OpenScreenSoundLibrary-v1/"
VIDEO_DIR = "/ossl-v1/OpenScreenSoundLibrary-v1/"

MODEL_NAME = "google/vivit-b-16x2-kinetics400"
CLIP_LEN = 32     
MAX_SECONDS = 150     

def carregar_metadata(csv_dir: str) -> pd.DataFrame:
    meta_all = os.path.join(csv_dir, "meta_all.csv")
    train_csv = os.path.join(csv_dir, "train_meta.csv")
    valid_csv = os.path.join(csv_dir, "valid_meta.csv")

    if os.path.exists(meta_all):
        df = pd.read_csv(meta_all)
        print("Usando metadata:", meta_all)
        return df

    if os.path.exists(train_csv) and os.path.exists(valid_csv):
        dft = pd.read_csv(train_csv)
        dfv = pd.read_csv(valid_csv)
        df = pd.concat([dft, dfv], ignore_index=True)
        print("Usando metadata:", f"{train_csv} + {valid_csv} (concatenados)")
        return df

    # Fallback: varrer arquivos .mp4 e extrair film_id/clip_id do nome "AA_0.mp4"
    pat = re.compile(r"^([A-Z]{2})_(\d+)\.mp4$")
    rows = []
    for f in os.listdir(VIDEO_DIR):
        m = pat.match(f)
        if m:
            rows.append({"film_id": m.group(1), "clip_id": int(m.group(2))})

    if not rows:
        raise FileNotFoundError("Não achei meta_all/train/valid e também não encontrei vídeos .mp4 no padrão AA_0.mp4.")

    df = pd.DataFrame(rows).sort_values(["film_id", "clip_id"]).reset_index(drop=True)
    print("Usando metadata (fallback): varredura de .mp4 na pasta")
    return df

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Carregando modelo ViViT:", MODEL_NAME)

processor = VivitImageProcessor.from_pretrained(MODEL_NAME)
vivit = VivitModel.from_pretrained(MODEL_NAME).to(device)
vivit.eval()

def sample_frame_indices(clip_len: int, seg_len: int):
    seg_len = max(seg_len, clip_len)  # garante seg_len >= clip_len
    seg_size = float(seg_len) / float(clip_len)
    idxs = [int(seg_size / 2.0 + seg_size * i) for i in range(clip_len)]
    # clamp
    idxs = [min(max(i, 0), seg_len - 1) for i in idxs]
    return idxs

def extrair_features(video_path: str) -> torch.Tensor:
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        fps = float(stream.base_rate) if stream.base_rate else 30.0

        total_frames = int(stream.frames) if stream.frames and stream.frames > 0 else int(MAX_SECONDS * fps)

        max_frames = min(int(MAX_SECONDS * fps), total_frames)
        indices = sample_frame_indices(CLIP_LEN, max_frames)

        frames = []
        for idx in indices:
  
            container.seek(int((idx / fps) * av.time_base))
            frame = next(container.decode(stream))
            frames.append(frame.to_rgb().to_ndarray())

        inputs = processor(frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = vivit(**inputs)

        feats = outputs.last_hidden_state.squeeze(0).cpu()
        return feats

    finally:
        container.close()

df = carregar_metadata(CSV_DIR)

if "film_id" not in df.columns or "clip_id" not in df.columns:
    raise RuntimeError("Metadata precisa ter colunas film_id e clip_id.")

df = df.drop_duplicates(subset=["film_id", "clip_id"]).reset_index(drop=True)

print(f"Processando {len(df)} entradas…\n")

falhas = 0
ausentes = 0
gerados = 0
pulados = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    base_name = f"{row['film_id']}_{int(row['clip_id'])}"
    video_path = os.path.join(VIDEO_DIR, base_name + ".mp4")
    output_path = os.path.join(VIDEO_DIR, base_name + ".pt")

    if os.path.exists(output_path):
        pulados += 1
        continue

    if not os.path.exists(video_path):
        ausentes += 1
        print("Vídeo ausente:", video_path)
        continue

    try:
        feats = extrair_features(video_path)
        torch.save(feats, output_path)
        gerados += 1

        # ajuda a não acumular memória
        del feats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        falhas += 1
        print(f"[ERRO] Falhou para {video_path}: {e}")
        continue

print("\n✔ FINALIZADO")
print(f"Gerados: {gerados} | Pulados (já existiam): {pulados} | Ausentes: {ausentes} | Falhas: {falhas}")
