import av
import numpy as np
from transformers import VivitImageProcessor, VivitModel
model_name = "google/vivit-b-16x2-kinetics400"
processor = VivitImageProcessor.from_pretrained(model_name)
vivit = VivitModel.from_pretrained(model_name)
def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, seg_len):
    end_idx = seg_len
    start_idx = 0
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def get_video_features(video_path):
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = stream.average_rate
    frames_30_sec = min(int(150 * fps), stream.frames)
    indices = sample_frame_indices(clip_len=32, seg_len = frames_30_sec)
    video = read_video_pyav(container=container, indices=indices)
    inputs = processor(list(video), return_tensors="pt")

    outputs = vivit(**inputs)

    last_hidden_states = outputs.last_hidden_state

    return last_hidden_states.squeeze(0)
import torch
import pandas as pd
from tqdm import tqdm
data = pd.read_csv("/ossl-v1/OESPUB/updated_meta.csv")
path = "/ossl-v1/OESPUB"
count = 0
for idx in tqdm(range(len(data))):
    try:
        item = data.iloc[idx]
        file_name = item['film_id'] + "_" + str(item['clip_id'])
        video_features = get_video_features(path + file_name + ".mp4")
        torch.save(video_features, path + file_name + ".pt")
    except:
        count +=1
