import numpy as np
import torch
import librosa
import json
from transformers import ClapModel, ClapProcessor
from tqdm import tqdm
from scipy.linalg import sqrtm
import torch.nn.functional as F
from hear21passt.base import get_model_passt
import glob
from prdc import compute_prdc

kl_model = get_model_passt(mode="logits")
kl_model.eval()
kl_model = kl_model.cuda()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

def extract_audio_embedding(audio_path):
    audio, sr = librosa.load(audio_path, sr=48000)
    inputs = processor(audios=audio, sampling_rate=sr, return_tensors="pt")

    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        embeddings = model.get_audio_features(**inputs)
    return embeddings.squeeze().cpu().numpy()

def calculate_audio_similarity(gt_embedding, model_embedding):
    score = np.dot(gt_embedding.flatten(), model_embedding.flatten()) / (
                np.linalg.norm(gt_embedding) * np.linalg.norm(model_embedding)
            )
    return float(score)



gen_path = "./generation/"
ground_truth_path = "./ground_truth/"
reference_path = "./reference/"

similarity_scores = {}
total_fad_list = []
kl_list = []
with torch.no_grad():
    for idx in tqdm(range(100)):
        gt_file = f"{ground_truth_path}{idx}.wav"
        model_file = f"{gen_path}{idx}.wav"

        gt_embedding = extract_audio_embedding(gt_file)
        model_embedding = extract_audio_embedding(model_file)
        random_num = np.random.randint(0, 32000 * 20) 
            
        gt_audio, _ = librosa.load(gt_file, sr = 32000)
        gen_audio, _ = librosa.load(model_file, sr = 32000)
        gt_audio = torch.from_numpy(gt_audio[random_num:random_num + 320000])
        gt_audio = gt_audio.unsqueeze(0) * 0.5
        gt_audio_wave = gt_audio.cuda()
        gt_logits=F.softmax(kl_model(gt_audio_wave), dim=1) 

        gen_audio = torch.from_numpy(gen_audio[random_num:random_num + 320000])
        gen_audio = gen_audio.unsqueeze(0) * 0.5
        gen_audio_wave = gen_audio.cuda()
        gen_logits=F.softmax(kl_model(gen_audio_wave), dim=1) 
        kl_div = F.kl_div(
                gen_logits.log(),
                gt_logits,
                reduction='batchmean'
            )
        sim = calculate_audio_similarity(gt_embedding, model_embedding) #CLAP Audio Similarity
        similarity_scores[idx] = sim
        total_fad_list.append(sim)
        kl_list.append(kl_div)
avg_sim = np.mean(total_fad_list)
kl = np.average(np.array(kl_list))


def get_audio_embeddings(audio_paths):

    batch_size = 8
    embeddings = []
    for i in tqdm(range(0, len(audio_paths), batch_size)):
        batch_paths = audio_paths[i:i + batch_size]
        batch_audios = []
        
        for audio_path in batch_paths:
            audio, sr = librosa.load(audio_path, sr=48000)
            batch_audios.append(audio)
        
        inputs = processor(audios=batch_audios, sampling_rate=sr, return_tensors="pt")
        
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embeddings.append(model.get_audio_features(**inputs))
            
    return torch.vstack(embeddings)

def calculate_fad(ref_embeds, test_embeds):
    
    mu_ref, sigma_ref = ref_embeds.mean(axis=0), np.cov(ref_embeds, rowvar=False)
    mu_test, sigma_test = test_embeds.mean(axis=0), np.cov(test_embeds, rowvar=False)

    mean_diff = np.sum((mu_ref - mu_test) ** 2)
    covmean, _ = sqrtm(sigma_ref @ sigma_test, disp=False)  # 공분산 행렬의 제곱근 계산
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad = mean_diff + np.trace(sigma_ref + sigma_test - 2 * covmean)
    return fad

ref_list = glob.glob(reference_path + "*.wav"),
gen_list = glob.glob(gen_path + "*.wav")
ref_embeddings = get_audio_embeddings(ref_list).cpu().numpy()
gen_embeddings = get_audio_embeddings(gen_list).cpu().numpy()
fad_score = calculate_fad(ref_embeddings, gen_embeddings)
metrics = compute_prdc(
                real_features=ref_embeddings,
                fake_features=gen_embeddings,
                nearest_k=5
)



results = {
    "average_clap": avg_sim,
    "sample_clap_scores": similarity_scores,
    "fad":fad_score,
    "kl":kl,
    "precision":metrics['precision'],
    "recall":metrics['recall']
}


with open("results.json", "w") as f:
    json.dump(results, f, indent=4)
