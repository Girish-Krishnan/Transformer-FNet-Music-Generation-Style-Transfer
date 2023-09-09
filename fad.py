import numpy as np
from scipy.linalg import sqrtm
import warnings
from towhee import AutoPipes

def compute_embeddings_towhee(audio_files):
    embedding_pipeline = AutoPipes.pipeline('towhee/audio-embedding-vggish')

    all_embeddings = []

    for audio_file in audio_files:
        outs = embedding_pipeline(audio_file)
        emb = outs.get()[0]
        avg_emb = np.mean(emb, axis=0)  # Averaging over the time frames
        all_embeddings.append(avg_emb)

    return np.array(all_embeddings)


def compute_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Compute Frechet Distance between two multivariate Gaussians."""
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    fd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return fd

def calculate_fad(real_files, generated_files):
    # Assume this function gives embeddings for the given audio files
    real_embeddings = compute_embeddings_towhee(real_files)
    generated_embeddings = compute_embeddings_towhee(generated_files)

    mu_real = np.mean(real_embeddings, axis=0)
    sigma_real = np.cov(real_embeddings, rowvar=False)

    mu_gen = np.mean(generated_embeddings, axis=0)
    sigma_gen = np.cov(generated_embeddings, rowvar=False)

    fad = compute_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return fad



real_files = [f'./wav_data/{i}.wav' for i in range(20)]
generated_files = ["Track00002_generated.wav", "Track00001_generated.wav", "Track00003_generated.wav", "Track00004_generated.wav", "Track00005_generated.wav"]

fad_score = calculate_fad(real_files, generated_files)
print(f"FAD Score: {fad_score}")
