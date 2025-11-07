import argparse
import logging
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from model_tcn_esp32 import TCNAutoencoderLite  
from audio_dataset import SimpleAudioDataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


def compute_esp32_loss(model, data, labels, config, center=None):
    if data.dim() == 4:
        data = data.squeeze(1)

    latent, recon = model(data)
    recon_loss = F.mse_loss(recon, data)

    if center is not None:
        pooled = torch.mean(latent, dim=2)
        distances = torch.sum((pooled - center.unsqueeze(0)) ** 2, dim=1)
        normal_mask = labels == 0
        anomaly_mask = labels == 1

        loss_normal = distances[normal_mask].mean() if normal_mask.any() else 0.0
        loss_anomaly = F.relu(config['margin'] - distances[anomaly_mask]).mean() if anomaly_mask.any() else 0.0
        svdd_loss = loss_normal + config['oe_weight'] * loss_anomaly
    else:
        svdd_loss = torch.tensor(0.0).to(data.device)

    total_loss = svdd_loss + config['score_weight'] * recon_loss


    return {
        'total': total_loss,
        'svdd': svdd_loss,
        'score': recon_loss,
        'latent': torch.mean(latent, dim=2)
    }


def train_epoch(model, train_loader, optimizer, config, center, device):
    model.train()
    losses = {'total': 0, 'svdd': 0, 'score': 0}
    all_latents = []

    for batch_data in tqdm(train_loader, desc='Training'):
        data, _, labels = batch_data
        data, labels = data.to(device), labels.to(device)

        if data.dim() == 4:
            data = data.squeeze(1)

        optimizer.zero_grad()
        loss_dict = compute_esp32_loss(model, data, labels, config, center)
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k in losses:
            losses[k] += loss_dict[k].item()

        if labels.sum() < len(labels):  # Has normal samples
            normal_mask = labels == 0
            all_latents.append(loss_dict['latent'][normal_mask].detach().cpu())

    for k in losses:
        losses[k] /= len(train_loader)

    if all_latents and center is not None:
        new_center = torch.cat(all_latents, dim=0).mean(dim=0)
        center = config['center_momentum'] * center + (1 - config['center_momentum']) * new_center.to(device)

    return losses, center


def evaluate_esp32(model, test_loader, center, device):
    model.eval()
    all_scores, all_labels, all_distances = [], [], []

    with torch.no_grad():
        for data, _, labels in tqdm(test_loader, desc='Evaluating'):
            data = data.to(device)
            if data.dim() == 4:
                data = data.squeeze(1)

            latent, recon = model(data)
            recon_error = F.mse_loss(recon, data, reduction='none').mean(dim=[1, 2])
            all_scores.extend(recon_error.cpu().numpy())
            all_labels.extend(labels.numpy())

            if center is not None:
                pooled = torch.mean(latent, dim=2)
                distances = torch.sum((pooled - center.unsqueeze(0)) ** 2, dim=1)
                all_distances.extend(distances.cpu().numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    results = {'auc': 0.0}
    if len(np.unique(all_labels)) > 1:
        results['auc'] = roc_auc_score(all_labels, all_scores)
        if center is not None:
            results['auc_svdd'] = roc_auc_score(all_labels, all_distances)

    return results


def profile_esp32_performance(model, test_loader, device):
    model.eval()
    data, _, _ = next(iter(test_loader))
    data = data.to(device)
    if data.dim() == 4:
        data = data.squeeze(1)

    for _ in range(10):
        with torch.no_grad():
            _ = model(data[:1])

    times = []
    for _ in range(100):
        start = time.time()
        with torch.no_grad():
            _ = model(data[:1])
        times.append((time.time() - start) * 1000)

    avg = np.mean(times)
    print(f"\nESP32 Inference Profile:")
    print(f"  Avg latency: {avg:.2f} ms (ESP32 est: {avg*10:.2f} ms)")
    print(f"  Model size (est): {sum(p.numel() for p in model.parameters()) * 4 / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./esp32_models')
    parser.add_argument('--machine_type', type=str, default='aircondition')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()

    save_path = Path(args.save_dir) / args.machine_type
    save_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ESP32-TCN")

    esp32_config = {
        'input_dim': 16,
        'latent_dim': 8,
        'window_size': 128
    }

    train_config = {
        'margin': 2.0,
        'oe_weight': 1.0,
        'score_weight': 0.5,
        'center_momentum': 0.9
    }

    logger.info("Creating ESP32 TCN model...")
    model = TCNAutoencoderLite(
        input_dim=esp32_config['input_dim'],
        latent_dim=esp32_config['latent_dim'],
        encoder_hidden=[8, 12],
        decoder_hidden=[12],
        kernel_size=3
    ).to(args.device)

    audio_config = {
        'sample_rate': 16000,
        'n_fft': 512,
        'hop_length': 256,
        'num_mel_bins': esp32_config['input_dim'],
        'target_length': esp32_config['window_size'],
        'norm_mean': -5.0,
        'norm_std': 4.5
    }

    train_dataset = SimpleAudioDataset(args.data_dir, 'train', audio_config)
    test_dataset = SimpleAudioDataset(args.data_dir, 'test', audio_config)

    if len(train_dataset) == 0:
        logger.error("No training files found.")
        return
    if len(test_dataset) == 0:
        logger.error("No test files found.")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Init SVDD center
    logger.info("Initializing SVDD center...")
    center = None
    latents = []
    with torch.no_grad():
        for i, (data, _, labels) in enumerate(train_loader):
            if i >= 10:
                break
            data, labels = data.to(args.device), labels.to(args.device)
            if data.dim() == 4:
                data = data.squeeze(1)
            normal = data[labels == 0]
            if len(normal) > 0:
                latent, _ = model(normal)
                pooled = torch.mean(latent, dim=2)
                latents.append(pooled.cpu())
    if latents:
        center = torch.cat(latents, dim=0).mean(dim=0).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        losses, center = train_epoch(model, train_loader, optimizer, train_config, center, args.device)
        logger.info(f"Train Loss: {losses['total']:.4f} (SVDD: {losses['svdd']:.4f}, Recon: {losses['score']:.4f})")

        results = evaluate_esp32(model, test_loader, center, args.device)
        logger.info(f"Eval AUC: {results.get('auc', 0):.4f} | SVDD AUC: {results.get('auc_svdd', 0):.4f}")

        if results.get('auc', 0) > best_auc:
            best_auc = results['auc']
            torch.save({
                'model_state_dict': model.state_dict(),
                'center': center.cpu(),
                'results': results,
                'config': esp32_config
            }, save_path / 'best_model.pth')
            logger.info("Saved best model.")

        scheduler.step()

    if args.profile:
        profile_esp32_performance(model, test_loader, args.device)

    logger.info(f"\nTraining complete. Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()