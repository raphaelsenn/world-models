from argparse import Namespace, ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split

from src import ConvVAE, VisionTrainer, VisionDataset, VisionLoss


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Variational Autoencoder Training-config")
    parser.add_argument("--vae_dataset", type=str, default="VAE-Dataset-CarRacing-v3")

    parser.add_argument("--n_channels", type=int, default=3)                # RGB image channels
    parser.add_argument("--z_dim", type=int, default=32)
    
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.00025)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--kl_weight", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=0)

    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--verbose", type=bool, default=True)

    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_datasets(args: Namespace) -> None:
    vae_dataset = VisionDataset(args.vae_dataset)
    vae_len = len(vae_dataset)
    
    train_len = int(0.98 * vae_len)
    trainval_len = int(0.01 * vae_len)
    val_len = vae_len - train_len - trainval_len

    train_set, trainval_set, val_set = random_split(vae_dataset, [train_len, trainval_len, val_len])
    train_set = ConcatDataset([train_set, trainval_set])
    return train_set, trainval_set, val_set


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    vae = ConvVAE(args.n_channels, args.z_dim)
    criterion = VisionLoss(args.kl_weight)
    optimizer = torch.optim.Adam(vae.parameters(), args.learning_rate)
    
    train_set, trainval_set, val_set = load_datasets(args)
    train_len, trainval_len, val_len = len(train_set), len(trainval_set), len(val_set) 
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    trainval_loader = DataLoader(trainval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    
    if args.verbose:
        print(f"|Train| = {train_len}, |Train-val| = {trainval_len}, |Val| = {val_len}")

    vae_trainer = VisionTrainer(vae, criterion, optimizer, args.epochs, args.device, args.eval_every)
    vae_trainer.train(train_loader, trainval_loader, val_loader)    


if __name__ == "__main__":
    main()