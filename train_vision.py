from argparse import Namespace, ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader

import torchvision.transforms as T

from src import ConvVAE, VisionTrainer, VisionDataset, VisionLoss


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Variational Autoencoder Training-config")
    parser.add_argument("--train_set", type=str, default="Vision-CarRacing-v3-Train")
    parser.add_argument("--val_set", type=str, default="Vision-CarRacing-v3-Val")

    parser.add_argument("--n_channels", type=int, default=3)                # RGB image channels
    parser.add_argument("--z_dim", type=int, default=32)
    
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--kl_weight", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=0)

    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--verbose", type=bool, default=True)

    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_datasets(args: Namespace) -> None:
    aug_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.05,
            hue=0.02,
        ),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    ])
    
    train_set = VisionDataset(args.train_set, aug_transform)
    val_set = VisionDataset(args.val_set)
    return train_set, val_set


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    vae = ConvVAE(args.n_channels, args.z_dim)
    criterion = VisionLoss(args.kl_weight)
    optimizer = torch.optim.Adam(vae.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    
    train_set, val_set = load_datasets(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    
    if args.verbose:
        print(f"|Train| = {len(train_set)}, |Val| = {len(val_set)}")

    vae_trainer = VisionTrainer(vae, criterion, optimizer, args.epochs, args.device, args.eval_every)
    vae_trainer.train(train_loader, val_loader)    


if __name__ == "__main__":
    main()