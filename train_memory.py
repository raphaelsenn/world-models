from argparse import Namespace, ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src import Memory, MemoryDataset, MemoryTrainer


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Training configuration for the memory component")

    parser.add_argument("--memory_dataset", type=str, default="MDN-RNN-Dataset-CarRacing-v3")

    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_mixtures", type=int, default=5)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_workers", type=str, default=0)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--no-verbose", action="store_false", dest="verbose")
    parser.set_defaults(verbose=True)

    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_dataloaders(args: Namespace) -> tuple[DataLoader, ...]:
    mem_dataset = MemoryDataset(args.memory_dataset)
    mem_len = len(mem_dataset)

    train_len = int(0.95 * mem_len)
    val_len = mem_len - train_len

    if args.verbose:
        print(f"|Train| = {train_len}, |Val| = {val_len}")

    generator = torch.Generator().manual_seed(args.seed)
    trainset, val_set = random_split(
        mem_dataset,
        [train_len, val_len],
        generator=generator,
    )

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=args.pin_memory,
    )
    
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=args.pin_memory,
    )
    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    memory = Memory(args.z_dim, args.action_dim, args.hidden_dim, args.n_mixtures)

    train_loader, val_loader = create_dataloaders(args)

    mem_trainer = MemoryTrainer(
        model=memory,
        learning_rate=args.learning_rate, 
        epochs=args.epochs,
        device=args.device,
        eval_every=args.eval_every,
        save_every=args.save_every,
        verbose=args.verbose,
    )
    mem_trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()