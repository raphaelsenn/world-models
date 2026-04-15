from argparse import Namespace, ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src import Memory, MemoryLoss, MemoryDataset, MemoryTrainer


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Memory Component Training-config")
    parser.add_argument("--memory_dataset", type=str, default="MDN-RNN-Dataset-CarRacing-v3")

    # Hyperparameter
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--action_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_mixtures", type=int, default=5)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.00025)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # CPU/GPU 
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)

    # Evaluation
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--verbose", type=bool, default=True)

    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    memory = Memory(args.z_dim, args.action_dim, args.hidden_dim, args.n_mixtures)
    criterion = MemoryLoss()
    optimizer = torch.optim.Adam(memory.parameters(), args.learning_rate)

    mem_dataset = MemoryDataset(args.memory_dataset)
    mem_len = len(mem_dataset)
    train_len = int(0.95 * mem_len)
    trainval_len = int(0.05 * mem_len)
    val_len = mem_len - train_len - trainval_len

    if args.verbose:
        print(f"|Train| = {train_len}, |Train-val| = {trainval_len}, |Val| = {val_len}")

    trainset, trainval_set, val_set = random_split(mem_dataset, [train_len, trainval_len, val_len])
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    trainval_loader = DataLoader(trainval_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    mem_trainer = MemoryTrainer(memory, criterion, optimizer, args.epochs, args.device, args.eval_every, args.save_every, args.verbose)
    mem_trainer.train(train_loader, trainval_loader, val_loader)    


if __name__ == "__main__":
    main()