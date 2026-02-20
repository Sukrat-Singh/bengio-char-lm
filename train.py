import argparse
import math
import os
from tqdm import tqdm

import torch
import torch.nn as nn

from model import BengioLM
from utils import build_vocab_from_text, encode_text, build_dataset, make_loaders


def save_vocab(stoi, itos, path="vocab.pt"):
    torch.save({"stoi": stoi, "itos": itos}, path)


def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def train(args):
    text = load_text(args.data_path)
    stoi, itos = build_vocab_from_text(text)
    data_indices = encode_text(text, stoi)

    # split
    split = int(0.9 * len(data_indices))
    train_idx = data_indices[:split]
    val_idx = data_indices[split:]

    X_train, Y_train = build_dataset(train_idx, args.context_len)
    X_val, Y_val = build_dataset(val_idx, args.context_len)

    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val, batch_size=args.batch_size)

    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')

    model = BengioLM(vocab_size=len(stoi), context_len=args.context_len, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = float('inf')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_ppl = math.exp(avg_train_loss)

        # validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_ppl = math.exp(avg_val_loss)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.3f} | Train PPL: {train_ppl:.2f} | Val Loss: {avg_val_loss:.3f} | Val PPL: {val_ppl:.2f}")

        # save best
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            ckpt_path = os.path.join(args.checkpoint_dir, 'best.pt')
            torch.save({
                'model_state': model.state_dict(),
                'stoi': stoi,
                'itos': itos,
                'args': vars(args),
            }, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}")

    # save vocab for later
    save_vocab(stoi, itos, path=os.path.join(args.checkpoint_dir, 'vocab.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/input.txt')
    parser.add_argument('--context_len', type=int, default=10)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    train(args)