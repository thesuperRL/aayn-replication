import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from layers.transformer import Transformer
from processing.dataloading import get_dataloaders


def train_epoch(model, loader, optimizer, device, pad_token_id):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        src = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        decoder_input = labels[:, :-1]
        targets = labels[:, 1:]

        optimizer.zero_grad()
        logits = model(src, decoder_input)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_token_id,
        )
        loss.backward()
        optimizer.step()

        num_tokens = (targets != pad_token_id).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, loader, device, pad_token_id):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        src = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        decoder_input = labels[:, :-1]
        targets = labels[:, 1:]

        logits = model(src, decoder_input)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_token_id,
        )

        num_tokens = (targets != pad_token_id).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer on Multi30k.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_loader, val_loader, test_loader, src_vocab_size, tgt_vocab_size, pad_token_id = get_dataloaders(batch_size=args.batch_size)
    print(f"Loaded datasets. src_vocab={src_vocab_size}, tgt_vocab={tgt_vocab_size}")

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        n_layers_encoder=args.n_layers,
        n_layers_decoder=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, pad_token_id
        )
        val_loss = evaluate(model, val_loader, device, pad_token_id)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                args.checkpoint_dir / "best.pt",
            )

    test_loss = evaluate(model, test_loader, device, pad_token_id)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Best checkpoint saved to {args.checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
