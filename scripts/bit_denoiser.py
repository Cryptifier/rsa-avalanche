#!/usr/bin/env python3

import torch
import argparse
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_MODEL_NAME = "bit_denoiser_model.pt"

# -----------------------------
# Synthetic noisy-bitstring data
# -----------------------------


class NoisyBitstringDataset(Dataset):
    def __init__(
        self,
        num_samples=50_000,
        bit_len=128,
        num_copies=64,
        p_correct_min=0.52,
        p_correct_max=0.60,
    ):
        self.num_samples = num_samples
        self.bit_len = bit_len
        self.num_copies = num_copies
        self.p_correct_min = p_correct_min
        self.p_correct_max = p_correct_max

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Clean hidden bitstring x in {0,1}^n
        x = torch.randint(0, 2, (self.bit_len,), dtype=torch.float32)

        # Each noisy copy may have a different reliability
        p_correct = torch.empty(self.num_copies, 1).uniform_(
            self.p_correct_min,
            self.p_correct_max,
        )

        # Flip mask: 1 means bit is preserved, 0 means flipped
        keep = torch.bernoulli(p_correct.expand(self.num_copies, self.bit_len))

        # Noisy copies y^(k)
        y = torch.where(keep.bool(), x.unsqueeze(0), 1.0 - x.unsqueeze(0))

        # Shape:
        # y: [K, n]
        # x: [n]
        return y, x


# -----------------------------
# Attention + CNN denoiser
# -----------------------------


class BitDenoiser(nn.Module):
    def __init__(
        self,
        bit_len=128,
        num_copies=64,
        d_model=96,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()

        self.bit_len = bit_len
        self.num_copies = num_copies

        # Each observation is represented by:
        # bit value plus bit-position embedding
        self.bit_value_embed = nn.Linear(1, d_model)
        self.position_embed = nn.Embedding(bit_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.attn_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Collapse across noisy copies using learned attention pooling
        self.copy_score = nn.Linear(d_model, 1)

        # CNN refines local bit patterns after multi-copy fusion
        self.cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Per-bit classifier
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, y):
        """
        y shape: [B, K, n]
        output logits shape: [B, n]
        """

        B, K, n = y.shape
        assert n == self.bit_len

        # Reshape to [B*K, n, 1]
        y_flat = y.reshape(B * K, n, 1)

        positions = torch.arange(n, device=y.device)
        pos_emb = self.position_embed(positions)  # [n, d]
        pos_emb = pos_emb.unsqueeze(0).expand(B * K, n, -1)

        h = self.bit_value_embed(y_flat) + pos_emb  # [B*K, n, d]

        # Attention over bit positions inside each noisy copy
        h = self.attn_encoder(h)  # [B*K, n, d]

        # Restore copy dimension
        h = h.reshape(B, K, n, -1)  # [B, K, n, d]

        # Learned attention over copies for each bit position
        scores = self.copy_score(h).squeeze(-1)  # [B, K, n]
        weights = torch.softmax(scores, dim=1)  # [B, K, n]

        fused = (h * weights.unsqueeze(-1)).sum(dim=1)  # [B, n, d]

        # CNN expects [B, channels, n]
        z = fused.transpose(1, 2)  # [B, d, n]
        z = self.cnn(z)
        z = z.transpose(1, 2)  # [B, n, d]

        logits = self.out(z).squeeze(-1)  # [B, n]

        return logits


# -----------------------------
# Majority vote baseline
# -----------------------------


def majority_vote(y):
    """
    y shape: [B, K, n]
    returns hard bits [B, n]
    """
    return (y.mean(dim=1) > 0.5).float()


def bit_accuracy(pred_bits, target_bits):
    return (pred_bits == target_bits).float().mean().item()


def default_model_path():
    return Path.cwd() / DEFAULT_MODEL_NAME


def save_model(model, path, bit_len, num_copies):
    checkpoint = {
        "state_dict": model.state_dict(),
        "bit_len": bit_len,
        "num_copies": num_copies,
        "d_model": 48,
        "num_heads": 4,
        "num_layers": 2,
    }
    torch.save(checkpoint, path)
    print(f"saved model to {path}")


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = BitDenoiser(
        bit_len=checkpoint["bit_len"],
        num_copies=checkpoint["num_copies"],
        d_model=checkpoint["d_model"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


# -----------------------------
# Training
# -----------------------------


def train(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    bit_len = 64
    num_copies = 64

    train_ds = NoisyBitstringDataset(
        num_samples=80_000,
        bit_len=bit_len,
        num_copies=num_copies,
        p_correct_min=0.52,
        p_correct_max=0.60,
    )

    test_ds = NoisyBitstringDataset(
        num_samples=4_000,
        bit_len=bit_len,
        num_copies=num_copies,
        p_correct_min=0.52,
        p_correct_max=0.60,
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    model = BitDenoiser(
        bit_len=bit_len,
        num_copies=num_copies,
        d_model=48,  # Was 96
        num_heads=4,
        num_layers=2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0

        count = 0
        for y, x in train_loader:
            y = y.to(device)
            x = x.to(device)

            logits = model(y)

            loss = F.binary_cross_entropy_with_logits(logits, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if count % 1 == 0:
                print(f"count={count} loss={total_loss / (count + 1):.4f}")
            count += 1

        # Evaluation
        model.eval()
        neural_accs = []
        majority_accs = []

        with torch.no_grad():
            for y, x in test_loader:
                y = y.to(device)
                x = x.to(device)

                logits = model(y)
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()

                mv = majority_vote(y)

                neural_accs.append(bit_accuracy(pred, x))
                majority_accs.append(bit_accuracy(mv, x))

        print(
            f"epoch={epoch:02d} "
            f"loss={total_loss / len(train_loader):.4f} "
            f"neural_acc={sum(neural_accs) / len(neural_accs):.4f} "
            f"majority_acc={sum(majority_accs) / len(majority_accs):.4f}"
        )

    save_model(model, model_path, bit_len=bit_len, num_copies=num_copies)
    return model


def run_inference(model_path, num_samples):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_model(model_path, device)

    dataset = NoisyBitstringDataset(
        num_samples=num_samples,
        bit_len=checkpoint["bit_len"],
        num_copies=checkpoint["num_copies"],
        p_correct_min=0.52,
        p_correct_max=0.60,
    )

    print(f"loaded model from {model_path}")
    with torch.no_grad():
        for sample_idx in range(num_samples):
            y, x = dataset[sample_idx]
            y = y.unsqueeze(0).to(device)
            x = x.to(device)

            logits = model(y)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float().squeeze(0)
            mv = majority_vote(y).squeeze(0)

            neural_acc = bit_accuracy(pred, x)
            majority_acc = bit_accuracy(mv, x)

            observed = "".join(str(int(bit)) for bit in y[0, 0].cpu().tolist())
            predicted = "".join(str(int(bit)) for bit in pred.cpu().tolist())
            target = "".join(str(int(bit)) for bit in x.cpu().tolist())

            print(
                f"sample={sample_idx} "
                f"neural_acc={neural_acc:.4f} "
                f"majority_acc={majority_acc:.4f}"
            )
            print(f"  first_copy={observed}")
            print(f"  predicted={predicted}")
            print(f"  target={target}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or run sample inference with the bit denoiser.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("train", "infer"),
        default="train",
        help="Choose whether to train a model or run inference on sample data.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model_path(),
        help=(
            "Checkpoint path. Defaults to bit_denoiser_model.pt in the "
            "current working directory."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of synthetic samples to evaluate in inference mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args.model_path)
    else:
        run_inference(args.model_path, args.samples)
