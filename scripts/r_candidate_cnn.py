#!/usr/bin/env python3
import argparse
import json
import random
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN on r-candidate batch data and emit PCA clustering."
    )
    parser.add_argument("--session", required=True, help="Path to session JSON.")
    parser.add_argument("--config", help="Optional config JSON/JSON5 for polynomial primes.")
    parser.add_argument("--batch-index", type=int, default=0, help="Batch index to load.")
    parser.add_argument("--all-batches", action="store_true", help="Use all batches in the session.")
    parser.add_argument(
        "--primes",
        help="Comma-separated prime list (overrides config polynomial_fields).",
    )
    parser.add_argument("--poly-degree", type=int, default=3, help="Polynomial degree per prime.")
    parser.add_argument("--no-magnitude", action="store_true", help="Disable magnitude feature.")
    parser.add_argument("--fc-layers", default="128,64", help="Comma-separated FC layer sizes.")
    parser.add_argument("--embedding-dim", type=int, default=16, help="Embedding size.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda.")
    parser.add_argument("--output", default="pca_clusters.png", help="PCA output PNG.")
    return parser.parse_args()


def load_jsonish(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            import json5  # type: ignore
        except ImportError:
            raise RuntimeError("Config parse failed; install json5 or provide --primes.")
        return json5.loads(raw)


def parse_primes(args: argparse.Namespace) -> List[int]:
    if args.primes:
        primes = [int(p.strip()) for p in args.primes.split(",") if p.strip()]
        if primes:
            return primes
    if args.config:
        cfg = load_jsonish(args.config)
        fields = cfg.get("polynomial_fields", {}).get("fields", [])
        primes = [int(field["prime"]) for field in fields if "prime" in field]
        if primes:
            return primes
    raise RuntimeError("No primes provided; use --primes or --config with polynomial_fields.")


def collect_samples(session: dict, batch_index: int, all_batches: bool) -> Tuple[List[int], List[int], List[float]]:
    batches = session.get("r_candidate_accuracy_batches", [])
    if not batches:
        raise RuntimeError("Session JSON missing r_candidate_accuracy_batches.")

    if all_batches:
        selected_batches = batches
    else:
        if batch_index < 0 or batch_index >= len(batches):
            raise RuntimeError(f"batch_index {batch_index} out of range (0..{len(batches)-1}).")
        selected_batches = [batches[batch_index]]

    values: List[int] = []
    labels: List[int] = []
    accuracy: List[float] = []

    for batch in selected_batches:
        messages = batch.get("messages", [])
        candidates = batch.get("candidates", [])
        if not messages or not candidates:
            raise RuntimeError("Batch missing messages or candidates.")

        for cand_idx, cand in enumerate(candidates):
            decryptions = cand.get("candidate_decryptions", [])
            if len(decryptions) != len(messages):
                raise RuntimeError("candidate_decryptions length does not match messages.")
            accuracy.append(float(cand.get("accuracy_pct", 0.0)))
            for dec in decryptions:
                values.append(int(dec))
                labels.append(cand_idx)

    return values, labels, accuracy


def build_features(values: List[int], primes: List[int], degree: int, include_magnitude: bool) -> List[List[float]]:
    if degree < 1:
        raise RuntimeError("poly-degree must be >= 1")
    max_val = max(values) if values else 1
    max_val = max_val if max_val > 0 else 1

    features: List[List[float]] = []
    for x in values:
        row: List[float] = []
        for prime in primes:
            if prime <= 1:
                raise RuntimeError("Prime list contains invalid value")
            residue = x % prime
            for pow_idx in range(1, degree + 1):
                row.append(pow(residue, pow_idx, prime) / float(prime - 1))
        if include_magnitude:
            row.append(x / float(max_val))
        features.append(row)
    return features


def build_fc_layers(arg: str) -> List[int]:
    if not arg:
        return []
    values = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies. Install torch, numpy, scikit-learn, matplotlib."
        ) from exc

    session = load_jsonish(args.session)
    primes = parse_primes(args)
    values, labels, accuracy = collect_samples(session, args.batch_index, args.all_batches)

    features = build_features(values, primes, args.poly_degree, not args.no_magnitude)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    num_classes = int(max(labels) + 1) if labels else 0
    if num_classes <= 1:
        raise RuntimeError("Need at least two r candidates for classification.")

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    class ResidueCNN(nn.Module):
        def __init__(self, input_len: int, classes: int, fc_layers: List[int], embedding_dim: int):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool1d(2)
            self.adapt = nn.AdaptiveAvgPool1d(16)

            conv_out = 32 * 16
            fc_sizes = [conv_out] + fc_layers
            fc_blocks = []
            for idx in range(len(fc_sizes) - 1):
                fc_blocks.append(nn.Linear(fc_sizes[idx], fc_sizes[idx + 1]))
                fc_blocks.append(nn.ReLU())
            self.fc = nn.Sequential(*fc_blocks)
            final_dim = fc_layers[-1] if fc_layers else conv_out
            self.embedding = nn.Linear(final_dim, embedding_dim)
            self.classifier = nn.Linear(embedding_dim, classes)

        def forward(self, inputs):
            x_local = inputs.unsqueeze(1)
            x_local = self.pool(self.relu(self.conv1(x_local)))
            x_local = self.pool(self.relu(self.conv2(x_local)))
            x_local = self.adapt(x_local)
            x_local = x_local.view(x_local.size(0), -1)
            x_local = self.fc(x_local)
            emb = self.embedding(x_local)
            logits = self.classifier(emb)
            return logits, emb

    device = torch.device(args.device)
    model = ResidueCNN(x.shape[1], num_classes, build_fc_layers(args.fc_layers), args.embedding_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        avg_loss = total_loss / max(total, 1)
        acc = 100.0 * correct / max(total, 1)
        print(f"Epoch {epoch+1}/{args.epochs} - loss {avg_loss:.4f} - acc {acc:.2f}%")

    model.eval()
    with torch.no_grad():
        logits, embeddings = model(x.to(device))
        embeddings = embeddings.cpu().numpy()

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, cmap="tab20")
    plt.title("R-candidate PCA Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)

    if accuracy:
        mean_acc = sum(accuracy) / len(accuracy)
        print(f"Mean candidate accuracy: {mean_acc:.2f}%")
    print(f"Saved PCA plot to {args.output}")


if __name__ == "__main__":
    main()
