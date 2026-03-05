#!/usr/bin/env python3
import argparse
import json
import random
from typing import Dict, List, Tuple


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
    parser.add_argument(
        "--target-dim",
        type=int,
        default=0,
        help="Target vector size (defaults to embedding-dim when 0).",
    )
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


def collect_samples(
    session: dict, batch_index: int, all_batches: bool
) -> Tuple[List[int], List[int], List[float], List[int]]:
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
    message_ids: List[int] = []
    candidate_offset = 0
    message_offset = 0

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
            for msg_idx, dec in enumerate(decryptions):
                values.append(int(dec))
                labels.append(candidate_offset + cand_idx)
                message_ids.append(message_offset + msg_idx)

        candidate_offset += len(candidates)
        message_offset += len(messages)

    return values, labels, accuracy, message_ids


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


def encode_targets(values: List[int], target_dim: int) -> List[List[float]]:
    if target_dim <= 0:
        raise RuntimeError("target_dim must be >= 1")

    targets: List[List[float]] = []
    for value in values:
        if value == 0:
            targets.append([0.0] * target_dim)
            continue

        byte_len = max(1, (value.bit_length() + 7) // 8)
        data = value.to_bytes(byte_len, byteorder="big", signed=False)

        if len(data) >= target_dim:
            buckets = [[] for _ in range(target_dim)]
            for idx, byte in enumerate(data):
                buckets[idx % target_dim].append(byte)
            row = [sum(bucket) / (len(bucket) * 255.0) for bucket in buckets]
        else:
            row = [0.0] * target_dim
            for idx, byte in enumerate(data):
                row[idx] = byte / 255.0

        targets.append(row)
    return targets


def build_centroid_targets(
    encoded: List[List[float]], message_ids: List[int], target_dim: int
) -> List[List[float]]:
    if len(encoded) != len(message_ids):
        raise RuntimeError("encoded vectors do not match message ids length")

    centers: Dict[int, List[float]] = {}
    counts: Dict[int, int] = {}
    for vec, msg_id in zip(encoded, message_ids):
        if msg_id not in centers:
            centers[msg_id] = [0.0] * target_dim
            counts[msg_id] = 0
        counts[msg_id] += 1
        for idx in range(target_dim):
            centers[msg_id][idx] += vec[idx]

    for msg_id, total in centers.items():
        count = max(1, counts.get(msg_id, 1))
        centers[msg_id] = [value / count for value in total]

    return [centers[msg_id] for msg_id in message_ids]


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
        from torch.utils.data import DataLoader, Dataset, TensorDataset
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies. Install torch, numpy, scikit-learn, matplotlib."
        ) from exc

    session = load_jsonish(args.session)
    primes = parse_primes(args)
    values, labels, accuracy, message_ids = collect_samples(
        session, args.batch_index, args.all_batches
    )

    features = build_features(values, primes, args.poly_degree, not args.no_magnitude)
    x = torch.tensor(features, dtype=torch.float32)
    target_dim = args.target_dim if args.target_dim > 0 else args.embedding_dim
    encoded_targets = encode_targets(values, target_dim)
    targets = build_centroid_targets(encoded_targets, message_ids, target_dim)
    y = torch.tensor(targets, dtype=torch.float32)

    class MessageGroupDataset(Dataset):
        def __init__(self, features: torch.Tensor, targets: torch.Tensor, msg_ids: List[int]):
            self.features = features
            self.targets = targets
            groups: Dict[int, List[int]] = {}
            for idx, msg_id in enumerate(msg_ids):
                groups.setdefault(msg_id, []).append(idx)
            self.groups = list(groups.values())

        def __len__(self) -> int:
            return len(self.groups)

        def __getitem__(self, idx: int):
            indices = self.groups[idx]
            return self.features[indices], self.targets[indices]

    dataset = MessageGroupDataset(x, y, message_ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    class ResidueCNN(nn.Module):
        def __init__(self, input_len: int, fc_layers: List[int], embedding_dim: int, target_dim: int):
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
            self.head = nn.Linear(embedding_dim, target_dim)

        def forward(self, inputs):
            x_local = inputs.unsqueeze(1)
            x_local = self.pool(self.relu(self.conv1(x_local)))
            x_local = self.pool(self.relu(self.conv2(x_local)))
            x_local = self.adapt(x_local)
            x_local = x_local.view(x_local.size(0), -1)
            x_local = self.fc(x_local)
            emb = self.embedding(x_local)
            outputs = self.head(emb)
            return outputs, emb

    device = torch.device(args.device)
    model = ResidueCNN(x.shape[1], build_fc_layers(args.fc_layers), args.embedding_dim, target_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        mse_total = 0.0
        total = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.squeeze(0).to(device)
            batch_y = batch_y.squeeze(0).to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            batch_mse = torch.mean((outputs - batch_y) ** 2).item()
            mse_total += batch_mse * batch_x.size(0)
            total += batch_x.size(0)

        avg_loss = total_loss / max(total, 1)
        avg_mse = mse_total / max(total, 1)
        acc = max(0.0, 100.0 * (1.0 - avg_mse))
        print(f"Epoch {epoch+1}/{args.epochs} - loss {avg_loss:.4f} - acc {acc:.2f}%")

    model.eval()
    with torch.no_grad():
        outputs, embeddings = model(x.to(device))
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
