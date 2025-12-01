from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch import nn

def get_params_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.flatten() for p in model.parameters()])

def set_params_vector(model: nn.Module, vec: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[offset:offset+n].view_as(p.data))
        offset += n

def get_grads_vector(model: nn.Module) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p.data).flatten())
        else:
            grads.append(p.grad.data.flatten())
    return torch.cat(grads)

def topk_sparsify(vec: torch.Tensor, k_frac: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Keep top-k fraction by absolute magnitude.
    Returns (indices, values).
    """
    n = vec.numel()
    k = max(1, int(n * k_frac))
    _, idx = torch.topk(vec.abs(), k=k, largest=True, sorted=False)
    vals = vec[idx]
    return idx.to(torch.int32), vals.to(torch.float32)

def topk_desparsify(n_params: int, idx: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(n_params, dtype=torch.float32)
    out[idx.long()] = vals
    return out

def estimate_bytes_full(n_params: int) -> int:
    return n_params * 4  # float32

def estimate_bytes_topk(k: int) -> int:
    # index int32 (4 bytes) + value float32 (4 bytes)
    return k * 8

def secure_mask(vec: torch.Tensor, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Toy secure aggregation masking (simulation only):
    Adds a random mask derived from seed; server subtracts masks later.
    """
    rng = np.random.default_rng(seed)
    mask = torch.tensor(rng.normal(0, 0.01, size=vec.numel()), dtype=torch.float32)
    return vec + mask, mask

@dataclass
class ClientUpdate:
    client_id: str
    n_samples: int
    quality: float
    masked_sparse_idx: torch.Tensor
    masked_sparse_vals: torch.Tensor
    mask_sparse_idx: torch.Tensor
    mask_sparse_vals: torch.Tensor
    bytes_sent: int
    sent: bool

class FederatedClient:
    def __init__(
        self,
        client_id: str,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        quality: float,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.X_train = X_train
        self.Y_train = Y_train
        self.quality = float(quality)
        self.device = device

    def local_train_and_update(
        self,
        global_model: nn.Module,
        lr: float = 1e-3,
        local_steps: int = 50,
        batch_size: int = 64,
        k_frac: float = 0.3,
        send_threshold: float = 0.0,
        mask_seed: int = 123,
    ) -> ClientUpdate:
        model = global_model
        model.train()
        n_params = sum(p.numel() for p in model.parameters())
        X = torch.from_numpy(self.X_train).to(torch.float32).to(self.device)
        Y = torch.from_numpy(self.Y_train).to(torch.float32).to(self.device)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.SmoothL1Loss()

        # Store starting params
        w0 = get_params_vector(model).detach().clone()

        n = X.shape[0]
        for _ in range(local_steps):
            idx = torch.randint(0, n, (batch_size,))
            xb = X[idx]
            yb = Y[idx]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        w1 = get_params_vector(model).detach().clone()
        update = (w1 - w0).to(torch.float32)

        # Selective update: skip tiny updates
        upd_norm = float(update.norm().item())
        if upd_norm <= send_threshold:
            return ClientUpdate(
                client_id=self.client_id,
                n_samples=int(n),
                quality=self.quality,
                masked_sparse_idx=torch.empty(0, dtype=torch.int32),
                masked_sparse_vals=torch.empty(0, dtype=torch.float32),
                mask_sparse_idx=torch.empty(0, dtype=torch.int32),
                mask_sparse_vals=torch.empty(0, dtype=torch.float32),
                bytes_sent=0,
                sent=False,
            )

        # Sparsify
        idx, vals = topk_sparsify(update, k_frac=k_frac)
        bytes_sent = estimate_bytes_topk(k=idx.numel())

        # Mask (toy secure-agg)
        sparse_vec = vals
        masked_vals, mask_vals = secure_mask(sparse_vec, seed=mask_seed + hash(self.client_id) % 10_000)

        return ClientUpdate(
            client_id=self.client_id,
            n_samples=int(n),
            quality=self.quality,
            masked_sparse_idx=idx,
            masked_sparse_vals=masked_vals,
            mask_sparse_idx=idx.clone(),
            mask_sparse_vals=mask_vals,
            bytes_sent=bytes_sent,
            sent=True,
        )

class FederatedServer:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device

    def aggregate(
        self,
        updates: List[ClientUpdate],
        k_frac: float,
    ) -> Dict[str, float]:
        """
        Client-aware weighted aggregation + unmask + apply to global model.
        """
        n_params = sum(p.numel() for p in self.model.parameters())
        sent_updates = [u for u in updates if u.sent]
        if not sent_updates:
            return {"clients_sent": 0, "bytes_sent": 0, "bytes_full": 0, "comm_reduction": 0.0}

        # Client-aware weights (data volume * quality)
        weights = np.array([u.n_samples * u.quality for u in sent_updates], dtype=np.float64)
        weights = weights / (weights.sum() + 1e-12)

        agg_update = torch.zeros(n_params, dtype=torch.float32)

        total_bytes = int(sum(u.bytes_sent for u in sent_updates))
        bytes_full = estimate_bytes_full(n_params) * len(sent_updates)

        for w, u in zip(weights, sent_updates):
            # Unmask (toy)
            unmasked_vals = (u.masked_sparse_vals - u.mask_sparse_vals).to(torch.float32)
            dense = topk_desparsify(n_params, u.masked_sparse_idx, unmasked_vals)
            agg_update += float(w) * dense

        # Apply aggregated update
        w0 = get_params_vector(self.model).detach().clone()
        set_params_vector(self.model, (w0 + agg_update).to(torch.float32))

        comm_reduction = 1.0 - (total_bytes / max(1, bytes_full))
        return {
            "clients_sent": int(len(sent_updates)),
            "bytes_sent": float(total_bytes),
            "bytes_full": float(bytes_full),
            "comm_reduction": float(comm_reduction),
        }
