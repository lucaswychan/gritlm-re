import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from gritlm import GritLM
from torch import Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj, num_slices: int = 256, n_global: Optional[int] = None):
        # Run SIGReg in fp32 to avoid bf16/fp32 matmul mismatches and keep
        # trigonometric computations numerically stable under mixed precision.
        with torch.autocast(device_type=proj.device.type, enabled=False):
            proj = proj.float()
            t = self.t.to(device=proj.device, dtype=proj.dtype)
            phi = self.phi.to(device=proj.device, dtype=proj.dtype)
            weights = self.weights.to(device=proj.device, dtype=proj.dtype)

            A = torch.randn(proj.size(-1), num_slices, device=proj.device, dtype=proj.dtype)
            A = A.div_(A.norm(p=2, dim=0))
            x_t = (proj @ A).unsqueeze(-1) * t
            err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
            # Use the global batch size for consistent scaling across device counts.
            # In distributed training n_global = N_local * world_size so that
            # sigreg_weight does not need to be re-tuned when changing GPU count.
            n = n_global if n_global is not None else proj.size(-2)
            statistic = (err @ weights) * n
            return statistic.mean()


@dataclass
class GritLMTrainOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    loss_emb: Optional[Tensor] = None
    loss_gen: Optional[Tensor] = None
    loss_sigreg: Optional[Tensor] = None


class DistributedContrastiveLoss:
    """Contrastive loss module that supports the standard InfoNCE objective as well as
    the *debiased* variant

    The implementation can optionally gather negatives across devices in a
    distributed setup so that every mini-batch benefits from a larger negative
    set without additional memory on a single GPU.
    """

    def __init__(
        self,
        temperature: float,
        negatives_cross_device: bool,
        *,
        debiased: bool = False,
        tau_plus: float = 0.1,
        cosine: bool = True,
    ):
        # Parameters shared by both loss variants
        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device

        # Parameters specific to debiased contrastive loss
        self.debiased = debiased
        self.tau_plus = tau_plus
        self.cosine = cosine

        # Components required only for the standard InfoNCE formulation
        if not self.debiased:
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")

        # Distributed setup
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError("Cannot use negatives_cross_device without distributed training")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # Small caches to reduce per-step allocations
        self._gather_buf: Optional[torch.Tensor] = None
        self._arange_cache = {}

    def __call__(self, q_reps, p_reps):
        if self.negatives_cross_device:
            # Gather q/p in one collective, then restore the same rank-major
            # ordering as two independent gathers.
            q_reps, p_reps = self._dist_gather_qp(q_reps, p_reps)

        if self.debiased:
            return self._debiased_contrastive_loss(q_reps, p_reps)

        # ---------- Standard InfoNCE loss ----------
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        # logger.info(f"scores in contrastive loss: {scores.shape}")
        scores = scores.view(q_reps.size(0), -1)

        # Cache arange to avoid re-allocation
        Bq = scores.size(0)
        base_idx = self._get_arange(Bq, scores.device)
        # Positive index stride within each block of passages
        target = base_idx * (p_reps.size(0) // q_reps.size(0))

        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        if self.world_size == 1:
            return t
        t = t.contiguous()

        # Prefer faster gather-into-tensor if available, while preserving autograd behavior
        use_into_tensor = hasattr(dist, "all_gather_into_tensor")
        if use_into_tensor:
            local_bs = t.size(0)
            out_shape = (local_bs * self.world_size, *t.size()[1:])
            buf_mismatch = self._gather_buf is None or tuple(self._gather_buf.shape) != out_shape or self._gather_buf.dtype != t.dtype or self._gather_buf.device != t.device
            if buf_mismatch:
                self._gather_buf = torch.empty(out_shape, dtype=t.dtype, device=t.device)
            with torch.no_grad():
                dist.all_gather_into_tensor(self._gather_buf, t)
            # Build result by replacing local slice with the original tensor to keep gradients
            chunks = list(self._gather_buf.chunk(self.world_size, dim=0))
            chunks[self.rank] = t
            return torch.cat(chunks, dim=0)
        else:
            logger.info(f"Using list-based gather for dist_gather_tensor")
            # Fallback: list-based gather
            all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
            dist.all_gather(all_tensors, t)
            all_tensors[self.rank] = t
            return torch.cat(all_tensors, dim=0)

    def _dist_gather_qp(self, q_reps: torch.Tensor, p_reps: torch.Tensor):
        if self.world_size == 1:
            return q_reps, p_reps

        q_local_bs = q_reps.size(0)
        p_local_bs = p_reps.size(0)
        local_total_bs = q_local_bs + p_local_bs
        gathered = self._dist_gather_tensor(torch.cat([q_reps, p_reps], dim=0))
        rank_chunks = gathered.split(local_total_bs, dim=0)
        q_chunks = []
        p_chunks = []
        for chunk in rank_chunks:
            q_chunk, p_chunk = chunk.split((q_local_bs, p_local_bs), dim=0)
            q_chunks.append(q_chunk)
            p_chunks.append(p_chunk)
        return torch.cat(q_chunks, dim=0), torch.cat(p_chunks, dim=0)

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    # ---------------------------------------------------------------------
    # Helper methods for the debiased contrastive loss
    # ---------------------------------------------------------------------
    def _debiased_contrastive_loss(self, q_reps: torch.Tensor, p_reps: torch.Tensor) -> torch.Tensor:
        """Assumes each query has *M* associated passages (`p_reps`) where the
        first passage in each consecutive block of size *M* is the positive
        example and the remaining *M-1* are negatives.
        """

        B = q_reps.size(0)
        # Determine group size M (passages per query)
        assert p_reps.size(0) % B == 0, "p_reps should contain an equal number of passages per query"
        M = p_reps.size(0) // B

        # Optionally normalise representations (cosine similarity)
        if self.cosine:
            q_norm = F.normalize(q_reps, dim=1)
            p_norm = F.normalize(p_reps, dim=1)
        else:
            q_norm = q_reps
            p_norm = p_reps

        # Similarity matrix between queries and ALL passages  [B, B*M]
        sim = torch.matmul(q_norm, p_norm.transpose(0, 1)) / self.temperature  # scaled by temperature

        # Compute row-wise log-sum-exp for numerical stability and reduced memory
        logsumexp_all = torch.logsumexp(sim, dim=1)  # [B]
        sum_exp_all = torch.exp(logsumexp_all)  # [B]

        # Positive indices are at i*M for query i
        base_idx = self._get_arange(B, sim.device)
        pos_logit = sim[base_idx, base_idx * M]  # [B]
        pos = torch.exp(pos_logit)  # [B]

        # All negatives for each query
        neg = sum_exp_all - pos  # [B]

        # De-biased estimator Ng
        if self.debiased:
            N = M * B - 1  # total negatives per query after concat across processes
            Ng = (-self.tau_plus * N * pos + neg) / (1.0 - self.tau_plus)
            Ng = torch.clamp(Ng, min=N * np.exp(-1.0 / self.temperature))
        else:
            Ng = neg

        loss = (-torch.log(pos / (pos + Ng))).mean()
        return loss

    @staticmethod
    def _get_masks(batch_size: int, device: torch.device):
        """Return boolean masks for negative and positive pairs.

        For a concatenated batch `[q_1..q_B, p_1..p_B]` of size `2B`, the
        positive for index *i* is `i+B` (if `i < B`) and `i-B` otherwise. All
        other indices are treated as negatives, except self-pairs which are
        excluded.
        """
        # Negative mask
        neg_mask = torch.ones((batch_size, 2 * batch_size), dtype=torch.bool, device=device)
        for i in range(batch_size):
            neg_mask[i, i] = False  # q_i with q_i (self)
            neg_mask[i, i + batch_size] = False  # q_i with p_i (positive)
        neg_mask = torch.cat([neg_mask, neg_mask], dim=0)

        # Positive mask
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=device)
        idx = torch.arange(batch_size, device=device)
        pos_mask[idx, idx + batch_size] = True
        pos_mask[idx + batch_size, idx] = True

        return neg_mask, pos_mask

    def _get_arange(self, n: int, device: torch.device) -> torch.Tensor:
        key = (device.type, device.index)
        cached = self._arange_cache.get(key)
        if cached is None or cached.numel() < n:
            cached = torch.arange(n, device=device)
            self._arange_cache[key] = cached
        else:
            # Narrow without allocating when smaller n is requested
            cached = cached[:n]
        return cached


class GritLMTrainModel(GritLM):
    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        temperature: float = 1.0,
        negatives_cross_device: bool = False,
        debiased: bool = False,
        tau_plus: float = 0.1,
        sigreg_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs, is_inference=False)
        self.emb_loss_fn = DistributedContrastiveLoss(temperature, negatives_cross_device, debiased=debiased, tau_plus=tau_plus)
        self.config = self.model.config  # Required for accelerate DeepSpeed integration
        self.sigreg_weight = sigreg_weight
        self.sigreg = SIGReg() if sigreg_weight > 0.0 else None

    def _scale_for_sigreg(self, reps: torch.Tensor) -> torch.Tensor:
        """Rescale embeddings so that 1D projections have ~unit variance.

        SIGReg's reference CF is exp(-t²/2), the CF of N(0,1).  For L2-normalised
        unit vectors in D dimensions the projection onto any direction has variance
        1/D, so the empirical CF stays near 1 for all t in [0,3].  The error
        (ECF - phi)² is then dominated by the (1 - phi)² term and its gradient
        pushes embeddings toward *higher* anisotropy — the exact opposite of the
        intended effect.  Multiplying by sqrt(D) gives projections with unit
        variance, matching the Gaussian target and making the loss zero for a
        perfectly isotropic distribution on the sphere.
        """
        if self.normalized:
            return reps * (reps.shape[-1] ** 0.5)
        return reps

    @property
    def combined_loss_fn(self):
        """Combined contrastive + SIGReg loss for the GradCache path.

        GradCache never calls forward() with both query and passage at the same
        time, so the SIGReg guard inside forward() never fires.  GradCache
        does however call loss_fn(q_reps, p_reps) in build_cache() on the full
        assembled local batch.  Returning this combined callable as the GradCache
        loss_fn injects SIGReg at that point: its gradient flows into the cache
        and is then propagated back into the encoder via the surrogate dot-product
        in forward_backward(), exactly like the contrastive gradient.
        """
        if self.sigreg is None:
            return self.emb_loss_fn

        emb_loss_fn = self.emb_loss_fn
        sigreg = self.sigreg
        sigreg_weight = self.sigreg_weight
        scale_for_sigreg = self._scale_for_sigreg

        def _combined(q_reps, p_reps):
            contrastive_loss = emb_loss_fn(q_reps, p_reps)
            all_reps = torch.cat([q_reps, p_reps], dim=0)
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            n_global = all_reps.size(0) * world_size
            sigreg_loss = sigreg(scale_for_sigreg(all_reps), n_global=n_global).to(contrastive_loss.dtype)
            return (1 - sigreg_weight) * contrastive_loss + sigreg_weight * sigreg_loss # changed from (1 - sigreg_weight) * contrastive_loss + sigreg_weight * sigreg_loss to (1 - sigreg_weight) * contrastive_loss + sigreg_weight * sigreg_loss
            # return contrastive_loss + sigreg_weight * sigreg_loss

        return _combined

    def encode(self, features):
        if features is None:
            return None
        # Use in-place operations where safe to reduce memory allocations
        attention_mask = features.get("attention_mask")
        instruction_lens = features.get("instruction_lens")
        kwargs = {"input_ids": features.get("input_ids"), "attention_mask": attention_mask}

        if self.attn == "bb":
            kwargs["is_causal"] = False
        out = self.model(**kwargs)[0]

        if self.projection is not None:
            out = self.projection(out)

        # Mask out the instruction tokens for pooling
        # @lucaswychan add checking of instruction_lens, since original approach will assume there is instruction in the passage
        if instruction_lens is not None and len(instruction_lens) > 0:
            # Clone only when necessary for instruction masking
            attention_mask = attention_mask.clone()
            # Vectorized masking for better performance - use advanced indexing
            batch_size = attention_mask.size(0)
            # Create a mask tensor for vectorized operation
            if batch_size > 0:
                # Always apply the vectorized mask. We intentionally avoid the previous
                # `max(instruction_lens) > 0` guard because reducing a CUDA tensor to a
                # Python int forces a GPU->CPU sync every encode (~20x/step), stalling the
                # CPU so it cannot queue the next FSDP forward. The guard is unnecessary:
                # for any entry whose instruction length is 0, (seq_indices < 0) is all-False,
                # so the masked assignment is a no-op for it and the result is identical.
                seq_indices = torch.arange(attention_mask.size(1), device=attention_mask.device)
                # Properly create tensor from list (not from another tensor)
                if isinstance(instruction_lens, torch.Tensor):
                    instr_lens_tensor = instruction_lens.to(attention_mask.device).unsqueeze(1)
                else:
                    instr_lens_tensor = torch.as_tensor(instruction_lens, dtype=torch.long, device=attention_mask.device).unsqueeze(1)
                attention_mask[seq_indices < instr_lens_tensor] = 0

                # Verify not all zeros - If this happens it is a bug. Gated behind an env flag
                # because `.all()` is another per-encode GPU->CPU sync; the check does not affect
                # training outputs, so it is opt-in for debugging only.
                if os.getenv("GRITLM_DEBUG_MASK", "0") == "1":
                    assert (attention_mask.sum(dim=1) > 0).all(), "Some samples have all-zero attention masks"

        reps = self.pooling(out, attention_mask)
        # Normalize can change the dtype (https://discuss.pytorch.org/t/tensor-in-float16-is-transformed-into-float32-after-torch-norm/110891)
        if self.normalized:
            in_dtype = reps.dtype
            return torch.nn.functional.normalize(reps, dim=-1).contiguous().to(in_dtype)
        return reps.contiguous()

    def forward(
        self,
        query: Dict[str, torch.Tensor] = None,
        passage: Dict[str, torch.Tensor] = None,
        q_reps: Optional[torch.Tensor] = None,
        p_reps: Optional[torch.Tensor] = None,
        q_grad: bool = True,
        p_grad: bool = True,
    ):
        """
        Args:
            query: [b, n]
            passage: [b*s, m] where s is group size (usually 2)
        """
        if (q_reps is None) and (query is not None):
            if q_grad:
                q_reps = self.encode(query)
            else:
                with torch.no_grad():
                    q_reps = self.encode(query)

        if (p_reps is None) and (passage is not None):
            if p_grad:
                p_reps = self.encode(passage)
            else:
                with torch.no_grad():
                    p_reps = self.encode(passage)

        loss_emb = self.emb_loss_fn(q_reps, p_reps) if (q_reps is not None and p_reps is not None) else None

        loss_sigreg = None
        if self.sigreg is not None and q_reps is not None and p_reps is not None:
            all_reps = torch.cat([q_reps, p_reps], dim=0)  # (N + N*M, D)
            # Pass the global batch size so the test statistic scales correctly
            # with world size. Memory stays proportional to N_local — no gather.
            # Note: SIGReg is skipped in the GradCache path because GradCache
            # calls forward() with passage=None per chunk; p_reps stays None.
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            n_global = all_reps.size(0) * world_size
            loss_sigreg = self.sigreg(self._scale_for_sigreg(all_reps), n_global=n_global).to(loss_emb.dtype)

        loss = loss_emb
        if loss_sigreg is not None:
            loss = (1 - self.sigreg_weight) * loss_emb + self.sigreg_weight * loss_sigreg

        # Also return q_reps in case of GradCache
        return GritLMTrainOutput(
            q_reps=q_reps,
            p_reps=p_reps,
            loss=loss,
            loss_emb=loss_emb,
            loss_gen=None,
            loss_sigreg=loss_sigreg,
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)
