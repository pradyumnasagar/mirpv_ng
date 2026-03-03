# mirpv_ng/mature_model.py
from __future__ import annotations
import joblib
from typing import Dict, Any, Tuple, Optional, List

from .features import run_rnafold
from .mature_ranker import generate_duplex_candidates


class MatureRanker:
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        self.feature_cols: List[str] = payload["feature_cols"]
        self.ranker = payload["ranker"]

    @staticmethod
    def load(path: str) -> "MatureRanker":
        payload = joblib.load(path)
        return MatureRanker(payload)

    def predict_top1(
        self,
        precursor_seq: str,
        rnafold_bin: str = "RNAfold",
        lengths: Tuple[int, ...] = (21, 22, 23, 24),
        max_per_arm: int = 30,
        min_paired_context: int = 6,
        loop_buffer: int = 0,
        fallback_loop_buffer: int = 10,
        fallback_max_per_arm: int = 120,
        fallback_min_paired_context: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Returns dict with arm, start, length, end, score, and sequences.
        Uses loop-edge fallback if needed.
        """
        seq = precursor_seq.upper().replace("T", "U")
        struct, _ = run_rnafold(seq, rnafold_bin=rnafold_bin)

        cands = generate_duplex_candidates(
            seq, struct,
            lengths=lengths,
            max_per_arm=max_per_arm,
            min_paired_context=min_paired_context,
            loop_buffer=loop_buffer,
        )

        if not cands:
        # If nothing sensible, fallback closer to loop edges + looser context
            cands = generate_duplex_candidates(
                seq, struct,
                lengths=lengths,
                max_per_arm=fallback_max_per_arm,
                min_paired_context=fallback_min_paired_context,
                loop_buffer=fallback_loop_buffer,
            )
            if not cands:
                return None

        # Ensure length priors exist (trainer added these; keep consistent at inference)
        for c in cands:
            L = int(c.length)
            c.features.setdefault("feat_len_abs22", float(abs(L - 22)))
            # Provide one-hot priors for 18–24 (safe even if model doesn't use them)
            for k in range(18, 25):
                c.features.setdefault(f"feat_len_is{k}", 1.0 if L == k else 0.0)

        X = [[float(c.features.get(k, 0.0)) for k in self.feature_cols] for c in cands]
        scores = self.ranker.predict(X)

        best_i = int(max(range(len(scores)), key=lambda i: scores[i]))
        best = cands[best_i]
        best_score = float(scores[best_i])

        start = best.start
        end = best.end
        arm = best.arm
        mature_seq = seq[start:end]

        out = {
            "arm": arm,
            "start": int(start),
            "end": int(end),
            "length": int(best.length),
            "score": best_score,
            "mature_seq": mature_seq,
            "partner_start": int(best.partner_start),
            "partner_end": int(best.partner_end),
        }
        return out
