#!/usr/bin/env python3
# mirpv_ng/train_mature_ranker.py
# Unified XGBRanker training with GroupKFold CV and loop-edge fallback.

from __future__ import annotations

import argparse
import pickle
from typing import Dict, Tuple, List, Optional

import numpy as np
import xgboost as xgb
from Bio import SeqIO
from sklearn.model_selection import GroupKFold

from mirpv_ng.features import run_rnafold
from mirpv_ng.mature_ranker import generate_duplex_candidates


def norm_seq(s: str) -> str:
    return str(s).upper().replace("T", "U").replace(" ", "").replace("\r", "")


def base_hairpin_id(seq_id: str) -> str:
    sid = seq_id.split()[0].replace("*", "")
    if "_" in sid:
        sid = sid.rsplit("_", 1)[0]
    return sid


def load_precursors_from_mixed_fasta(pre_fa: str) -> Dict[str, str]:
    pre_seqs: Dict[str, str] = {}
    for rec in SeqIO.parse(pre_fa, "fasta"):
        rid = rec.id.split()[0]
        if not rid.endswith("_pre"):
            continue
        bid = base_hairpin_id(rid)
        pre_seqs[bid] = norm_seq(rec.seq)
    return pre_seqs


def load_arm_truth(fa_path: str, pre_seqs: Dict[str, str]) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    for rec in SeqIO.parse(fa_path, "fasta"):
        bid = base_hairpin_id(rec.id)
        if bid not in pre_seqs:
            continue
        mseq = norm_seq(rec.seq)
        pos = pre_seqs[bid].find(mseq)
        if pos != -1:
            out[bid] = (pos, len(mseq))
    return out


def build_truth_map(
    pre_fa: str,
    fa_5p: Optional[str],
    fa_3p: Optional[str],
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Tuple[int, int]]]]:
    pre_seqs = load_precursors_from_mixed_fasta(pre_fa)
    truth: Dict[str, Dict[str, Tuple[int, int]]] = {}
    if fa_5p:
        t5 = load_arm_truth(fa_5p, pre_seqs)
        for bid, v in t5.items():
            truth.setdefault(bid, {})["5p"] = v
    if fa_3p:
        t3 = load_arm_truth(fa_3p, pre_seqs)
        for bid, v in t3.items():
            truth.setdefault(bid, {})["3p"] = v
    return pre_seqs, truth


def relevance_for_candidate(
    cand_start: int,
    cand_len: int,
    cand_arm: str,
    truth_entry: Dict[str, Tuple[int, int]],
    near_start_window: int,
    near_len_window: int,
) -> int:
    if cand_arm not in truth_entry:
        return 0
    t_start, t_len = truth_entry[cand_arm]
    if cand_start == t_start and cand_len == t_len:
        return 2
    if abs(cand_start - t_start) <= near_start_window and abs(cand_len - t_len) <= near_len_window:
        return 1
    return 0


def add_length_consistency_features(feat: Dict[str, float], L: int) -> None:
    feat["feat_len_abs22"] = float(abs(L - 22))
    feat["feat_len_is21"] = 1.0 if L == 21 else 0.0
    feat["feat_len_is22"] = 1.0 if L == 22 else 0.0
    feat["feat_len_is23"] = 1.0 if L == 23 else 0.0
    feat["feat_len_is24"] = 1.0 if L == 24 else 0.0


def score_groups_top1(
    scores: np.ndarray,
    group_sizes: List[int],
    cand_arms: List[str],
    cand_starts: List[int],
    cand_lens: List[int],
    truth_per_group: List[Dict[str, Tuple[int, int]]],
    start_window: int = 1,
) -> Dict[str, float]:
    idx = 0
    n_q = 0
    full_exact = full_near = start_exact = start_near = arm_correct = start_exact_len_wrong = 0

    for gi, g in enumerate(group_sizes):
        sl = slice(idx, idx + g)
        best = int(np.argmax(scores[sl]))

        best_arm = cand_arms[idx + best]
        best_start = cand_starts[idx + best]
        best_len = cand_lens[idx + best]
        tinfo = truth_per_group[gi]

        if best_arm in tinfo:
            arm_correct += 1
            t_start, t_len = tinfo[best_arm]

            if best_start == t_start:
                start_exact += 1
                if best_len != t_len:
                    start_exact_len_wrong += 1
            if abs(best_start - t_start) <= start_window:
                start_near += 1

            if best_start == t_start and best_len == t_len:
                full_exact += 1
            if abs(best_start - t_start) <= start_window and abs(best_len - t_len) <= 1:
                full_near += 1

        n_q += 1
        idx += g

    if n_q == 0:
        return {k: 0.0 for k in [
            "top1_full_exact", "top1_full_near",
            "top1_start_exact", "top1_start_near",
            "arm_correct", "start_exact_len_wrong",
        ]}

    return {
        "top1_full_exact": full_exact / n_q,
        "top1_full_near": full_near / n_q,
        "top1_start_exact": start_exact / n_q,
        "top1_start_near": start_near / n_q,
        "arm_correct": arm_correct / n_q,
        "start_exact_len_wrong": start_exact_len_wrong / n_q,
    }


class HairpinGroup:
    def __init__(self, hid: str, X: np.ndarray, y: np.ndarray,
                 arms: List[str], starts: List[int], lens: List[int],
                 truth: Dict[str, Tuple[int, int]]):
        self.hid = hid
        self.X = X
        self.y = y
        self.arms = arms
        self.starts = starts
        self.lens = lens
        self.truth = truth

    @property
    def size(self) -> int:
        return int(self.X.shape[0])


def _cands_and_labels(
    seq: str,
    struct: str,
    tinfo: Dict[str, Tuple[int, int]],
    lengths: Tuple[int, ...],
    max_per_arm: int,
    min_paired_context: int,
    loop_buffer: int,
    near_start_window: int,
    near_len_window: int,
) -> Tuple[list, list]:
    cands = generate_duplex_candidates(
        seq, struct,
        lengths=lengths,
        max_per_arm=max_per_arm,
        min_paired_context=min_paired_context,
        loop_buffer=loop_buffer,
    )
    if not cands:
        return [], []
    for c in cands:
        add_length_consistency_features(c.features, c.length)
    labels = [
        relevance_for_candidate(
            c.start, c.length, c.arm, tinfo,
            near_start_window=near_start_window,
            near_len_window=near_len_window,
        )
        for c in cands
    ]
    return cands, labels


def build_groups(
    pre_seqs: Dict[str, str],
    truth: Dict[str, Dict[str, Tuple[int, int]]],
    rnafold_bin: str,
    lengths: Tuple[int, ...],
    max_per_arm: int,
    min_paired_context: int,
    loop_buffer: int,
    near_start_window: int,
    near_len_window: int,
    fallback_enabled: bool,
    fallback_max_per_arm: int,
    fallback_min_paired_context: int,
    fallback_loop_buffer: int,
) -> Tuple[List[HairpinGroup], List[str], Dict[str, int]]:
    raw = []
    feat_names = set()
    stats = {"mapped_truth": 0, "usable_groups": 0, "dropped_no_cands": 0, "dropped_no_pos": 0, "rescued": 0}

    for hid, tinfo in truth.items():
        stats["mapped_truth"] += 1
        seq = pre_seqs.get(hid)
        if not seq:
            continue

        struct, _ = run_rnafold(seq, rnafold_bin=rnafold_bin)

        cands, labels = _cands_and_labels(
            seq, struct, tinfo,
            lengths=lengths,
            max_per_arm=max_per_arm,
            min_paired_context=min_paired_context,
            loop_buffer=loop_buffer,
            near_start_window=near_start_window,
            near_len_window=near_len_window,
        )
        if not cands:
            stats["dropped_no_cands"] += 1
            continue

        if max(labels) == 0 and fallback_enabled:
            c2, y2 = _cands_and_labels(
                seq, struct, tinfo,
                lengths=lengths,
                max_per_arm=fallback_max_per_arm,
                min_paired_context=fallback_min_paired_context,
                loop_buffer=fallback_loop_buffer,
                near_start_window=near_start_window,
                near_len_window=near_len_window,
            )
            if c2 and max(y2) > 0:
                cands, labels = c2, y2
                stats["rescued"] += 1

        if max(labels) == 0:
            stats["dropped_no_pos"] += 1
            continue

        for c in cands:
            feat_names.update(c.features.keys())

        raw.append((hid, cands, labels, tinfo))
        stats["usable_groups"] += 1

    if not raw:
        raise SystemExit("[error] No usable hairpins produced training groups.")

    feature_cols = sorted(feat_names)
    groups: List[HairpinGroup] = []
    for hid, cands, labels, tinfo in raw:
        X = np.asarray([[float(c.features.get(k, 0.0)) for k in feature_cols] for c in cands], dtype=float)
        y = np.asarray(labels, dtype=int)
        arms = [c.arm for c in cands]
        starts = [int(c.start) for c in cands]
        lens = [int(c.length) for c in cands]
        groups.append(HairpinGroup(hid, X, y, arms, starts, lens, tinfo))

    return groups, feature_cols, stats


def flatten_groups(groups: List[HairpinGroup]):
    X = np.vstack([g.X for g in groups])
    y = np.concatenate([g.y for g in groups])
    gsz = [g.size for g in groups]
    arms, starts, lens, truth, gids = [], [], [], [], []
    for g in groups:
        arms.extend(g.arms)
        starts.extend(g.starts)
        lens.extend(g.lens)
        truth.append(g.truth)
        gids.append(g.hid)
    return X, y, gsz, arms, starts, lens, truth, gids


def make_ranker(args) -> xgb.XGBRanker:
    return xgb.XGBRanker(
        objective="rank:ndcg",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )


def run_cv(groups: List[HairpinGroup], args):
    gkf = GroupKFold(n_splits=args.cv)
    idxs = np.arange(len(groups))
    labels = np.array([g.hid for g in groups], dtype=object)

    folds = []
    for fold, (tr, te) in enumerate(gkf.split(idxs, groups=labels), start=1):
        tr_groups = [groups[i] for i in tr]
        te_groups = [groups[i] for i in te]

        Xtr, ytr, gtr, *_ = flatten_groups(tr_groups)
        Xte, yte, gte, arms, starts, lens, truth_te, _ = flatten_groups(te_groups)

        model = make_ranker(args)
        model.fit(Xtr, ytr, group=gtr, verbose=False)
        scores = model.predict(Xte)

        m = score_groups_top1(scores, gte, arms, starts, lens, truth_te, start_window=args.report_start_window)
        folds.append(m)

        print(
            f"[cv] fold {fold}/{args.cv}: "
            f"full_exact={m['top1_full_exact']:.3f} full_near={m['top1_full_near']:.3f} "
            f"start_exact={m['top1_start_exact']:.3f} start_near={m['top1_start_near']:.3f} "
            f"arm_ok={m['arm_correct']:.3f} start_ok_len_wrong={m['start_exact_len_wrong']:.3f}"
        )

    def ms(key: str):
        v = np.array([f[key] for f in folds], dtype=float)
        return float(v.mean()), float(v.std(ddof=0))

    for k in ["top1_full_exact","top1_full_near","top1_start_exact","top1_start_near","arm_correct","start_exact_len_wrong"]:
        mu, sd = ms(k)
        print(f"[cv] {k}: {mu:.3f} ± {sd:.3f}")


def train_final(groups: List[HairpinGroup], feature_cols: List[str], args):
    X, y, gsz, arms, starts, lens, truth, _ = flatten_groups(groups)
    model = make_ranker(args)
    model.fit(X, y, group=gsz, verbose=False)

    payload = {"model_type": "xgboost_ranker_unified", "feature_cols": feature_cols, "ranker": model}
    with open(args.model_out, "wb") as f:
        pickle.dump(payload, f)

    print(f"[train] saved model: {args.model_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-fasta", required=True)
    ap.add_argument("--fasta-5p", default=None)
    ap.add_argument("--fasta-3p", default=None)
    ap.add_argument("--rnafold-bin", default="RNAfold")
    ap.add_argument("--model-out", required=True)

    ap.add_argument("--lengths", nargs="+", type=int, default=[21, 22, 23, 24])
    ap.add_argument("--max-per-arm", type=int, default=30)
    ap.add_argument("--min-paired-context", type=int, default=6)
    ap.add_argument("--loop-buffer", type=int, default=0)

    ap.add_argument("--near-start-window", type=int, default=1)
    ap.add_argument("--near-len-window", type=int, default=3)
    ap.add_argument("--report-start-window", type=int, default=1)

    ap.add_argument("--fallback-enabled", action="store_true", default=True)
    ap.add_argument("--no-fallback", dest="fallback_enabled", action="store_false")
    ap.add_argument("--fallback-max-per-arm", type=int, default=120)
    ap.add_argument("--fallback-min-paired-context", type=int, default=0)
    ap.add_argument("--fallback-loop-buffer", type=int, default=10)

    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample-bytree", type=float, default=0.9)
    ap.add_argument("--n-jobs", type=int, default=8)

    ap.add_argument("--cv", type=int, default=0)
    args = ap.parse_args()
    args.lengths = tuple(int(x) for x in args.lengths)

    pre, truth = build_truth_map(args.pre_fasta, args.fasta_5p, args.fasta_3p)
    print(f"[data] precursors: {len(pre)}")
    print(f"[data] hairpins with mapped truth: {len(truth)}")

    groups, feature_cols, stats = build_groups(
        pre_seqs=pre,
        truth=truth,
        rnafold_bin=args.rnafold_bin,
        lengths=args.lengths,
        max_per_arm=args.max_per_arm,
        min_paired_context=args.min_paired_context,
        loop_buffer=args.loop_buffer,
        near_start_window=args.near_start_window,
        near_len_window=args.near_len_window,
        fallback_enabled=args.fallback_enabled,
        fallback_max_per_arm=args.fallback_max_per_arm,
        fallback_min_paired_context=args.fallback_min_paired_context,
        fallback_loop_buffer=args.fallback_loop_buffer,
    )

    X, y, gsz, *_ = flatten_groups(groups)
    print(f"[data] usable groups (hairpins): {len(groups)}")
    print(f"[data] total candidate rows: {X.shape[0]}")
    print(f"[data] feature count: {len(feature_cols)}")
    print(f"[data] dropped: no_cands={stats['dropped_no_cands']} no_pos={stats['dropped_no_pos']} rescued={stats['rescued']}")

    if args.cv and args.cv > 1:
        run_cv(groups, args)
        return

    train_final(groups, feature_cols, args)


if __name__ == "__main__":
    main()
