#!/usr/bin/env python3
"""Legacy diagnostics -- NOT used by the published analysis.

These are the graph-neural-network and GraphMixUp components from the earlier
version of this work, together with the circular-label helpers. The published
manuscript reports none of their output: the two-node graph was shown to be
algebraically equivalent to a multilayer perceptron, and interpolation-based
augmentation was shown not to reproduce held-out intermediate-dose physiology.

They are retained here, separate from the reproduction path, so the negative
results can be regenerated if a reviewer asks about them. The main pipeline
`arsenic_system.py` does not import this file.

Usage:
    import arsenic_system as ap      # data loading and metrics
    import legacy_diagnostics as ld
    ld.exp2_gcn_audit(ap.load_all())
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.base import BaseEstimator, clone
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from arsenic_system import (BIOMARKERS, CONDITIONS, LOEC, OUT, SEED, SYMBOL,
                            CTEX, CKEYS, exposed_label, fast_auc, lodgo,
                            metrics, subset, write_table)

INF = float("inf")

def legacy_label(d, tau):
    """y_i = max_m 1[c_i >= tau_m]  (the submitted max-aggregation rule)."""
    c = d.concentration.to_numpy()
    return np.stack([(c >= tau.get(m, INF)).astype(int)
                     for m in BIOMARKERS]).max(axis=0)

def collapsed_label(d, tau):
    """y_i = 1[c_i >= min_m tau_m]  -- algebraically identical to the above."""
    return (d.concentration.to_numpy() >= min(tau.get(m, INF)
                                              for m in BIOMARKERS)).astype(int)

def normalised_adjacency(A, self_loops=True):
    A = np.asarray(A, dtype=float)
    At = A + np.eye(A.shape[0]) if self_loops else A.copy()
    deg = At.sum(1)
    dinv = np.where(deg > 0, deg ** -0.5, 0.0)
    return (dinv[:, None] * At) * dinv[None, :]

def two_node_operator():
    return normalised_adjacency(np.array([[0.0, 1.0], [1.0, 0.0]]))

def two_node_features(conc, bio):
    n, m = bio.shape
    xe = np.zeros((n, m + 1)); xe[:, 0] = conc; xe[:, -1] = 1.0
    xb = np.concatenate([bio, np.ones((n, 1))], axis=1)
    return np.stack([xe, xb], axis=1)

def two_node_collapsed(conc, bio):
    """The only function of the inputs a two-node GCN can represent."""
    n, m = bio.shape
    u = np.zeros((n, m + 1))
    u[:, 0] = 0.5 * (conc + bio[:, 0])
    u[:, 1:m] = 0.5 * bio[:, 1:]
    u[:, m] = 1.0
    return u

def biomarker_graph_features(z):
    n, m = z.shape
    X = np.zeros((n, m, m + 1))
    X[:, :, 0] = z
    X[:, :, 1:] = np.eye(m)[None, :, :]
    return X

class GCN:
    """Two GCNConv layers, ReLU, dropout, mean pooling, two FC layers,
    class-weighted BCE-with-logits, Adam with decoupled L2."""

    def __init__(self, S, d0, hidden=32, fc=32, dropout=0.2, lr=1e-3, wd=1e-4,
                 epochs=200, batch=16, seed=0, pos_weight=1.0):
        self.__dict__.update(S=np.asarray(S, float), d0=d0, hidden=hidden, fc=fc,
                             dropout=dropout, lr=lr, wd=wd, epochs=epochs,
                             batch=batch, seed=seed, pos_weight=pos_weight)

    def _init(self, rng):
        g = lambda a, b: rng.uniform(-np.sqrt(6 / (a + b)), np.sqrt(6 / (a + b)), (a, b))
        self.P = {"W1": g(self.d0, self.hidden), "W2": g(self.hidden, self.hidden),
                  "W3": g(self.hidden, self.fc), "b3": np.zeros(self.fc),
                  "W4": g(self.fc, 1), "b4": np.zeros(1)}
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    def forward(self, X, rng=None, train=False):
        P, S, p = self.P, self.S, self.dropout
        SH0 = np.einsum("nm,bmd->bnd", S, X)
        Z1 = SH0 @ P["W1"]; H1 = np.maximum(Z1, 0)
        m1 = (rng.random(H1.shape) >= p) / (1 - p) if (train and p > 0) else None
        H1d = H1 * m1 if m1 is not None else H1
        SH1 = np.einsum("nm,bmd->bnd", S, H1d)
        Z2 = SH1 @ P["W2"]; H2 = np.maximum(Z2, 0)
        g = H2.mean(1)
        Z3 = g @ P["W3"] + P["b3"]; H3 = np.maximum(Z3, 0)
        m3 = (rng.random(H3.shape) >= p) / (1 - p) if (train and p > 0) else None
        H3d = H3 * m3 if m3 is not None else H3
        logit = (H3d @ P["W4"] + P["b4"]).ravel()
        return logit, dict(SH0=SH0, Z1=Z1, m1=m1, H1d=H1d, SH1=SH1, Z2=Z2,
                           g=g, Z3=Z3, m3=m3, H3d=H3d)

    def _backward(self, c, logit, y, w):
        P, S = self.P, self.S
        B = len(y)
        sig = 1 / (1 + np.exp(-np.clip(logit, -60, 60)))
        dz = ((sig * (w * y + 1 - y) - w * y) / B)[:, None]
        G = {"W4": c["H3d"].T @ dz, "b4": dz.sum(0)}
        dH3 = dz @ P["W4"].T
        if c["m3"] is not None:
            dH3 = dH3 * c["m3"]
        dZ3 = dH3 * (c["Z3"] > 0)
        G["W3"] = c["g"].T @ dZ3; G["b3"] = dZ3.sum(0)
        dg = dZ3 @ P["W3"].T
        n = c["Z2"].shape[1]
        dZ2 = np.repeat(dg[:, None, :], n, 1) / n * (c["Z2"] > 0)
        G["W2"] = np.einsum("bnd,bnk->dk", c["SH1"], dZ2)
        dH1 = np.einsum("mn,bnd->bmd", S.T, dZ2 @ P["W2"].T)
        if c["m1"] is not None:
            dH1 = dH1 * c["m1"]
        dZ1 = dH1 * (c["Z1"] > 0)
        G["W1"] = np.einsum("bnd,bnk->dk", c["SH0"], dZ1)
        return G

    def _adam(self, G, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k, g in G.items():
            if k.startswith("W"):
                g = g + self.wd * self.P[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g ** 2
            self.P[k] -= self.lr * (self.m[k] / (1 - b1 ** self.t)) / \
                (np.sqrt(self.v[k] / (1 - b2 ** self.t)) + eps)

    def fit(self, X, y, mixup_alpha=None, mix_labels=True):
        rng = np.random.default_rng(self.seed)
        self._init(rng)
        y = np.asarray(y, float); n = len(y)
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.batch):
                idx = order[s:s + self.batch]
                xb, yb = X[idx], y[idx]
                if mixup_alpha is not None and len(idx) > 1:
                    lam = rng.beta(mixup_alpha, mixup_alpha)
                    perm = rng.permutation(len(idx))
                    xb = lam * xb + (1 - lam) * xb[perm]
                    if mix_labels:
                        yb = lam * yb + (1 - lam) * yb[perm]
                logit, c = self.forward(xb, rng=rng, train=True)
                self._adam(self._backward(c, logit, yb, self.pos_weight))
        return self

    def decision_function(self, X):
        return self.forward(X, train=False)[0]

    def predict_proba_1d(self, X):
        return 1 / (1 + np.exp(-np.clip(self.decision_function(X), -60, 60)))

class BiomarkerGraphGCN(BaseEstimator):
    """GCN over a data-driven biomarker-dependency graph."""

    def __init__(self, adjacency=None, epochs=200, seed=SEED,
                 mixup_alpha=None, mix_labels=True):
        self.__dict__.update(adjacency=adjacency, epochs=epochs, seed=seed,
                             mixup_alpha=mixup_alpha, mix_labels=mix_labels)

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.mu_, sd = X.mean(0), X.std(0)
        self.sd_ = np.where(sd > 0, sd, 1.0)
        A = self.adjacency if self.adjacency is not None else \
            np.ones((X.shape[1],) * 2) - np.eye(X.shape[1])
        Xg = biomarker_graph_features((X - self.mu_) / self.sd_)
        y = np.asarray(y, float)
        pw = (len(y) - y.sum()) / max(y.sum(), 1)
        self.net_ = GCN(normalised_adjacency(A), Xg.shape[2], epochs=self.epochs,
                        seed=self.seed, pos_weight=float(pw))
        self.net_.fit(Xg, y, self.mixup_alpha, self.mix_labels)
        return self

    def predict_proba(self, X):
        Xg = biomarker_graph_features((np.asarray(X, float) - self.mu_) / self.sd_)
        p = self.net_.predict_proba_1d(Xg)
        return np.column_stack([1 - p, p])

def partial_correlation_graph(X, groups=None, threshold=0.10):
    X = np.asarray(X, float)
    if groups is not None:
        X = X.copy()
        for g in np.unique(groups):
            s = groups == g
            X[s] -= X[s].mean(0)
    sd = X.std(0, ddof=1)
    X = X / np.where(sd > 0, sd, 1.0)
    try:
        prec = GraphicalLassoCV(alphas=6, max_iter=200).fit(X).precision_
    except Exception:
        prec = np.linalg.pinv(LedoitWolf().fit(X).covariance_)
    d = np.sqrt(np.abs(np.diag(prec))); d = np.where(d > 0, d, 1.0)
    A = np.abs(-prec / np.outer(d, d))
    np.fill_diagonal(A, 0.0)
    A[A < threshold] = 0.0
    return A

def exp2_gcn_audit(df):
    log_step("E2  two-node GCN architectural audit")
    S = two_node_operator()
    rows = []
    for c in CONDITIONS:
        d = subset(df, c)
        tau = LOEC[(c.species, c.day)]
        y = legacy_label(d, tau).astype(float)
        conc, bio = d.concentration.to_numpy(), d[BIOMARKERS].to_numpy()
        X2, u = two_node_features(conc, bio), two_node_collapsed(conc, bio)[:, None, :]
        kw = dict(d0=X2.shape[2], dropout=0.0, epochs=120, seed=0, pos_weight=1.0)
        g2 = GCN(S=S, **kw).fit(X2, y)
        g1 = GCN(S=normalised_adjacency(np.zeros((1, 1))), **kw).fit(u, y)
        z2, z1 = g2.decision_function(X2), g1.decision_function(u)
        gcn = GCN(S=S, d0=X2.shape[2], epochs=200, seed=0,
                  pos_weight=float((len(y) - y.sum()) / max(y.sum(), 1))).fit(X2, y)
        u0 = 0.5 * (conc + bio[:, 0])
        rows.append(dict(condition=c.key, max_logit_diff=float(np.max(np.abs(z2 - z1))),
                         rank_S=int(np.linalg.matrix_rank(S)),
                         gcn_resub_auc=roc_auc_score(y, gcn.predict_proba_1d(X2)),
                         trivial_auc=roc_auc_score(y, conc),
                         corr_u0_conc=float(np.corrcoef(u0, conc)[0, 1]),
                         corr_u0_hb=float(np.corrcoef(u0, bio[:, 0])[0, 1])))
    for x in rows:
        log(f"{CTEX_PLAIN[x['condition']]}: rank(S)={x['rank_S']}, "
            f"max|dlogit|={x['max_logit_diff']:.1e}, "
            f"resub AUC={x['gcn_resub_auc']:.3f}")
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "e2_gcn_audit.csv", index=False)
    write_table("t3_gcn.tex",
                [f"{CTEX[x.condition]} & {x.max_logit_diff:.1e} & {x.rank_S} & "
                 f"{x.gcn_resub_auc:.3f} & {x.trivial_auc:.3f} & "
                 f"{x.corr_u0_conc:+.2f} & {x.corr_u0_hb:+.2f} \\\\"
                 for x in r.itertuples()],
                "lcccccc",
                (r"Condition & $\max|\Delta\mathrm{logit}|$ & $\mathrm{rank}(S)$ & "
                 r"GCN resub.\ AUC & Trivial AUC & $r(u_0,c)$ & $r(u_0,\mathrm{Hb})$"))
    return r

def exp9_mixup(df, n_rep=200, alpha=0.4, seeds=(42, 43, 44)):
    log_step("E9  augmentation validity and ablation")
    rng = np.random.default_rng(0); rows = []
    for c in CONDITIONS:
        d = subset(df, c)
        csd = d[d.concentration == 0][BIOMARKERS].std(ddof=1).to_numpy()
        csd = np.where(csd > 0, csd, 1.0)
        lv = np.sort(d.concentration.unique())
        for t in range(1, len(lv) - 1):
            real = d.loc[d.concentration == lv[t], BIOMARKERS].to_numpy()
            lo = d.loc[d.concentration == lv[t - 1], BIOMARKERS].to_numpy()
            hi = d.loc[d.concentration == lv[t + 1], BIOMARKERS].to_numpy()
            if min(len(real), len(lo), len(hi)) < 3:
                continue
            lam = rng.beta(alpha, alpha, size=(n_rep, 1))
            variants = {
                "GraphMixUp (across doses)":
                    lam * lo[rng.integers(0, len(lo), n_rep)] +
                    (1 - lam) * hi[rng.integers(0, len(hi), n_rep)],
                "Neighbour resample":
                    np.vstack([lo, hi])[rng.integers(0, len(lo) + len(hi), n_rep)],
                "Within-dose mixing (oracle)":
                    lam * real[rng.integers(0, len(real), n_rep)] +
                    (1 - lam) * real[rng.integers(0, len(real), n_rep)]}
            for vn, syn in variants.items():
                w = np.array([wasserstein_distance(real[:, k], syn[:, k]) / csd[k]
                              for k in range(len(BIOMARKERS))])
                bias = (syn.mean(0) - real.mean(0)) / csd
                Xc = np.vstack([real, syn])
                yc = np.r_[np.ones(len(real)), np.zeros(len(syn))]
                ns = int(min(5, len(real), len(syn)))
                p = cross_val_predict(
                    RandomForestClassifier(n_jobs=1, n_estimators=300, min_samples_leaf=2,
                                           random_state=0), Xc, yc,
                    cv=StratifiedKFold(ns, shuffle=True, random_state=0),
                    method="predict_proba")[:, 1]
                rows.append(dict(condition=c.key, dose=float(lv[t]), variant=vn,
                                 wasserstein=float(w.mean()),
                                 abs_bias=float(np.abs(bias).mean()),
                                 two_sample_auc=float(roc_auc_score(yc, p))))
    V = pd.DataFrame(rows); V.to_csv(OUT / "e9_mixup_validity.csv", index=False)
    for v, g in V.groupby("variant"):
        kv(v, f"two-sample AUC={g.two_sample_auc.mean():.3f}  "
              f"Wasserstein={g.wasserstein.mean():.3f} SD  "
              f"|bias|={g.abs_bias.mean():.3f} SD", indent=2, width=32)
    step("does augmentation help the honest task? (3 seeds)")

    abl = []
    for c in CONDITIONS:
        d = subset(df, c)
        X, y = d[BIOMARKERS].to_numpy(), exposed_label(d)
        folds = lodgo(d.concentration.to_numpy())
        adj = [partial_correlation_graph(X[tr], d.dose_group.to_numpy()[tr])
               for tr, _ in folds]
        for sn, kw in {"no augmentation": dict(mixup_alpha=None),
                       "MixUp Beta(0.4,0.4)": dict(mixup_alpha=0.4),
                       "MixUp Beta(1.0,1.0)": dict(mixup_alpha=1.0),
                       "MixUp, features only": dict(mixup_alpha=0.4,
                                                    mix_labels=False)}.items():
            for sd in seeds:
                oof = np.full(len(y), np.nan)
                for (tr, te), A in zip(folds, adj):
                    oof[te] = BiomarkerGraphGCN(adjacency=A, seed=sd, **kw).fit(
                        X[tr], y[tr]).predict_proba(X[te])[:, 1]
                ok = ~np.isnan(oof)
                abl.append(dict(condition=c.key, setting=sn, seed=sd,
                                roc_auc=fast_auc(y[ok], oof[ok])))
    A = pd.DataFrame(abl); A.to_csv(OUT / "e9_mixup_ablation.csv", index=False)
    for st, g in A.groupby("setting"):
        kv(st, f"mean LODGO AUC={g.roc_auc.mean():.3f} "
               f"(+/-{g.roc_auc.std():.3f})", indent=2, width=32)

    order = ["Within-dose mixing (oracle)", "GraphMixUp (across doses)",
             "Neighbour resample"]
    write_table("t11_mixup.tex",
                [f"{v} & {V[V.variant==v].two_sample_auc.mean():.2f} $\\pm$ "
                 f"{V[V.variant==v].two_sample_auc.std():.2f} & "
                 f"{V[V.variant==v].wasserstein.mean():.2f} & "
                 f"{V[V.variant==v].abs_bias.mean():.2f} \\\\" for v in order],
                "lccc",
                (r"Synthesis scheme & Two-sample AUC & Wasserstein (ctrl SD) "
                 r"& $|$bias$|$ (SD)"))
    g = A.groupby(["setting", "condition"]).roc_auc.agg(["mean", "std"])
    write_table("t12_mixup_ablation.tex",
                [f"{s} & " + " & ".join(f"{g.loc[(s,k),'mean']:.3f} $\\pm$ "
                                        f"{g.loc[(s,k),'std']:.3f}" for k in CKEYS)
                 + r" \\" for s in ["no augmentation", "MixUp Beta(0.4,0.4)",
                                    "MixUp Beta(1.0,1.0)", "MixUp, features only"]],
                "lcccc", r"Setting & " + " & ".join(CTEX[k] for k in CKEYS))
    return V, A
