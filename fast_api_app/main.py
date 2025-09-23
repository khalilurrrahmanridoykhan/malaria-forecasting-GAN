"""FastAPI app exposing malaria forecasting model outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----------------------------
# Configuration
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "TrainingFile2012_2024.csv"
CLIMATE_PATH = BASE_DIR / "ClimatePredictionFile2023_2032_WithIDs.xlsx"
SEQ = 12
VAL_H_MONTHS = 6
TEST_H_MONTHS = 6
INDICATORS = [
    "Average_temperature",
    "Total_rainfall",
    "Relative_humidity",
    "Average_NDVI",
    "Average_NDWI",
]
LATENT = 12
HIDDEN_MULT = 4
DROP = 0.25
DEVICE = torch.device("cpu")

torch.set_num_threads(4)

MODEL_SPECS = {
    "target_pv": {
        "target_col": "pv_rate",
        "friendly_name": "Target_PV_over_Pop",
        "directory": BASE_DIR / "final_output/malaria_e2e/Target_PV_over_Pop",
    },
    "target_pf": {
        "target_col": "pf_rate",
        "friendly_name": "Target_PF_over_Pop",
        "directory": BASE_DIR / "final_output/malaria_e2e/Target_PF_over_Pop",
    },
    "target_mixed": {
        "target_col": "mixed_rate",
        "friendly_name": "Target_MIXED_over_Pop",
        "directory": BASE_DIR / "final_output/malaria_e2e/Target_MIXED_over_Pop",
    },
}

# ----------------------------
# Utility data classes
# ----------------------------


class Split:
    def __init__(self, X: np.ndarray, y: np.ndarray, c: np.ndarray, df: pd.DataFrame):
        self.X = X
        self.y = y
        self.c = c
        self.df = df


# ----------------------------
# Model components needed for feature generation
# ----------------------------


class CondVAE(nn.Module):
    def __init__(self, inp: int, hid: int, lat: int, n_upazila: int, n_year: int, emb: int = 16) -> None:
        super().__init__()
        self.ue = nn.Embedding(n_upazila, emb)
        self.ye = nn.Embedding(n_year, emb)
        self.p = nn.Linear(inp, hid // 2)
        self.f = nn.Linear(hid // 2 + 2 * emb, hid)
        self.mu = nn.Linear(hid, lat)
        self.lv = nn.Linear(hid, lat)
        self.fd = nn.Linear(lat + 2 * emb, hid)
        self.out = nn.Linear(hid, inp)
        self.l1 = nn.LayerNorm(hid)
        self.l2 = nn.LayerNorm(hid)
        self.drop = nn.Dropout(DROP)

    def encode(self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor):
        h = torch.relu(self.p(x))
        h = torch.cat([h, self.ue(u), self.ye(y)], 1)
        h = self.drop(torch.relu(self.l1(self.f(h))))
        return self.mu(h), self.lv(h)

    def reparam(self, m: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        std = (0.5 * l).exp()
        return m + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, self.ue(u), self.ye(y)], 1)
        h = self.drop(torch.relu(self.l2(self.fd(h))))
        return self.out(h)

    def forward(self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor):
        mu, lv = self.encode(x, u, y)
        z = self.reparam(mu, lv)
        return self.decode(z, u, y), mu, lv


class CausalTCN(nn.Module):
    def __init__(self, in_ch: int, hid: int = 64, levels: int = 3, k: int = 3):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        ch = in_ch
        for l in range(levels):
            dil = 2 ** l
            pad = (k - 1) * dil
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, hid, kernel_size=k, dilation=dil, padding=pad),
                nn.GELU()
            ))
            ch = hid

    def forward(self, x: torch.Tensor):
        y = x.transpose(1, 2)
        L = y.size(-1)
        for _, b in enumerate(self.blocks):
            y = b(y)
            y = y[..., :L]
        return y.transpose(1, 2)


class Generator(nn.Module):
    def __init__(self, cond_dim: int, noise_dim: int, nc: int, emb: int = 8, lstm: int = 128, heads: int = 8, drop: float = DROP, qu: tuple = (0.1, 0.5, 0.9)):
        super().__init__()
        self.q = qu
        self.ce = nn.Embedding(nc, emb)
        self.pn = nn.Linear(noise_dim, cond_dim)
        self.tcn = CausalTCN(cond_dim * 2 + emb, hid=lstm)
        self.lstm = nn.LSTM(lstm, lstm, 1, batch_first=True)
        self.mha = nn.MultiheadAttention(lstm, heads, batch_first=True, dropout=drop)
        self.ln = nn.LayerNorm(lstm)
        self.drop = nn.Dropout(drop)
        self.mu = nn.Linear(lstm, 1)
        self.ls = nn.Linear(lstm, 1)
        self.qh = nn.ModuleList([nn.Linear(lstm, 1) for _ in qu])
        self.last = None

    def forward(self, cond: torch.Tensor, noise: torch.Tensor, cid: torch.Tensor):
        B, L, D = cond.shape
        emb = self.ce(cid).unsqueeze(1).repeat(1, L, 1)
        z = self.pn(noise)
        h = self.tcn(torch.cat([cond, z, emb], -1))
        h, _ = self.lstm(h)
        L = h.size(1)
        mask = torch.triu(torch.ones(L, L, device=h.device, dtype=torch.bool), diagonal=1)
        att, w = self.mha(h, h, h, attn_mask=mask, need_weights=True)
        self.last = w.detach()
        h = self.drop(self.ln(att))
        mu = self.mu(h)
        ls = torch.clamp(self.ls(h), -5., 3.)
        qs = [q(h) for q in self.qh]
        return mu, ls, qs


# ----------------------------
# Data preparation helpers
# ----------------------------


def load_malaria(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    colmap = {
        "Month": "month",
        "MonthNo": "month",
        "MONTH": "month",
        "Year": "year",
        "YEAR": "year",
        "UpazilaId": "UpazilaID",
        "UPAZILAID": "UpazilaID",
    }
    for src, dst in colmap.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    df["UpazilaID"] = pd.to_numeric(df["UpazilaID"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)

    for cases_col, tgt in [("PV", "pv_rate"), ("PF", "pf_rate"), ("MIXED", "mixed_rate")]:
        if cases_col in df.columns:
            cc = pd.to_numeric(df[cases_col], errors="coerce")
            pp = pd.to_numeric(df["Population"], errors="coerce")
            m = (pp > 0) & cc.notna()
            df[tgt] = np.nan
            df.loc[m, tgt] = (cc[m] / pp[m]).astype(float)
        else:
            df[tgt] = np.nan

    df = df.sort_values(["UpazilaID", "year", "month"]).reset_index(drop=True)
    df["ym"] = df["year"] * 12 + df["month"]

    cols_to_ffill = set(INDICATORS + ["Population", "pv_rate", "pf_rate", "mixed_rate"])
    present = [c for c in cols_to_ffill if c in df.columns]
    df[present] = df.groupby("UpazilaID")[present].apply(lambda g: g.ffill()).reset_index(level=0, drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[present] = df.groupby("UpazilaID")[present].apply(lambda g: g.ffill()).reset_index(level=0, drop=True)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0).astype(np.float32)

    df["upazila_code"] = pd.Categorical(df["UpazilaID"]).codes
    df["year_code"] = pd.Categorical(df["year"]).codes
    return df


def vae_latent_df(vae: CondVAE, df: pd.DataFrame) -> pd.DataFrame:
    vae.eval()
    X = torch.tensor(df[INDICATORS].astype(np.float32).values, device=DEVICE)
    U = torch.tensor(df["upazila_code"].astype(np.int64).values, device=DEVICE)
    Y = torch.tensor(df["year_code"].astype(np.int64).values, device=DEVICE)
    with torch.no_grad():
        _, mu, _ = vae(X, U, Y)
    Z = mu.cpu().numpy().astype(np.float32)
    out = df[["UpazilaID", "year", "month", "ym"]].copy()
    for i in range(LATENT):
        out[f"pc_mean_{i+1}"] = Z[:, i]
        out[f"pc_std_{i+1}"] = 0.0
    out["month_sin"] = df["month_sin"].values.astype(np.float32)
    out["month_cos"] = df["month_cos"].values.astype(np.float32)
    return out


def add_lags(df: pd.DataFrame, base_cols: List[str], group_col: str = "UpazilaID", lags=(1, 3, 6), rolls=(3, 6)):
    df = df.sort_values([group_col, "year", "month"]).copy()
    new_cols: List[str] = []
    for c in base_cols:
        for L in lags:
            col = f"{c}_lag{L}"
            df[col] = df.groupby(group_col)[c].shift(L)
            new_cols.append(col)
        for R in rolls:
            m = f"{c}_rmean{R}"
            s = f"{c}_rstd{R}"
            g = df.groupby(group_col)[c]
            df[m] = g.rolling(R, min_periods=1).mean().reset_index(level=0, drop=True)
            df[s] = g.rolling(R, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
            new_cols.extend([m, s])
    df[new_cols] = df[new_cols].fillna(0.0)
    return df, new_cols


def build_mats(df: pd.DataFrame, feat_cols: List[str], target_col: str, entity_col: str = "UpazilaID"):
    le = LabelEncoder().fit(df[entity_col].astype(str).values)
    d = df.copy()
    d["entity_id"] = le.transform(d[entity_col].astype(str))

    X = d[feat_cols].values.astype(np.float32)
    feat_scaler = StandardScaler().fit(X)
    Xs = feat_scaler.transform(X)

    y = np.log1p(np.clip(d[target_col].values.reshape(-1, 1), 0, None)).astype(np.float32)
    Ys = np.zeros_like(y, np.float32)

    y_scalers: Dict[int, StandardScaler] = {}
    for cid, g in d.groupby("entity_id"):
        idx = g.index.values
        sc = StandardScaler().fit(y[idx])
        y_scalers[int(cid)] = sc
        Ys[idx] = sc.transform(y[idx])

    return Split(Xs, Ys, d["entity_id"].values.astype(np.int64), d), le, feat_scaler, y_scalers


def apply_mats(df: pd.DataFrame, feat_cols: List[str], target_col: str, le: LabelEncoder,
               feat_scaler: StandardScaler, y_scalers: Dict[int, StandardScaler],
               entity_col: str = "UpazilaID"):
    d = df.copy()
    d["entity_id"] = le.transform(d[entity_col].astype(str))
    X = d[feat_cols].values.astype(np.float32)
    Xs = feat_scaler.transform(X)

    y = np.log1p(np.clip(d[target_col].values.reshape(-1, 1), 0, None)).astype(np.float32)
    Ys = np.zeros_like(y, np.float32)
    for cid, g in d.groupby("entity_id"):
        idx = g.index.values
        sc = y_scalers.get(int(cid))
        if sc is None:
            sc = StandardScaler().fit(y[idx])
            y_scalers[int(cid)] = sc
        Ys[idx] = sc.transform(y[idx])
    return Split(Xs, Ys, d["entity_id"].values.astype(np.int64), d)


def to_seq_with_meta(split: Split, L: int = SEQ, prev_y: bool = True):
    X, y, c, df = split.X, split.y, split.c, split.df
    ym = (df["year"].astype(int) * 12 + df["month"].astype(int)).to_numpy()
    SX: List[np.ndarray] = []
    SY: List[np.ndarray] = []
    SC: List[int] = []
    meta: List[Dict[str, object]] = []

    unique_c = np.unique(c)
    for cid in unique_c:
        idx = np.where(c == cid)[0]
        ordered = idx[np.argsort(ym[idx])]
        for i in range(len(ordered) - L + 1):
            window_idx = ordered[i:i + L]
            if not np.all(np.diff(ym[window_idx]) == 1):
                continue
            Xi = X[window_idx]
            Yi = y[window_idx]
            if prev_y:
                prev = np.vstack([np.zeros((1, 1), np.float32), Yi[:-1]])
                Xi = np.concatenate([Xi, prev], 1)
            SX.append(Xi)
            SY.append(Yi)
            SC.append(int(cid))
            rows = df.iloc[window_idx]
            meta.append({
                "entity_id": int(cid),
                "upazila_id": int(rows["UpazilaID"].iloc[0]) if not rows["UpazilaID"].isna().all() else None,
                "years": rows["year"].astype(int).tolist(),
                "months": rows["month"].astype(int).tolist(),
            })
    return (
        np.asarray(SX, np.float32),
        np.asarray(SY, np.float32),
        np.asarray(SC, np.int64),
        meta,
    )


# ----------------------------
# Loading stored model outputs
# ----------------------------


def load_target_artifacts(df_raw: pd.DataFrame, spec: Dict[str, object]):
    target_dir = spec["directory"]
    target_col = spec["target_col"]
    friendly_name = spec["friendly_name"]

    if not target_dir.exists():
        raise FileNotFoundError(f"Missing target directory: {target_dir}")

    manifest_path = target_dir / "manifest.json"
    pred_path = target_dir / "predictions_test.npz"
    vae_path = target_dir / "conditional_vae.pt"
    gen_path = target_dir / "generator.pt"

    if not manifest_path.exists() or not pred_path.exists() or not vae_path.exists() or not gen_path.exists():
        raise FileNotFoundError(f"Required artifact missing in {target_dir}")

    manifest = json.loads(manifest_path.read_text())
    preds = np.load(pred_path)

    df = df_raw.copy()
    n_upazila = int(df["upazila_code"].max()) + 1
    n_year = int(df["year_code"].max()) + 1
    inp = len(INDICATORS)
    hid = HIDDEN_MULT * inp

    vae = CondVAE(inp, hid, LATENT, n_upazila, n_year).to(DEVICE)
    vae.load_state_dict(torch.load(vae_path, map_location=DEVICE))
    vae.eval()

    lat = vae_latent_df(vae, df)
    base = lat.merge(
        df[["UpazilaID", "year", "month", "ym"] + INDICATORS + [target_col]],
        on=["UpazilaID", "year", "month", "ym"],
        how="left",
        validate="1:1",
    )

    base, lag_cols = add_lags(base, INDICATORS, group_col="UpazilaID", lags=(1, 3, 6), rolls=(3, 6))
    pc_cols = [c for c in base.columns if c.startswith("pc_mean_") or c.startswith("pc_std_")]
    feats = pc_cols + ["month_sin", "month_cos"] + lag_cols

    tr_parts: List[pd.DataFrame] = []
    va_parts: List[pd.DataFrame] = []
    te_parts: List[pd.DataFrame] = []
    for uid, g in base.groupby("UpazilaID"):
        g = g.sort_values(["year", "month"]).copy()
        if len(g) > (VAL_H_MONTHS + TEST_H_MONTHS):
            te_parts.append(g.iloc[-TEST_H_MONTHS:])
            va_parts.append(g.iloc[-(VAL_H_MONTHS + TEST_H_MONTHS):-TEST_H_MONTHS])
            tr_parts.append(g.iloc[:-(VAL_H_MONTHS + TEST_H_MONTHS)])
        elif len(g) > TEST_H_MONTHS:
            te_parts.append(g.iloc[-TEST_H_MONTHS:])
            tr_parts.append(g.iloc[:-TEST_H_MONTHS])
        else:
            tr_parts.append(g)

    tr_df = pd.concat(tr_parts).reset_index(drop=True)
    va_df = pd.concat(va_parts).reset_index(drop=True) if va_parts else tr_df.iloc[0:0].copy()
    te_df = pd.concat(te_parts).reset_index(drop=True) if te_parts else tr_df.iloc[0:0].copy()

    split_tr, le_tr, fsc_tr, ysc_tr = build_mats(tr_df, feats, target_col, entity_col="UpazilaID")
    split_te = apply_mats(te_df, feats, target_col, le_tr, fsc_tr, ysc_tr, "UpazilaID")
    X_te, Y_te, C_te, meta = to_seq_with_meta(split_te, SEQ, True)

    # Load Generator
    nc = int(split_tr.c.max()) + 1
    cond_dim = split_tr.X.shape[1] + 1
    gen = Generator(cond_dim=cond_dim, noise_dim=16, nc=nc).to(DEVICE)
    gen.load_state_dict(torch.load(gen_path, map_location=DEVICE))
    gen.eval()

    mf = preds["mf"].reshape(-1, SEQ)
    lo = preds["lo"].reshape(-1, SEQ)
    hi = preds["hi"].reshape(-1, SEQ)
    y = preds["y"].reshape(-1, SEQ)
    c_saved = preds["C_te"].astype(np.int64)

    if len(meta) != mf.shape[0]:
        print(f"Warning: Mismatch between stored predictions ({mf.shape[0]}) and generated metadata ({len(meta)}), skipping stored predictions")
        predictions_df = pd.DataFrame()
    elif not np.array_equal(c_saved, C_te.astype(np.int64)):
        # Align using ordering just in case of minor differences
        c_saved = C_te.astype(np.int64)
        records = []
        for seq_idx, info in enumerate(meta):
            upazila_id = info.get("upazila_id")
            years = info["years"]
            months = info["months"]
            for step in range(SEQ):
                records.append({
                    "sequence_index": seq_idx,
                    "position": step,
                    "upazila_id": upazila_id,
                    "year": int(years[step]),
                    "month": int(months[step]),
                    "date": f"{int(years[step])}-{int(months[step]):02d}",
                    "prediction": float(mf[seq_idx, step]),
                    "lower": float(lo[seq_idx, step]),
                    "upper": float(hi[seq_idx, step]),
                    "actual": float(y[seq_idx, step]),
                })

        predictions_df = pd.DataFrame.from_records(records)
        predictions_df.sort_values(["upazila_id", "year", "month", "position"], inplace=True)
        predictions_df.reset_index(drop=True, inplace=True)
    else:
        records = []
        for seq_idx, info in enumerate(meta):
            upazila_id = info.get("upazila_id")
            years = info["years"]
            months = info["months"]
            for step in range(SEQ):
                records.append({
                    "sequence_index": seq_idx,
                    "position": step,
                    "upazila_id": upazila_id,
                    "year": int(years[step]),
                    "month": int(months[step]),
                    "date": f"{int(years[step])}-{int(months[step]):02d}",
                    "prediction": float(mf[seq_idx, step]),
                    "lower": float(lo[seq_idx, step]),
                    "upper": float(hi[seq_idx, step]),
                    "actual": float(y[seq_idx, step]),
                })

        predictions_df = pd.DataFrame.from_records(records)
        predictions_df.sort_values(["upazila_id", "year", "month", "position"], inplace=True)
        predictions_df.reset_index(drop=True, inplace=True)

    artifacts = {
        "id": friendly_name,
        "target_col": target_col,
        "metrics": manifest.get("metrics_test", {}),
        "config": manifest.get("config", {}),
        "notes": manifest.get("notes"),
        "predictions": predictions_df,
        "available_upazilas": [int(uid) for uid in le_tr.classes_],
        "vae": vae,
        "gen": gen,
        "le": le_tr,
        "fsc": fsc_tr,
        "ysc": ysc_tr,
        "feats": feats,
    }
    return artifacts


def predict_rate(artifacts: Dict[str, object], input_data: PredictionInput) -> float:
    vae = artifacts["vae"]
    gen = artifacts["gen"]
    le = artifacts["le"]
    fsc = artifacts["fsc"]
    ysc = artifacts["ysc"]
    feats = artifacts["feats"]
    target_col = artifacts["target_col"]
    climate_df = artifacts["climate_df"]

    # Fetch last 12 months data for the upazila
    upazila_data = climate_df[climate_df["UpazilaID"] == input_data.upazila_id].copy()
    if upazila_data.empty:
        raise ValueError(f"No climate data for upazila {input_data.upazila_id}")
    upazila_data = upazila_data.sort_values(["Year", "Month"]).reset_index(drop=True)
    # Find the index of the given month
    target_ym = input_data.year * 12 + input_data.month
    idx = upazila_data[upazila_data["ym"] == target_ym].index
    if idx.empty:
        available_dates = upazila_data["ym"].tolist()
        raise ValueError(f"No data for {input_data.year}-{input_data.month:02d} (target_ym={target_ym}). Available ym: {sorted(available_dates)[:5]}...")
    idx = idx[0]
    # Take the 12 months before
    start_idx = max(0, idx - 12)
    last_12 = upazila_data.iloc[start_idx:idx]
    if len(last_12) < 12:
        available_count = len(last_12)
        date_range = f"{last_12.iloc[0]['Year']}-{last_12.iloc[0]['Month']:02d} to {last_12.iloc[-1]['Year']}-{last_12.iloc[-1]['Month']:02d}" if not last_12.empty else "none"
        raise ValueError(f"Not enough historical data (need 12 months, have {available_count}). Available range: {date_range}")
    # Prepare df
    df = last_12.copy()
    df["month"] = df["Month"]
    df["year"] = df["Year"]
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0).astype(np.float32)
    # Dummy codes
    df["upazila_code"] = 0
    df["year_code"] = 0

    # Get latents
    lat = vae_latent_df(vae, df)
    base = lat.merge(df[["UpazilaID", "year", "month", "ym"] + INDICATORS], on=["UpazilaID", "year", "month", "ym"], how="left")
    # Add lags
    base, _ = add_lags(base, INDICATORS, group_col="UpazilaID", lags=(1, 3, 6), rolls=(3, 6))
    # Feats
    pc_cols = [c for c in base.columns if c.startswith("pc_mean_") or c.startswith("pc_std_")]
    # feats should match

    # Take last SEQ
    g = base.sort_values(["year", "month"]).copy()
    last_seq = g.iloc[-SEQ:].copy()
    X = last_seq[feats].values.astype(np.float32)
    Xs = fsc.transform(X)
    prev_y = np.zeros((SEQ, 1), np.float32)
    X_full = np.concatenate([Xs, prev_y], axis=1)

    X_t = torch.tensor(X_full[np.newaxis, :, :], device=DEVICE)
    C_t = torch.tensor([le.transform([str(input_data.upazila_id)])[0]], device=DEVICE)
    with torch.no_grad():
        mu, ls, qs = gen(X_t, torch.zeros(1, SEQ, 16, device=DEVICE), C_t)
    pred_scaled = mu.cpu().numpy()[0, -1, 0]
    pred_rate = np.expm1(ysc[int(C_t.item())].inverse_transform([[pred_scaled]])[0, 0])

    # Compute quantiles
    q10_scaled = qs[0].cpu().numpy()[0, -1, 0]
    q90_scaled = qs[2].cpu().numpy()[0, -1, 0]
    lower = np.expm1(ysc[int(C_t.item())].inverse_transform([[q10_scaled]])[0, 0])
    upper = np.expm1(ysc[int(C_t.item())].inverse_transform([[q90_scaled]])[0, 0])

    return pred_rate, lower, upper


# ----------------------------
# FastAPI application setup
# ----------------------------


class ModelSummary(BaseModel):
    id: str
    target_col: str
    friendly_name: str
    metrics: Dict[str, float]
    config: Dict[str, float]
    notes: Optional[str]
    upazila_ids: List[int]
    num_predictions: int


class PredictionsResponse(BaseModel):
    total: int
    count: int
    items: List[Dict[str, object]]


class PredictAllInput(BaseModel):
    year: int
    month: int


class PredictionInput(BaseModel):
    upazila_id: int
    year: int
    month: int


class PredictionOutput(BaseModel):
    predicted_rate: float
    lower_bound: float
    upper_bound: float


app = FastAPI(title="Malaria Forecast API", version="0.1.0")


def load_climate_data():
    df = pd.read_excel(CLIMATE_PATH)
    df.columns = [c.strip() for c in df.columns]
    # Clean Year
    df["Year"] = df["Year"].astype(str).str.replace(",", "").astype(int)
    # Map month names to numbers
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }
    df["Month"] = df["Month"].map(month_map)
    # Columns are already correctly named in the Excel file
    # Rename upazila_id to UpazilaID for consistency
    if "upazila_id" in df.columns:
        df = df.rename(columns={"upazila_id": "UpazilaID"})
    df["ym"] = df["Year"] * 12 + df["Month"]
    df = df.sort_values(["UpazilaID", "Year", "Month"]).reset_index(drop=True)
    return df


def load_registry():
    df_raw = load_malaria(DATA_PATH)
    climate_df = load_climate_data()
    registry = {}
    for key, spec in MODEL_SPECS.items():
        try:
            artifacts = load_target_artifacts(df_raw.copy(), spec)
            artifacts["climate_df"] = climate_df
            registry[key] = artifacts
        except ValueError as e:
            print(f"Warning: Failed to load {key}: {e}")
            # skip this model
    return registry


@app.on_event("startup")
def startup_event():
    app.state.registry = load_registry()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models", response_model=List[ModelSummary])
def list_models():
    registry = getattr(app.state, "registry", {})
    summaries: List[ModelSummary] = []
    for key, spec in MODEL_SPECS.items():
        artifacts = registry.get(key)
        if artifacts is None:
            continue
        df = artifacts["predictions"]
        summaries.append(ModelSummary(
            id=key,
            target_col=artifacts["target_col"],
            friendly_name=spec["friendly_name"],
            metrics=artifacts["metrics"],
            config=artifacts["config"],
            notes=artifacts["notes"],
            upazila_ids=artifacts["available_upazilas"],
            num_predictions=int(len(df)),
        ))
    return summaries


@app.get("/models/{model_id}", response_model=ModelSummary)
def get_model(model_id: str):
    registry = getattr(app.state, "registry", {})
    if model_id not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="Unknown model id")
    artifacts = registry.get(model_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail="Artifacts not loaded")
    df = artifacts["predictions"]
    spec = MODEL_SPECS[model_id]
    return ModelSummary(
        id=model_id,
        target_col=artifacts["target_col"],
        friendly_name=spec["friendly_name"],
        metrics=artifacts["metrics"],
        config=artifacts["config"],
        notes=artifacts["notes"],
        upazila_ids=artifacts["available_upazilas"],
        num_predictions=int(len(df)),
    )


@app.get("/models/{model_id}/predictions", response_model=PredictionsResponse)
def get_predictions(
    model_id: str,
    upazila_id: Optional[int] = Query(None, description="Filter by UpazilaID"),
    year: Optional[int] = Query(None, description="Filter by year"),
    month: Optional[int] = Query(None, ge=1, le=12, description="Filter by month"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    registry = getattr(app.state, "registry", {})
    if model_id not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="Unknown model id")
    artifacts = registry.get(model_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail="Artifacts not loaded")

    df = artifacts["predictions"]

    # If no stored predictions available, return empty response
    if df.empty:
        return PredictionsResponse(total=0, count=0, items=[])

    filtered = df
    if upazila_id is not None:
        filtered = filtered[filtered["upazila_id"] == upazila_id]
    if year is not None:
        filtered = filtered[filtered["year"] == year]
    if month is not None:
        filtered = filtered[filtered["month"] == month]

    total = int(len(filtered))
    if offset >= total:
        items: List[Dict[str, object]] = []
    else:
        window = filtered.iloc[offset: offset + limit]
        items = window.to_dict(orient="records")
    return PredictionsResponse(total=total, count=len(items), items=items)


@app.get("/models/{model_id}/upazilas")
def list_upazilas(model_id: str):
    registry = getattr(app.state, "registry", {})
    if model_id not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="Unknown model id")
    artifacts = registry.get(model_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail="Artifacts not loaded")
    return {"model_id": model_id, "upazila_ids": artifacts["available_upazilas"]}


@app.post("/models/{model_id}/predict_all", response_model=List[Dict[str, object]])
def predict_all(model_id: str, input_data: PredictAllInput):
    registry = getattr(app.state, "registry", {})
    if model_id not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="Unknown model id")
    artifacts = registry.get(model_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail="Artifacts not loaded")

    results = []
    for uid in artifacts["available_upazilas"]:
        try:
            pred_input = PredictionInput(upazila_id=uid, year=input_data.year, month=input_data.month)
            pred_rate, lower, upper = predict_rate(artifacts, pred_input)
            results.append({
                "upazila_id": uid,
                "predicted_rate": pred_rate
            })
        except Exception as e:
            results.append({
                "upazila_id": uid,
                "error": str(e)
            })
    return results


@app.post("/models/{model_id}/predict")
def predict_single(model_id: str, input_data: PredictionInput):
    registry = getattr(app.state, "registry", {})
    if model_id not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="Unknown model id")
    artifacts = registry.get(model_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail="Artifacts not loaded")

    try:
        pred_rate, lower, upper = predict_rate(artifacts, input_data)
        return {
            "upazila_id": input_data.upazila_id,
            "predicted_rate": pred_rate
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Convenience for running with `python -m fast_api_app.main`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fast_api_app.main:app", host="0.0.0.0", port=8000, reload=True)
