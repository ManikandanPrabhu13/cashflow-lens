"""Synthetic longitudinal MSME data generator.

*** ALL DATA IS SYNTHETIC. No real bank, UPI, GST or government data is used or accessed. ***

Timeline (months, from ``data.start_date``):
    [0, observation_months)      -> observation window: everything a lender could see
    application_date             -> first day after the observation window
    [observation_months, end)    -> outcome window: used ONLY to create the target

Target ``default_flag`` = 1 if the borrower misses two or more consecutive EMIs of the
loan disbursed at ``application_date`` inside the outcome window. It is drawn from a latent
probability driven by several noisy drivers (leverage, volatility, discipline, thin credit
history, controlled evidence-inconsistency scenarios, noise). It is NOT a deterministic
function of any observable feature. Latent variables are never written to the output.

Anti-leakage notes for downstream code:
    * Only rows dated before ``application_date`` may be used to build features.
    * Settlements/filings dated after ``application_date`` are not yet visible to a lender.
    * ``scenarios`` (controlled inconsistency labels) exist for demos/tests only and must
      never be used as model features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import DISCLAIMER_SYNTHETIC, get_paths, get_seed, get_settings
from src.preprocessing import DATE_COLUMNS
from src.utils import get_logger

logger = get_logger(__name__)

TABLE_NAMES = [
    "borrowers", "bank_transactions", "upi_transactions", "invoices", "vendors",
    "inventory", "gst_records", "loans", "repayments", "scenarios",
]

SCENARIO_TYPES = [
    "unsettled_invoices", "gst_revenue_mismatch", "duplicate_invoices",
    "vendor_identity_mismatch", "inventory_shortfall", "pre_application_inflow_spike",
    "round_trip_transfers",
]

SIZE_PARAMS = {
    "Micro": dict(rev_median=3.0e5, inv_per_month=4, cust_range=(5, 16), history_range=(0, 48),
                  upi_beta=(3.0, 3.0), upi_max=200_000),
    "Small": dict(rev_median=2.0e6, inv_per_month=8, cust_range=(10, 31), history_range=(6, 84),
                  upi_beta=(2.5, 3.5), upi_max=500_000),
    "Medium": dict(rev_median=9.0e6, inv_per_month=14, cust_range=(20, 51), history_range=(12, 120),
                   upi_beta=(2.0, 4.0), upi_max=1_000_000),
}
SECTOR_PARAMS = {
    "Retail": dict(cogs=0.74, opex=0.14, season_amp=0.15, goods=True, upi_bonus=0.10, gst_rate=0.18),
    "Manufacturing": dict(cogs=0.68, opex=0.15, season_amp=0.10, goods=True, upi_bonus=0.0, gst_rate=0.18),
    "Services": dict(cogs=0.35, opex=0.35, season_amp=0.06, goods=False, upi_bonus=0.05, gst_rate=0.18),
    "Agri-processing": dict(cogs=0.72, opex=0.12, season_amp=0.30, goods=True, upi_bonus=-0.05, gst_rate=0.05),
    "Logistics": dict(cogs=0.60, opex=0.25, season_amp=0.08, goods=False, upi_bonus=0.0, gst_rate=0.12),
    "Hospitality": dict(cogs=0.38, opex=0.40, season_amp=0.22, goods=False, upi_bonus=0.10, gst_rate=0.12),
}
# Illustrative structural differences that give the fairness audit something real to measure.
REGION_RISK_OFFSET = {"North": 0.0, "South": -0.10, "East": 0.20, "West": 0.0, "Central": 0.15}
PRIOR_LOAN_PROBS = {"Thin": [0.60, 0.35, 0.05], "Moderate": [0.30, 0.50, 0.20], "Deep": [0.15, 0.50, 0.35]}

_FIRST = ["Sri", "Shree", "Om", "Annapurna", "Balaji", "Kaveri", "Ganga", "Lotus", "Anand", "Sai",
          "Royal", "National", "Modern", "Bharat", "Vinayaka", "Surya", "Metro", "Green", "Prime", "Excel"]
_MID = ["Lakshmi", "Narmada", "Vaigai", "Krishna", "Godavari", "Sutlej", "Tapti", "Periyar", "Mahi", "Pennar"]
_LAST = ["Textiles", "Foods", "Traders", "Agencies", "Enterprises", "Works", "Logistics", "Supplies",
         "Industries", "Stores", "Solutions", "Mart", "Distributors", "Engineering", "Packaging", "Exports"]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _make_names(rng: np.random.Generator, n: int) -> List[str]:
    total = len(_FIRST) * len(_MID) * len(_LAST)
    idx = rng.choice(total, size=n, replace=n > total)
    names = []
    for i in idx:
        i = int(i)
        a = i // (len(_MID) * len(_LAST))
        b = (i // len(_LAST)) % len(_MID)
        c = i % len(_LAST)
        names.append(f"{_FIRST[a]} {_MID[b]} {_LAST[c]}")
    return names


def _annuity_pv(emi: float, annual_rate: float, tenure: int) -> float:
    i = annual_rate / 12.0
    return float(emi * (1 - (1 + i) ** (-tenure)) / i)


def _norm_probs(d: Dict[str, float]) -> Tuple[List[str], np.ndarray]:
    keys = list(d.keys())
    p = np.array([d[k] for k in keys], dtype=float)
    return keys, p / p.sum()


def _txn_frame(rng, day, amount, direction, category, cp_id, cp_name, reference, link,
               mode, channel, status="SUCCESS", odd_hours=False) -> pd.DataFrame:
    """Build a transaction frame keyed by integer day offsets (converted to dates later)."""
    day = np.asarray(day, dtype=int)
    n = len(day)
    ch = np.broadcast_to(np.asarray(channel), (n,))
    hour = np.where(ch == "UPI", rng.integers(7, 23, n), rng.integers(9, 19, n))
    if odd_hours:
        hour = rng.choice([22, 23, 0, 1, 2, 3, 4], size=n)
    return pd.DataFrame({
        "day": day, "txn_hour": hour, "amount": np.round(np.asarray(amount, dtype=float), 2),
        "direction": direction, "txn_type": category, "payment_mode": mode,
        "counterparty_id": cp_id, "counterparty_name": cp_name, "reference": reference,
        "link_invoice_number": link, "channel": ch, "status": status,
    })


def _settlement(rng, inv_day, total, delay, inv_no, cp_id, cp_name, direction, category,
                upi_share, upi_max, H, partial_p, tds_p, blank_p) -> List[pd.DataFrame]:
    """Create settlement transactions (full, partial, or 1%-TDS-short) for a set of invoices."""
    n = len(inv_day)
    u = rng.random(n)
    partial = u < partial_p
    tds = (u >= partial_p) & (u < partial_p + tds_p)
    first = np.where(partial, total * rng.uniform(0.4, 0.7, n), np.where(tds, total * 0.99, total))
    day1 = inv_day + delay
    use_upi = (total <= upi_max) & (rng.random(n) < upi_share)
    ref = np.where(rng.random(n) < blank_p, "", inv_no)
    parts = [(day1, first, np.arange(n))]
    pidx = np.where(partial)[0]
    if pidx.size:
        parts.append((day1[pidx] + rng.integers(5, 26, pidx.size), total[pidx] - first[pidx], pidx))
    frames = []
    for d, a, idx in parts:
        keep = d < H
        if not keep.any():
            continue
        idx, d, a = idx[keep], d[keep], a[keep]
        upi = use_upi[idx]
        bank_mode = rng.choice(["NEFT", "IMPS", "CHEQUE"], size=len(idx), p=[0.55, 0.25, 0.20])
        mode = np.where(upi, "UPI", np.where(a >= 200000, "RTGS", bank_mode))
        frames.append(_txn_frame(rng, d, a, direction, category, cp_id[idx], cp_name[idx],
                                 ref[idx], inv_no[idx], mode, np.where(upi, "UPI", "BANK")))
    return frames


# --------------------------------------------------------------------------- #
# Population
# --------------------------------------------------------------------------- #
def _draw_population(rng: np.random.Generator, n: int, dcfg: Dict[str, Any], obs: int,
                     outcome: int) -> List[Dict[str, Any]]:
    sizes_k, sizes_p = _norm_probs(dcfg["enterprise_sizes"])
    sect_k, sect_p = _norm_probs(dcfg["sectors"])
    reg_k, reg_p = _norm_probs(dcfg["regions"])
    sizes = rng.choice(sizes_k, size=n, p=sizes_p)
    sectors = rng.choice(sect_k, size=n, p=sect_p)
    regions = rng.choice(reg_k, size=n, p=reg_p)
    biz = _make_names(rng, n)
    scen_rate = float(dcfg["inconsistency_scenario_rate"])
    profiles: List[Dict[str, Any]] = []
    for i in range(n):
        size, sector, region = str(sizes[i]), str(sectors[i]), str(regions[i])
        sp, sc = SIZE_PARAMS[size], SECTOR_PARAMS[sector]
        R0 = float(rng.lognormal(np.log(sp["rev_median"]), 0.45))
        cogs = float(np.clip(sc["cogs"] * rng.uniform(0.93, 1.07), 0.2, 0.9))
        opex = float(np.clip(sc["opex"] * rng.uniform(0.90, 1.10), 0.05, 0.6))
        if 1 - cogs - opex < 0.06:
            cogs = 1 - opex - 0.06
        S = R0 * (1 - cogs - opex)  # mean monthly operating surplus before debt service
        lo, hi = sp["history_range"]
        hist = int(rng.integers(lo, hi + 1))
        band = "Thin" if hist < 12 else ("Moderate" if hist < 36 else "Deep")
        upi_share = float(np.clip(rng.beta(*sp["upi_beta"]) + sc["upi_bonus"], 0.02, 0.95))
        level = "Low" if upi_share < 0.25 else ("Medium" if upi_share < 0.5 else "High")
        disc = float(rng.beta(5, 2))
        gst_comp = float(rng.beta(6, 1.5))
        vol = float(rng.uniform(0.05, 0.30))
        n_prior = int(rng.choice([0, 1, 2], p=PRIOR_LOAN_PROBS[band]))
        prior_shares = rng.uniform(0.10, 0.35, n_prior)
        app_share = float(rng.uniform(0.15, 0.55))
        lev = float(prior_shares.sum() + app_share)  # total EMI / operating surplus
        scen: List[str] = []
        if rng.random() < scen_rate:
            allowed = [s for s in SCENARIO_TYPES if sc["goods"] or s != "inventory_shortfall"]
            k = 1 if rng.random() < 0.7 else 2
            scen = [str(x) for x in rng.choice(allowed, size=k, replace=False)]
        z = (-2.6 + 1.1 * (lev - 0.6) / 0.3 + 0.9 * (vol - 0.175) / 0.07 - 1.0 * (disc - 0.714) / 0.16
             + 0.35 * (band == "Thin") + 0.25 * (size == "Micro") + REGION_RISK_OFFSET[region]
             + 0.5 * bool(scen) + rng.normal(0, 0.7))
        default = bool(rng.random() < _sigmoid(z))
        profiles.append(dict(
            borrower_id=f"B{i + 1:04d}", business_name=biz[i] if n <= len(biz) else f"{biz[i]} {i}",
            enterprise_size=size, sector=sector, region=region, credit_history_months=hist,
            credit_history_band=band, digital_adoption_level=level, upi_share=upi_share,
            R0=R0, cogs=cogs, opex_ratio=opex, S=S, disc=disc, gst_comp=gst_comp, vol=vol,
            growth=float(rng.normal(0.004, 0.008)), season_amp=sc["season_amp"],
            season_phase=float(rng.uniform(0, 12)), conc_alpha=float(rng.uniform(0.5, 1.6)),
            unit_cost=float(rng.uniform(40, 900)), gst_rate=sc["gst_rate"],
            inventory_tracked=bool(sc["goods"]), goods=bool(sc["goods"]),
            inv_per_month=sp["inv_per_month"], cust_range=sp["cust_range"], upi_max=sp["upi_max"],
            n_prior=n_prior, prior_shares=prior_shares, app_share=app_share,
            scenarios=scen, default_flag=int(default),
            deteriorating=bool(default and rng.random() < 0.55),
            first_miss=int(rng.integers(1, max(2, outcome - 1))) if default else 0,
            incorporation_year=int(pd.Timestamp(dcfg["start_date"]).year - hist // 12 - rng.integers(0, 6)),
        ))
    return profiles


# --------------------------------------------------------------------------- #
# Scenario injection (controlled evidence inconsistencies)
# --------------------------------------------------------------------------- #
def _apply_scenarios(scen, inv, txn, gst, invm, p, app_day, rng, bid):
    win_lo = app_day - 270
    order = {s: i for i, s in enumerate(SCENARIO_TYPES)}
    for s in sorted(scen, key=lambda x: order[x]):
        if s == "unsettled_invoices":
            sel = inv[(inv.invoice_type == "SALE") & inv.day.between(app_day - 180, app_day - 75)]
            drop = sel.loc[rng.random(len(sel)) < 0.35, "invoice_number"]
            txn = txn[~(txn.link_invoice_number.isin(drop) & (txn.direction == "CREDIT"))]
        elif s == "gst_revenue_mismatch" and len(gst):
            r = rng.random(len(gst))
            in_win = ((gst.direction == "OUTWARD") & (gst.day >= win_lo) & (gst.day < app_day)).to_numpy()
            factor = np.where(in_win & (r < 0.45), rng.uniform(0.5, 0.8, len(gst)), 1.0)
            gst = gst.copy()
            gst["taxable_value"] = np.round(gst["taxable_value"] * factor, 2)
            gst["tax_amount"] = np.round(gst["tax_amount"] * factor, 2)
            gst = gst[~(in_win & (r >= 0.45) & (r < 0.60))]
        elif s == "vendor_identity_mismatch":
            pur = inv[(inv.invoice_type == "PURCHASE") & inv.day.between(win_lo, app_day - 1)]
            idx = pur.index[rng.random(len(pur)) < 0.35]
            if len(idx):
                names, ids = _make_names(rng, 2), [f"{bid}-V90", f"{bid}-V91"]
                choose = rng.integers(0, 2, len(idx))
                inv.loc[idx, "counterparty_id"] = np.array(ids)[choose]
                inv.loc[idx, "counterparty_name"] = np.array(names)[choose]
        elif s == "inventory_shortfall" and len(invm):
            pur_nos = inv[(inv.invoice_type == "PURCHASE") & inv.day.between(win_lo, app_day - 1)]["invoice_number"]
            sel = (invm["reference"].isin(pur_nos) & (invm.movement_type == "PURCHASE_IN")).to_numpy()
            r = rng.random(len(invm))
            factor = np.where(sel & (r < 0.35), rng.uniform(0.3, 0.6, len(invm)), 1.0)
            invm = invm.copy()
            invm["quantity"] = np.round(invm["quantity"] * factor, 2)
            invm = invm[~(sel & (r >= 0.35) & (r < 0.48))]
        elif s == "pre_application_inflow_spike":
            k = int(rng.integers(4, 9))
            names = _make_names(rng, k)
            amt = np.round(p["R0"] * rng.uniform(0.3, 0.9, k), -3)
            day = app_day - rng.integers(1, 46, k)
            txn = pd.concat([txn, _txn_frame(
                rng, day, amt, "CREDIT", "TRANSFER", [f"{bid}-X{j:02d}" for j in range(k)],
                [x.upper() for x in names], "", "", "NEFT", "BANK", odd_hours=True)], ignore_index=True)
        elif s == "round_trip_transfers":
            names = _make_names(rng, 2)
            ids = [f"{bid}-R0", f"{bid}-R1"]
            for _ in range(int(rng.integers(3, 7))):
                c = int(rng.integers(0, 2))
                d0 = int(app_day - rng.integers(20, 260))
                a = float(np.round(p["R0"] * rng.uniform(0.15, 0.5), -3))
                back = a * (1 + rng.uniform(-0.003, 0.003))
                txn = pd.concat([
                    txn,
                    _txn_frame(rng, [d0], [a], "DEBIT", "TRANSFER", ids[c], names[c].upper(), "", "", "NEFT", "BANK"),
                    _txn_frame(rng, [d0 + int(rng.integers(1, 11))], [back], "CREDIT", "TRANSFER", ids[c],
                               names[c].upper(), "", "", "NEFT", "BANK"),
                ], ignore_index=True)
        elif s == "duplicate_invoices":
            pool = inv[inv.day.between(win_lo, app_day - 1)]
            if len(pool):
                k = min(len(pool), int(rng.integers(6, 11)))
                picks = pool.sample(n=k, random_state=int(rng.integers(0, 2**31 - 1)))
                rows = []
                for j, (_, r) in enumerate(picks.iterrows()):
                    row = r.copy()
                    if j % 2 == 1:  # near-duplicate: new number, ~same amount, nearby date
                        f = 1 + rng.uniform(-0.004, 0.004)
                        row["invoice_number"] = f"INV-{9000 + j}" if r.invoice_type == "SALE" else f"PI-{90000 + j}"
                        row["day"] = int(r.day + rng.integers(1, 5))
                        for c in ("amount", "tax_amount", "total_amount"):
                            row[c] = round(float(r[c]) * f, 2)
                    rows.append(row)
                inv = pd.concat([inv, pd.DataFrame(rows)], ignore_index=True)
    return inv.reset_index(drop=True), txn.reset_index(drop=True), gst.reset_index(drop=True), invm.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-borrower generation
# --------------------------------------------------------------------------- #
def _emi_dates(disb: pd.Timestamp, tenure: int, horizon_end: pd.Timestamp):
    out = []
    for k in range(1, tenure + 1):
        d = disb + pd.DateOffset(months=k)
        if d >= horizon_end:
            break
        out.append((k, d))
    return out


def _generate_borrower(p: Dict[str, Any], ctx: Dict[str, Any], rng: np.random.Generator):
    bid = p["borrower_id"]
    start, months, mday = ctx["start"], ctx["months"], ctx["month_day0"]
    nm, obs, app_day, H = len(months), ctx["obs"], ctx["app_day"], ctx["H"]
    app_date, horizon_end = ctx["app_date"], ctx["horizon_end"]
    mday_ext = np.append(mday, H)
    m_idx = np.arange(nm)

    # ---- revenue path (obs window + outcome window)
    season = 1 + p["season_amp"] * np.sin(2 * np.pi * (m_idx + p["season_phase"]) / 12)
    rev = p["R0"] * (1 + p["growth"]) ** m_idx * season * np.exp(rng.normal(0, p["vol"], nm))
    if p["default_flag"]:
        if p["deteriorating"]:
            for k, f in zip((obs - 3, obs - 2, obs - 1), (0.97, 0.92, 0.86)):
                rev[k] *= f
        rev[obs + max(0, p["first_miss"] - 2):] *= rng.uniform(0.55, 0.85)
    elif rng.random() < 0.10:
        rev[min(nm - 1, obs + 2):] *= rng.uniform(0.80, 0.95)

    # ---- counterparties
    n_c = int(rng.integers(*p["cust_range"]))
    c_ids = np.array([f"{bid}-C{j:03d}" for j in range(n_c)])
    c_names = np.array(_make_names(rng, n_c))
    c_gstin = np.array([f"SYN{int(x):010d}" for x in rng.integers(0, 10**10, n_c)])
    c_w = np.arange(1, n_c + 1, dtype=float) ** (-p["conc_alpha"])
    c_start = np.where(np.arange(n_c) < max(2, n_c // 2), 0, rng.integers(0, nm, n_c))
    c_delay = rng.uniform(5, 45, n_c)
    n_v = int(rng.integers(3, 13))
    v_ids = np.array([f"{bid}-V{j:02d}" for j in range(n_v)])
    v_names = np.array(_make_names(rng, n_v))
    v_gstin = np.array([f"SYN{int(x):010d}" for x in rng.integers(0, 10**10, n_v)])
    v_w = np.arange(1, n_v + 1, dtype=float) ** (-p["conc_alpha"])
    v_start = np.where(np.arange(n_v) < max(2, n_v // 2), 0, rng.integers(0, nm, n_v))
    v_delay = rng.uniform(3, 35, n_v)

    # ---- monthly invoices
    s_day, s_amt, s_c, p_day, p_amt, p_v = [], [], [], [], [], []
    for m in range(nm):
        pr = c_w * (c_start <= m)
        pr = pr / pr.sum()
        ns = max(1, int(rng.poisson(p["inv_per_month"] * max(rev[m] / p["R0"], 0.3))))
        w = rng.lognormal(0, 0.5, ns)
        s_day.append(mday[m] + rng.integers(0, 28, ns))
        s_amt.append(rev[m] * w / w.sum())
        s_c.append(rng.choice(n_c, size=ns, p=pr))
        cm = p["cogs"] * rev[m] * float(np.exp(rng.normal(0, 0.08)))
        npch = max(1, int(rng.poisson(0.6 * ns)))
        w2 = rng.lognormal(0, 0.5, npch)
        vp = v_w * (v_start <= m)
        vp = vp / vp.sum()
        p_day.append(mday[m] + rng.integers(0, 28, npch))
        p_amt.append(cm * w2 / w2.sum())
        p_v.append(rng.choice(n_v, size=npch, p=vp))
    o = np.argsort(np.concatenate(s_day), kind="stable")
    sale_day = np.concatenate(s_day)[o]
    samt = np.round(np.concatenate(s_amt)[o], 2)
    sale_c = np.concatenate(s_c)[o]
    o = np.argsort(np.concatenate(p_day), kind="stable")
    pur_day = np.concatenate(p_day)[o]
    pamt = np.round(np.concatenate(p_amt)[o], 2)
    pur_v = np.concatenate(p_v)[o]
    ns_tot, np_tot = len(sale_day), len(pur_day)

    rate = p["gst_rate"]
    stax, ptax = np.round(samt * rate, 2), np.round(pamt * rate, 2)
    stotal, ptotal = np.round(samt + stax, 2), np.round(pamt + ptax, 2)
    sale_price = p["unit_cost"] / p["cogs"]
    sqty = np.round(samt / sale_price, 2) if p["goods"] else np.full(ns_tot, np.nan)
    pqty = np.round(pamt / p["unit_cost"], 2) if p["goods"] else np.full(np_tot, np.nan)
    sale_no = np.array([f"INV-{1000 + k}" for k in range(ns_tot)])
    pur_no = np.array([f"PI-{k + 1:05d}" for k in range(np_tot)])
    item = f"{p['sector']} goods" if p["goods"] else "Services"

    inv = pd.concat([
        pd.DataFrame({"day": sale_day, "invoice_type": "SALE", "invoice_number": sale_no,
                      "counterparty_id": c_ids[sale_c], "counterparty_name": c_names[sale_c],
                      "amount": samt, "tax_amount": stax, "total_amount": stotal, "quantity": sqty}),
        pd.DataFrame({"day": pur_day, "invoice_type": "PURCHASE", "invoice_number": pur_no,
                      "counterparty_id": v_ids[pur_v], "counterparty_name": v_names[pur_v],
                      "amount": pamt, "tax_amount": ptax, "total_amount": ptotal, "quantity": pqty}),
    ], ignore_index=True)
    inv["item"] = item

    # ---- settlements (bank / UPI)
    frames: List[pd.DataFrame] = []
    delay_s = np.maximum(0, np.round(rng.normal(c_delay[sale_c], 6))).astype(int)
    frames += _settlement(rng, sale_day, stotal, delay_s, sale_no, c_ids[sale_c],
                          np.char.upper(c_names[sale_c]), "CREDIT", "SALES_RECEIPT",
                          p["upi_share"], p["upi_max"], H, 0.07, 0.08, 0.12)
    delay_p = np.maximum(0, np.round(rng.normal(v_delay[pur_v], 4))).astype(int)
    frames += _settlement(rng, pur_day, ptotal, delay_p, pur_no, v_ids[pur_v],
                          np.char.upper(v_names[pur_v]), "DEBIT", "VENDOR_PAYMENT",
                          p["upi_share"], p["upi_max"] * 0.5, H, 0.05, 0.0, 0.10)

    # ---- GST-style records
    file_delay = np.where(rng.random(nm) < p["gst_comp"], 0, rng.integers(2, 46, nm))
    missing = rng.random(nm) < 0.06 * (1 - p["gst_comp"])
    fday = mday_ext[1:nm + 1] + 19 + file_delay
    periods = np.array([mm.strftime("%Y-%m") for mm in months])

    def _gst(day, no, names, gstin, amt, tax, direction):
        mi = np.searchsorted(mday, day, side="right") - 1
        up = np.char.upper(names)
        up = np.where(rng.random(len(day)) < 0.10, np.char.add(up, " PVT LTD"), up)
        shift = np.where(rng.random(len(day)) < 0.03, rng.integers(1, 6, len(day)), 0)
        keep = (fday[mi] < H) & ~missing[mi]
        return pd.DataFrame({
            "day": (day + shift)[keep], "invoice_number": no[keep], "counterparty_name": up[keep],
            "counterparty_gstin": gstin[keep], "taxable_value": amt[keep], "tax_amount": tax[keep],
            "direction": direction, "filing_period": periods[mi][keep], "filing_day": fday[mi][keep],
        })
    gst = pd.concat([
        _gst(sale_day, sale_no, c_names[sale_c], c_gstin[sale_c], samt, stax, "OUTWARD"),
        _gst(pur_day, pur_no, v_names[pur_v], v_gstin[pur_v], pamt, ptax, "INWARD"),
    ], ignore_index=True)

    # monthly GST payment to the (synthetic) exchequer
    mi_s = np.searchsorted(mday, sale_day, side="right") - 1
    mi_p = np.searchsorted(mday, pur_day, side="right") - 1
    net_tax = np.maximum(np.bincount(mi_s, weights=stax, minlength=nm) -
                         np.bincount(mi_p, weights=ptax, minlength=nm), 0)
    ok = (fday < H) & ~missing & (net_tax > 0)
    if ok.any():
        frames.append(_txn_frame(rng, fday[ok], net_tax[ok], "DEBIT", "GST_PAYMENT", "GOV-GST",
                                 "GST PAYMENT (SYNTHETIC)", "", "", "NEFT", "BANK"))

    # ---- operating expenses
    ox = {"day": [], "amt": [], "type": [], "cp": []}
    for m in range(nm):
        om = p["opex_ratio"] * p["R0"] * (0.6 + 0.4 * rev[m] / p["R0"]) * float(np.exp(rng.normal(0, 0.05)))
        for t, share, d, cp in (("SALARY", 0.45, rng.integers(0, 5), "PAYROLL"),
                                ("RENT", 0.20, rng.integers(5, 10), "LANDLORD"),
                                ("UTILITY", 0.10, rng.integers(9, 16), "UTILITY PROVIDER")):
            ox["day"].append(mday[m] + int(d)); ox["amt"].append(om * share); ox["type"].append(t); ox["cp"].append(cp)
        k = int(rng.integers(3, 9))
        w = rng.dirichlet(np.ones(k))
        for j in range(k):
            ox["day"].append(mday[m] + int(rng.integers(0, 28))); ox["amt"].append(om * 0.25 * w[j])
            ox["type"].append("OTHER_OPEX"); ox["cp"].append(f"MISC-{int(rng.integers(1, 15)):02d}")
        ox["day"].append(mday[m] + 27); ox["amt"].append(float(rng.uniform(200, 900)))
        ox["type"].append("BANK_CHARGE"); ox["cp"].append("BANK")
    ox_type = np.array(ox["type"])
    ox_upi = np.isin(ox_type, ["UTILITY", "OTHER_OPEX"]) & (rng.random(len(ox_type)) < p["upi_share"])
    frames.append(_txn_frame(rng, ox["day"], ox["amt"], "DEBIT", ox_type, np.array(ox["cp"]), np.array(ox["cp"]),
                             "", "", np.where(ox_upi, "UPI", "NEFT"), np.where(ox_upi, "UPI", "BANK")))

    # ---- loans, repayments and their bank footprints
    loans, repays = [], []
    S = p["S"]

    def _emi_status(due, k, is_app):
        months_after = k if is_app else (due.year - app_date.year) * 12 + (due.month - app_date.month)
        if due >= app_date:
            if p["default_flag"] and months_after >= p["first_miss"]:
                return "MISSED", 0
            p_miss, p_late = 0.0, 0.25 * (1 - p["disc"]) + 0.03
        else:
            det = p["default_flag"] and p["deteriorating"] and due >= app_date - pd.DateOffset(months=6)
            p_miss = 0.05 * (1 - p["disc"]) * (3 if det else 1)
            p_late = (0.25 * (1 - p["disc"]) + 0.03) * (2 if det else 1)
        u = rng.random()
        if u < p_miss:
            return "MISSED", 0
        if u < p_miss + p_late:
            return "PAID_LATE", int(rng.integers(1, 21))
        return "PAID_ON_TIME", 0

    loan_specs = []
    for j in range(p["n_prior"]):
        disb = months[int(rng.integers(0, max(1, obs - 5)))] + pd.Timedelta(days=int(rng.integers(0, 20)))
        emi = S * float(p["prior_shares"][j])
        r, ten = float(rng.uniform(0.10, 0.18)), int(rng.choice([24, 36, 48, 60]))
        loan_specs.append((f"LN-{bid}-{j + 1}", disb, emi, r, ten, False))
    app_emi, app_rate = S * p["app_share"], float(rng.uniform(0.11, 0.19))
    loan_specs.append((f"LN-{bid}-APP", app_date, app_emi, app_rate, 36, True))

    emi_rows = []
    for loan_id, disb, emi, r, ten, is_app in loan_specs:
        principal = round(_annuity_pv(emi, r, ten), 2)
        loans.append(dict(loan_id=loan_id, borrower_id=bid, lender_type=str(rng.choice(["Bank", "NBFC", "Cooperative"])),
                          disbursal_date=disb, principal=principal, annual_interest_rate=round(r, 4),
                          tenure_months=ten, emi=round(emi, 2), is_application_loan=is_app))
        dday = (disb - start).days
        emi_rows.append((dday, principal, "CREDIT", "LOAN_DISBURSAL", "LENDER", "LENDER", "OK"))
        emi_rows.append((dday + int(rng.integers(2, 11)), principal * 0.92, "DEBIT", "CAPEX_PAYMENT",
                         "EQUIPMENT SUPPLIER", "EQUIPMENT SUPPLIER", "OK"))
        for k, due in _emi_dates(disb, ten, horizon_end):
            status, dpd = _emi_status(due, k, is_app)
            due_day = (due - start).days
            paid_day = min(due_day + dpd, H - 1)
            repays.append(dict(repayment_id=f"RP-{loan_id}-{k:02d}", loan_id=loan_id, borrower_id=bid,
                               due_date=due, paid_date=(start + pd.Timedelta(days=paid_day)) if status != "MISSED" else pd.NaT,
                               due_amount=round(emi, 2), paid_amount=round(emi, 2) if status != "MISSED" else 0.0,
                               days_past_due=float(dpd) if status != "MISSED" else np.nan, status=status))
            if status == "MISSED":
                emi_rows.append((due_day, emi, "DEBIT", "LOAN_EMI", loan_id, loan_id, "BOUNCED"))
            else:
                emi_rows.append((paid_day, emi, "DEBIT", "LOAN_EMI", loan_id, loan_id, "OK"))
    for d, a, dr, cat, cid, cn, st in emi_rows:
        if d < H:
            frames.append(_txn_frame(
                rng, [d], [a], dr, cat, cid, cn, "" if cat != "LOAN_EMI" else cid, "",
                "AUTO_DEBIT" if cat == "LOAN_EMI" else "NEFT", "BANK",
                status="BOUNCED" if st == "BOUNCED" else "SUCCESS"))

    # unrelated bounced cheques (behavioural signal)
    nb = int(rng.poisson(1.2 * (1 - p["disc"]) * obs / 12))
    if nb:
        frames.append(_txn_frame(rng, rng.integers(0, app_day, nb), p["R0"] * rng.uniform(0.05, 0.2, nb), "DEBIT",
                                 "VENDOR_PAYMENT", v_ids[0], v_names[0].upper(), "", "", "CHEQUE", "BANK",
                                 status="BOUNCED"))

    # ---- inventory movements (goods sectors only)
    invm = pd.DataFrame(columns=["day", "movement_type", "quantity", "unit_cost", "reference"])
    if p["inventory_tracked"]:
        pm = np.searchsorted(mday, pur_day, side="right") - 1
        monthly_in = np.bincount(pm, weights=pqty, minlength=nm)
        invm = pd.concat([
            pd.DataFrame({"day": pur_day + rng.integers(0, 6, np_tot), "movement_type": "PURCHASE_IN",
                          "quantity": np.round(pqty * np.exp(rng.normal(0, 0.03, np_tot)), 2),
                          "unit_cost": np.round(p["unit_cost"] * np.exp(rng.normal(0, 0.02, np_tot)), 2),
                          "reference": pur_no}),
            pd.DataFrame({"day": sale_day, "movement_type": "SALE_OUT",
                          "quantity": np.round(sqty * np.exp(rng.normal(0, 0.04, ns_tot)), 2),
                          "unit_cost": round(sale_price, 2), "reference": sale_no}),
            pd.DataFrame({"day": mday + 27, "movement_type": "ADJUSTMENT_OUT",
                          "quantity": np.round(0.005 * monthly_in, 2), "unit_cost": p["unit_cost"], "reference": ""}),
        ], ignore_index=True)

    # ---- assemble bank/UPI frame, inject scenarios, finalise
    txn = pd.concat([f for f in frames if len(f)], ignore_index=True)
    gst_df = gst
    inv, txn, gst_df, invm = _apply_scenarios(p["scenarios"], inv, txn, gst_df, invm, p, app_day, rng, bid)

    txn = txn[(txn.day >= 0) & (txn.day < H)].copy()
    txn["order"] = np.arange(len(txn))
    txn = txn.sort_values(["day", "order"], kind="stable").reset_index(drop=True)
    sign = np.where(txn.direction == "CREDIT", 1.0, -1.0)
    delta = np.where(txn.status == "SUCCESS", sign * txn.amount, 0.0)
    txn["balance_after"] = np.round(p["R0"] * rng.uniform(0.4, 1.0) + np.cumsum(delta), 2)
    txn["txn_date"] = start + pd.to_timedelta(txn["day"], unit="D")
    txn["borrower_id"] = bid
    seq = txn.groupby("channel").cumcount().astype(str).str.zfill(6)
    txn["txn_id"] = txn["channel"].map({"BANK": "BNK", "UPI": "UPI"}) + f"-{bid}-" + seq
    tcols = ["txn_id", "borrower_id", "channel", "txn_date", "txn_hour", "amount", "direction", "txn_type",
             "payment_mode", "counterparty_id", "counterparty_name", "reference", "status", "balance_after"]
    txn = txn[tcols]

    inv = inv.sort_values(["day", "invoice_type"], kind="stable").reset_index(drop=True)
    inv["invoice_id"] = [f"IV-{bid}-{i:05d}" for i in range(len(inv))]
    inv["borrower_id"] = bid
    inv["invoice_date"] = start + pd.to_timedelta(inv["day"], unit="D")
    inv["due_date"] = inv["invoice_date"] + pd.Timedelta(days=30)
    inv = inv[["invoice_id", "borrower_id", "invoice_type", "invoice_number", "counterparty_id",
               "counterparty_name", "invoice_date", "due_date", "amount", "tax_amount", "total_amount",
               "item", "quantity"]]

    gst_df = gst_df.reset_index(drop=True)
    gst_df["gst_record_id"] = [f"GS-{bid}-{i:05d}" for i in range(len(gst_df))]
    gst_df["borrower_id"] = bid
    gst_df["invoice_date"] = start + pd.to_timedelta(gst_df["day"], unit="D")
    gst_df["filing_date"] = start + pd.to_timedelta(gst_df["filing_day"], unit="D")
    gst_df = gst_df[["gst_record_id", "borrower_id", "invoice_number", "counterparty_name", "counterparty_gstin",
                     "invoice_date", "taxable_value", "tax_amount", "direction", "filing_period", "filing_date"]]

    if len(invm):
        invm = invm[invm.day < H].copy()
        invm["order"] = np.arange(len(invm))
        invm = invm.sort_values(["day", "order"], kind="stable").reset_index(drop=True)
        opening = 2.0 * float(np.nansum(pqty)) / nm
        signed = np.where(invm.movement_type == "PURCHASE_IN", invm.quantity, -invm.quantity)
        invm["stock_after"] = np.round(opening + np.cumsum(signed), 2)
        invm["inventory_id"] = [f"IM-{bid}-{i:05d}" for i in range(len(invm))]
        invm["borrower_id"] = bid
        invm["movement_date"] = start + pd.to_timedelta(invm["day"], unit="D")
        invm["item"] = item
        invm = invm[["inventory_id", "borrower_id", "movement_date", "item", "movement_type", "quantity",
                     "unit_cost", "reference", "stock_after"]]
    else:
        invm = pd.DataFrame(columns=["inventory_id", "borrower_id", "movement_date", "item", "movement_type",
                                     "quantity", "unit_cost", "reference", "stock_after"])

    vendors = pd.DataFrame({
        "vendor_id": v_ids, "borrower_id": bid, "vendor_name": v_names, "vendor_gstin": v_gstin,
        "relationship_start_date": [months[int(s)] for s in v_start],
        "avg_payment_terms_days": np.round(v_delay).astype(int), "is_registered": True,
    })
    borrower = dict(
        borrower_id=bid, business_name=p["business_name"], enterprise_size=p["enterprise_size"],
        sector=p["sector"], region=p["region"], credit_history_months=p["credit_history_months"],
        credit_history_band=p["credit_history_band"], digital_adoption_level=p["digital_adoption_level"],
        incorporation_year=p["incorporation_year"], inventory_tracked=p["inventory_tracked"],
        application_date=app_date, requested_amount=loans[-1]["principal"], requested_tenure_months=36,
        requested_interest_rate=loans[-1]["annual_interest_rate"], requested_emi=loans[-1]["emi"],
        default_flag=p["default_flag"],
    )
    return dict(borrower=borrower, txn=txn, inv=inv, vendors=vendors, invm=invm, gst=gst_df,
                loans=pd.DataFrame(loans), repays=pd.DataFrame(repays))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_synthetic_data(n_borrowers: int | None = None, n_months: int | None = None,
                            seed: int | None = None) -> Dict[str, pd.DataFrame]:
    """Generate the full synthetic dataset as a dict of DataFrames (see ``TABLE_NAMES``)."""
    cfg = get_settings()
    dcfg, fcfg = cfg["data"], cfg["features"]
    n_borrowers = int(n_borrowers or dcfg["n_borrowers"])
    n_months = int(n_months or dcfg["n_months"])
    seed = get_seed() if seed is None else int(seed)
    obs = min(int(fcfg["observation_months"]), n_months - 2)
    outcome = n_months - obs
    start = pd.Timestamp(dcfg["start_date"])
    months = pd.date_range(start, periods=n_months, freq="MS")
    mday = np.array([(m - start).days for m in months])
    horizon_end = months[-1] + pd.offsets.MonthBegin(1)
    ctx = dict(start=start, months=months, month_day0=mday, obs=obs, app_day=int(mday[obs]),
               app_date=months[obs], horizon_end=horizon_end, H=int((horizon_end - start).days))

    logger.info("Generating SYNTHETIC data: %d borrowers, %d months (seed=%d)", n_borrowers, n_months, seed)
    profiles = _draw_population(np.random.default_rng(seed), n_borrowers, dcfg, obs, outcome)
    parts: Dict[str, List[Any]] = {k: [] for k in ["borrower", "txn", "inv", "vendors", "invm", "gst", "loans", "repays"]}
    for i, prof in enumerate(profiles):
        res = _generate_borrower(prof, ctx, np.random.default_rng([seed, 1000 + i]))
        for k in parts:
            parts[k].append(res[k])

    def _cat(key):
        frames = [f for f in parts[key] if len(f)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    borrowers = pd.DataFrame(parts["borrower"])
    txn = _cat("txn")
    scen = pd.DataFrame([
        {"borrower_id": pr["borrower_id"], "scenario_types": ",".join(pr["scenarios"]),
         "n_scenarios": len(pr["scenarios"])} for pr in profiles if pr["scenarios"]
    ], columns=["borrower_id", "scenario_types", "n_scenarios"])
    data = {
        "borrowers": borrowers,
        "bank_transactions": txn[txn.channel == "BANK"].reset_index(drop=True),
        "upi_transactions": txn[txn.channel == "UPI"].reset_index(drop=True),
        "invoices": _cat("inv"), "vendors": _cat("vendors"), "inventory": _cat("invm"),
        "gst_records": _cat("gst"), "loans": _cat("loans"), "repayments": _cat("repays"),
        "scenarios": scen,
    }
    logger.info("Generated %s | default rate=%.3f | %s", {k: len(v) for k, v in data.items()},
                borrowers["default_flag"].mean(), DISCLAIMER_SYNTHETIC)
    return data


def dataset_summary(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Row counts and target prevalence, useful for logging and tests."""
    return {
        "rows": {k: int(len(v)) for k, v in data.items()},
        "default_rate": float(data["borrowers"]["default_flag"].mean()),
        "n_scenario_borrowers": int(len(data["scenarios"])),
        "disclaimer": DISCLAIMER_SYNTHETIC,
    }


def save_raw_data(data: Dict[str, pd.DataFrame], directory: str | Path | None = None) -> Path:
    """Write each table to ``data/raw/<table>.csv``."""
    d = Path(directory) if directory else get_paths().raw_data
    d.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_csv(d / f"{name}.csv", index=False)
    (d / "README_SYNTHETIC.txt").write_text(DISCLAIMER_SYNTHETIC + "\n", encoding="utf-8")
    return d


def load_raw_data(directory: str | Path | None = None) -> Dict[str, pd.DataFrame]:
    """Read tables previously written by ``save_raw_data``."""
    d = Path(directory) if directory else get_paths().raw_data
    data = {}
    for name in TABLE_NAMES:
        f = d / f"{name}.csv"
        if not f.exists():
            raise FileNotFoundError(f"Missing raw table: {f}")
        data[name] = pd.read_csv(f, parse_dates=DATE_COLUMNS.get(name, []))
    return data


def pick_demo_borrowers(data: Dict[str, pd.DataFrame], n: int | None = None) -> List[str]:
    """Choose demo borrowers: one clean, then one per distinct injected scenario type."""
    n = int(n or get_settings()["data"]["demo_borrowers"])
    b, sc = data["borrowers"], data["scenarios"]
    scen_ids = set(sc["borrower_id"])
    chosen: List[str] = []
    clean = b[(~b.borrower_id.isin(scen_ids)) & (b.default_flag == 0)]
    if len(clean):
        chosen.append(str(clean.iloc[0].borrower_id))
    seen = set()
    for _, r in sc.iterrows():
        first = r["scenario_types"].split(",")[0]
        if first not in seen and len(chosen) < n:
            chosen.append(str(r["borrower_id"]))
            seen.add(first)
    for bid in b["borrower_id"]:
        if len(chosen) >= n:
            break
        if bid not in chosen:
            chosen.append(str(bid))
    return chosen[:n]