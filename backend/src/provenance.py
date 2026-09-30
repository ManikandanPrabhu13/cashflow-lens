"""Evidence provenance: the chain of evidence behind a financial claim.

Answers "Why should I trust this financial claim?" by exposing, for each
invoice, the linked bank transaction, GST-style record, counterparty/vendor
and inventory movement (or the explicit absence of one), together with the
amount / date / quantity differences and the matching logic that produced the
link.

Provenance is separate from Evidence Integrity (which produces findings) and
from Fairness (which concerns model behaviour across groups). No confidence
percentages are reported: no calibrated confidence exists for these rule-based
links, so ``confidence`` is always null.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .evidence_integrity import (
    EvidenceConfig, EvidenceResult, _prep_invoices, get_table, related_party_ids, to_jsonable,
)

EDGE_COLUMNS = [
    "edge_id", "borrower_id", "source_id", "source_type", "target_id", "target_type",
    "relationship_type", "match_status", "amount_difference", "amount_difference_pct",
    "date_difference_days", "quantity_difference", "quantity_difference_pct",
    "matching_logic", "confidence",
]

BANK_REL = {"sales": "settled_by_bank_inflow", "purchase": "paid_by_bank_outflow"}
GST_REL = {"sales": "reported_in_gst_record", "purchase": "reported_in_gst_record"}
INVENTORY_REL = {"sales": "fulfilled_from_inventory_outflow", "purchase": "received_into_inventory"}
VENDOR_REL = {"sales": "billed_to_counterparty", "purchase": "issued_by_vendor"}

PROBLEM_STATUSES = {"amount_mismatch", "date_mismatch", "quantity_mismatch",
                    "unmatched", "related_party_flagged"}

CONFIDENCE_NOTE = (
    "Confidence values are intentionally not reported: links are produced by deterministic "
    "matching rules, and no calibrated probability of correctness exists. Use match_status, "
    "the differences and matching_logic to judge each link."
)


def _logic(kind: str, method: Optional[str], status: str, cfg: EvidenceConfig) -> str:
    if status == "insufficient_evidence":
        return "Source data not available for this borrower; check not evaluated."
    if status == "unmatched":
        return "No eligible evidence record found under the matching rules for this source."
    if kind == "bank":
        if method == "direct_reference":
            return "Transaction carries a direct reference to this invoice_id; values then compared."
        return (
            "Same borrower, counterparty and direction; date within "
            f"-{cfg.bank_days_before}..+{cfg.bank_max_days_after} days and amount within "
            f"{cfg.partial_match_tolerance_pct:.0%} to link; closest amount then date chosen, each "
            f"transaction used once. Reconciled if amount within {cfg.amount_tolerance_pct:.0%} and "
            f"date within -{cfg.bank_days_before}..+{cfg.bank_expected_days_after} days."
        )
    if kind == "gst":
        return (
            "GST-style record linked by invoice_id. Reconciled if taxable amount within "
            f"{cfg.gst_amount_tolerance_pct:.0%} of invoice amount and filing date within "
            f"-{cfg.gst_days_before}..+{cfg.gst_expected_days_after} days of the invoice."
        )
    if method == "direct_reference":
        return "Inventory movement carries a direct reference to this invoice_id; quantities then compared."
    return (
        "Same borrower and item, expected movement direction (purchase=in, sales=out); date within "
        f"+/-{cfg.inventory_days_window} days and quantity within {cfg.partial_match_tolerance_pct:.0%} "
        f"to link. Reconciled if quantity within {cfg.inventory_quantity_tolerance_pct:.0%}."
    )


def _match_edges(m: pd.DataFrame, kind: str, cfg: EvidenceConfig, rel_map: Dict[str, str],
                 target_prefix: str, target_type: str) -> pd.DataFrame:
    if m is None or m.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    has = m["evidence_id"].notna().to_numpy()
    inv_str = m["invoice_id"].astype(str)
    real_target = (target_prefix + ":" + m["evidence_id"].astype(str)).to_numpy()
    missing_target = ("missing:" + kind + ":" + inv_str).to_numpy()
    rel = m["invoice_type"].map(rel_map).fillna(rel_map["sales"])
    e = pd.DataFrame({
        "borrower_id": m["borrower_id"].to_numpy(),
        "source_id": ("invoice:" + inv_str).to_numpy(),
        "source_type": "invoice",
        "target_id": np.where(has, real_target, missing_target),
        "target_type": np.where(has, target_type, "missing_evidence"),
        "relationship_type": rel.to_numpy(),
        "match_status": m["match_status"].to_numpy(),
        "amount_difference": m["amount_difference"].to_numpy(),
        "amount_difference_pct": m["amount_difference_pct"].to_numpy(),
        "date_difference_days": m["date_difference_days"].to_numpy(),
        "quantity_difference": m["quantity_difference"].to_numpy(),
        "quantity_difference_pct": m["quantity_difference_pct"].to_numpy(),
        "matching_logic": [_logic(kind, meth, st, cfg)
                           for meth, st in zip(m["match_method"], m["match_status"])],
        "confidence": None,
    })
    return e


def _vendor_edges(inv: pd.DataFrame, related: Optional[set]) -> pd.DataFrame:
    if inv.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    d = inv.dropna(subset=["counterparty_id"])
    if d.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    if related is None:
        status = pd.Series("not_evaluable", index=d.index)
        logic = "Vendor relationship data not available; related-party status not evaluated."
    else:
        status = pd.Series(np.where(d["counterparty_id"].isin(related),
                                    "related_party_flagged", "no_relationship_flag"), index=d.index)
        logic = "Counterparty looked up in vendor records' related-party flag."
    return pd.DataFrame({
        "borrower_id": d["borrower_id"].to_numpy(),
        "source_id": ("invoice:" + d["invoice_id"].astype(str)).to_numpy(),
        "source_type": "invoice",
        "target_id": ("vendor:" + d["counterparty_id"].astype(str)).to_numpy(),
        "target_type": "vendor",
        "relationship_type": d["invoice_type"].map(VENDOR_REL).fillna(VENDOR_REL["sales"]).to_numpy(),
        "match_status": status.to_numpy(),
        "amount_difference": np.nan, "amount_difference_pct": np.nan,
        "date_difference_days": np.nan, "quantity_difference": np.nan,
        "quantity_difference_pct": np.nan,
        "matching_logic": logic, "confidence": None,
    })


def build_provenance_edges(
    tables: Mapping[str, pd.DataFrame], result: EvidenceResult,
    config: Optional[EvidenceConfig] = None,
) -> pd.DataFrame:
    """Flat edge table (all borrowers) built from the evidence match tables."""
    cfg = config or result.config
    inv = _prep_invoices(get_table(tables, "invoices"))
    related = related_party_ids(get_table(tables, "vendors"))
    parts = [
        _match_edges(result.bank_matches, "bank", cfg, BANK_REL, "txn", "bank_transaction"),
        _match_edges(result.gst_matches, "gst", cfg, GST_REL, "gst", "gst_record"),
        _match_edges(result.inventory_matches, "inventory", cfg, INVENTORY_REL, "inventory", "inventory_movement"),
        _vendor_edges(inv, related),
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    edges = pd.concat(parts, ignore_index=True)
    edges["edge_id"] = edges["source_id"] + "->" + edges["target_id"] + ":" + edges["relationship_type"]
    return edges[EDGE_COLUMNS]


def save_provenance_edges(edges: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(path, index=False)
    return path


def load_provenance_edges(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# Per-borrower graph
# --------------------------------------------------------------------------
def _records_by_id(df: pd.DataFrame, id_col: str, ids: set) -> Dict[str, Dict[str, Any]]:
    if df.empty or id_col not in df.columns or not ids:
        return {}
    sub = df[df[id_col].astype(str).isin(ids)]
    return {str(r[id_col]): r for r in sub.to_dict("records")}


def _share(part: float, whole: float) -> Optional[float]:
    return part / whole if whole and whole > 0 else None


def build_provenance_graph(
    borrower_id: Any,
    tables: Mapping[str, pd.DataFrame],
    edges: pd.DataFrame,
    period: Optional[str] = None,
    max_invoices: int = 25,
    invoice_ids: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Provenance graph for one borrower.

    Chain: Claimed Revenue -> Invoice -> Bank Settlement / GST-style Record /
    Vendor / Inventory. Invoices with issues are shown first; ``truncated``
    reports whether more invoices exist than are displayed. ``claim`` and
    ``stages`` are computed over ALL invoices in the period, not only those shown.
    """
    e_b = edges[edges["borrower_id"] == borrower_id]
    inv = _prep_invoices(get_table(tables, "invoices"))
    inv = inv[inv["borrower_id"] == borrower_id].copy() if not inv.empty else inv
    empty = {
        "borrower_id": borrower_id, "period": period, "claim": None, "stages": [],
        "nodes": [], "edges": [], "selection": {"invoices_in_period": 0, "invoices_shown": 0, "truncated": False},
        "notes": [CONFIDENCE_NOTE, "No invoices available for this borrower/period."],
    }
    if inv.empty:
        return to_jsonable(empty)
    if period:
        p = pd.Period(period, "M")
        inv = inv[(inv["invoice_date"] >= p.start_time) & (inv["invoice_date"] <= p.end_time)]
        if inv.empty:
            return to_jsonable(empty)

    inv["_iid"] = inv["invoice_id"].astype(str)
    problem_src = set(e_b.loc[e_b["match_status"].isin(PROBLEM_STATUSES), "source_id"])
    inv["_problem"] = ("invoice:" + inv["_iid"]).isin(problem_src)
    if invoice_ids:
        sel = inv[inv["_iid"].isin({str(i) for i in invoice_ids})]
    else:
        sel = inv.sort_values(["_problem", "invoice_date"], ascending=[False, False]).head(max_invoices)
    shown_src = {"invoice:" + i for i in sel["_iid"]}

    # ---- claim + stage summaries over ALL invoices in the period ----
    sales = inv[inv["invoice_type"] == "sales"]
    sales_src = {"invoice:" + i for i in sales["_iid"]}
    claimed = float(sales["amount"].sum())

    def reconciled_amount(rel: str) -> float:
        sub = e_b[e_b["source_id"].isin(sales_src) & (e_b["relationship_type"] == rel)
                  & (e_b["match_status"] == "reconciled")]
        ids = sub["source_id"].str[len("invoice:"):]
        return float(sales.loc[sales["_iid"].isin(ids), "amount"].sum())

    bank_amt = reconciled_amount(BANK_REL["sales"])
    gst_amt = reconciled_amount(GST_REL["sales"])
    claim_id = f"claim:{borrower_id}:{period or 'all'}"
    claim = {
        "claim_id": claim_id, "type": "claimed_revenue", "period": period or "all",
        "claimed_revenue": claimed, "sales_invoice_count": int(len(sales)),
        "bank_reconciled_amount": bank_amt, "bank_reconciled_share": _share(bank_amt, claimed),
        "gst_reconciled_amount": gst_amt, "gst_reconciled_share": _share(gst_amt, claimed),
        "not_bank_reconciled_amount": claimed - bank_amt if claimed > 0 else None,
    }

    all_src = {"invoice:" + i for i in inv["_iid"]}
    e_period = e_b[e_b["source_id"].isin(all_src)]
    stage_defs = [
        ("bank_settlement", set(BANK_REL.values()), "reconciled"),
        ("gst_record", set(GST_REL.values()), "reconciled"),
        ("inventory", set(INVENTORY_REL.values()), "reconciled"),
        ("vendor_link", set(VENDOR_REL.values()), "no_relationship_flag"),
    ]
    stages: List[Dict[str, Any]] = [{"stage": "claimed_revenue", "amount": claimed}, {
        "stage": "invoice", "count": int(len(inv)), "sales": int(len(sales)),
        "purchase": int((inv["invoice_type"] == "purchase").sum())}]
    for name, rels, ok_status in stage_defs:
        sub = e_period[e_period["relationship_type"].isin(rels)]
        ev = sub[~sub["match_status"].isin(["insufficient_evidence", "not_evaluable"])]
        good = int((ev["match_status"] == ok_status).sum())
        stages.append({
            "stage": name, "evaluated": int(len(ev)), "reconciled": good,
            "issues": int(len(ev)) - good,
            "not_evaluated": int(len(sub) - len(ev)),
            "status_counts": ev["match_status"].value_counts().to_dict(),
        })

    # ---- nodes / edges for the shown invoices ----
    e_shown = e_b[e_b["source_id"].isin(shown_src)]
    tx = get_table(tables, "transactions")
    tx = tx[tx["borrower_id"] == borrower_id] if not tx.empty and "borrower_id" in tx.columns else tx
    gst = get_table(tables, "gst_records")
    stock = get_table(tables, "inventory")
    vendors = get_table(tables, "vendors")

    def target_ids(prefix: str) -> set:
        s = e_shown.loc[e_shown["target_id"].str.startswith(prefix + ":"), "target_id"]
        return {t[len(prefix) + 1:] for t in s}

    tx_rec = _records_by_id(tx, "txn_id", target_ids("txn"))
    gst_rec = _records_by_id(gst, "gst_id", target_ids("gst"))
    inv_rec = _records_by_id(stock, "inventory_id", target_ids("inventory"))
    ven_rec = _records_by_id(vendors, "vendor_id", target_ids("vendor"))

    borrowers = get_table(tables, "borrowers")
    b_rec: Dict[str, Any] = {}
    if not borrowers.empty:
        m = borrowers[borrowers["borrower_id"] == borrower_id]
        b_rec = m.iloc[0].to_dict() if len(m) else {}
    b_label = str(b_rec.get("business_name") or b_rec.get("name") or borrower_id)

    nodes: Dict[str, Dict[str, Any]] = {
        f"borrower:{borrower_id}": {"id": f"borrower:{borrower_id}", "type": "borrower",
                                     "label": b_label, "attributes": to_jsonable(b_rec)},
        claim_id: {"id": claim_id, "type": "claimed_revenue",
                   "label": f"Claimed revenue ({period or 'all invoiced periods'})",
                   "attributes": to_jsonable(claim)},
    }
    for r in sel.to_dict("records"):
        nid = f"invoice:{r['_iid']}"
        rec = {k: v for k, v in r.items() if not k.startswith("_")}
        nodes[nid] = {"id": nid, "type": "invoice",
                      "label": f"Invoice {r['_iid']} ({r['invoice_type']}, {r['amount']:,.0f})",
                      "attributes": to_jsonable(rec)}

    for tid in e_shown["target_id"].unique():
        if tid in nodes:
            continue
        prefix, _, raw = tid.partition(":")
        if prefix == "missing":
            _, _, rest = raw.partition(":")
            kind = raw.split(":", 1)[0]
            nodes[tid] = {"id": tid, "type": "missing_evidence",
                          "label": f"No {kind} evidence linked to invoice {rest}", "attributes": {"source": kind}}
            continue
        lookup = {"txn": (tx_rec, "bank_transaction", "Bank txn"), "gst": (gst_rec, "gst_record", "GST record"),
                  "inventory": (inv_rec, "inventory_movement", "Inventory movement"),
                  "vendor": (ven_rec, "vendor", "Counterparty")}[prefix]
        rec = lookup[0].get(raw, {})
        nodes[tid] = {"id": tid, "type": lookup[1], "label": f"{lookup[2]} {raw}",
                      "attributes": to_jsonable(rec)}

    edge_records = e_shown.to_dict("records")
    for r in sel.to_dict("records"):
        if r["invoice_type"] == "sales":
            edge_records.append({
                "edge_id": f"{claim_id}->invoice:{r['_iid']}:claimed_revenue_supported_by_invoice",
                "borrower_id": borrower_id, "source_id": claim_id, "source_type": "claimed_revenue",
                "target_id": f"invoice:{r['_iid']}", "target_type": "invoice",
                "relationship_type": "claimed_revenue_supported_by_invoice", "match_status": "invoiced",
                "amount_difference": None, "amount_difference_pct": None, "date_difference_days": None,
                "quantity_difference": None, "quantity_difference_pct": None,
                "matching_logic": "Sales invoices dated within the selected period are summed into claimed revenue.",
                "confidence": None,
            })
    edge_records.append({
        "edge_id": f"borrower:{borrower_id}->{claim_id}:asserts_claim", "borrower_id": borrower_id,
        "source_id": f"borrower:{borrower_id}", "source_type": "borrower", "target_id": claim_id,
        "target_type": "claimed_revenue", "relationship_type": "asserts_claim", "match_status": "asserted",
        "amount_difference": None, "amount_difference_pct": None, "date_difference_days": None,
        "quantity_difference": None, "quantity_difference_pct": None,
        "matching_logic": "Revenue claim derived from the borrower's own sales invoices.", "confidence": None,
    })

    return to_jsonable({
        "borrower_id": borrower_id, "period": period or "all", "claim": claim, "stages": stages,
        "nodes": list(nodes.values()), "edges": edge_records,
        "selection": {"invoices_in_period": int(len(inv)), "invoices_shown": int(len(sel)),
                      "truncated": bool(len(sel) < len(inv)),
                      "ordering": "invoices with issues first, then most recent"},
        "notes": [CONFIDENCE_NOTE],
    })