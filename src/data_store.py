"""Deterministic data access layer over the Olist CSVs.

Loaded once per process and indexed by order_id / seller_id so every agent
looks up the *same* verifiable rows instead of re-parsing CSVs or letting an
LLM invent numbers. All lookup functions return plain dict/list/None so
results are trivially JSON-serialisable for tracing.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from . import config


def _read_csv(name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(config.DATA_DIR / name, dtype=str)
    if parse_dates:
        for col in parse_dates:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@dataclass
class DataStore:
    orders: pd.DataFrame = field(repr=False)
    order_items: pd.DataFrame = field(repr=False)
    order_payments: pd.DataFrame = field(repr=False)
    sellers: pd.DataFrame = field(repr=False)
    customers: pd.DataFrame = field(repr=False)

    def order(self, order_id: str) -> dict[str, Any] | None:
        rows = self.orders[self.orders["order_id"] == order_id]
        if rows.empty:
            return None
        r = rows.iloc[0]
        return {
            "order_id": r["order_id"],
            "customer_id": r["customer_id"],
            "order_status": r["order_status"],
            "order_purchase_timestamp": _iso(r["order_purchase_timestamp"]),
            "order_approved_at": _iso(r["order_approved_at"]),
            "order_delivered_carrier_date": _iso(r["order_delivered_carrier_date"]),
            "order_delivered_customer_date": _iso(r["order_delivered_customer_date"]),
            "order_estimated_delivery_date": _iso(r["order_estimated_delivery_date"]),
        }

    def order_items_for(self, order_id: str) -> list[dict[str, Any]]:
        rows = self.order_items[self.order_items["order_id"] == order_id]
        rows = rows.sort_values("order_item_id", key=lambda s: s.astype(int))
        out = []
        for _, r in rows.iterrows():
            out.append(
                {
                    "order_id": r["order_id"],
                    "order_item_id": r["order_item_id"],
                    "product_id": r["product_id"],
                    "seller_id": r["seller_id"],
                    "shipping_limit_date": _iso(r["shipping_limit_date"]),
                    "price": round(float(r["price"]), 2),
                    "freight_value": round(float(r["freight_value"]), 2),
                }
            )
        return out

    def payments_for(self, order_id: str) -> list[dict[str, Any]]:
        rows = self.order_payments[self.order_payments["order_id"] == order_id]
        rows = rows.sort_values("payment_sequential", key=lambda s: s.astype(int))
        out = []
        for _, r in rows.iterrows():
            out.append(
                {
                    "order_id": r["order_id"],
                    "payment_sequential": r["payment_sequential"],
                    "payment_type": r["payment_type"],
                    "payment_installments": r["payment_installments"],
                    "payment_value": round(float(r["payment_value"]), 2),
                }
            )
        return out

    def seller(self, seller_id: str) -> dict[str, Any] | None:
        rows = self.sellers[self.sellers["seller_id"] == seller_id]
        if rows.empty:
            return None
        r = rows.iloc[0]
        return {
            "seller_id": r["seller_id"],
            "seller_zip_code_prefix": r["seller_zip_code_prefix"],
            "seller_city": r["seller_city"],
            "seller_state": r["seller_state"],
        }

    def customer_unique_id(self, customer_id: str) -> str | None:
        rows = self.customers[self.customers["customer_id"] == customer_id]
        if rows.empty:
            return None
        return rows.iloc[0]["customer_unique_id"]


def _iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    ts: datetime = value
    return ts.isoformat()


@functools.lru_cache(maxsize=1)
def load_data_store() -> DataStore:
    orders = _read_csv(
        "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    order_items = _read_csv("olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
    order_payments = _read_csv("olist_order_payments_dataset.csv")
    sellers = _read_csv("olist_sellers_dataset.csv")
    customers = _read_csv("olist_customers_dataset.csv")
    return DataStore(
        orders=orders,
        order_items=order_items,
        order_payments=order_payments,
        sellers=sellers,
        customers=customers,
    )
