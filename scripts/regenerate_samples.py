"""
regenerate_samples.py

Regenerates sample.csv, sample.xlsx, and sample.json without any timestamp
column. The server stamps received_at automatically when records enter the
queue, so branches only need to provide store_id, product, quantity, and
optionally update_id.

Usage:
    python3 scripts/regenerate_samples.py

Safe to run at any time — only rewrites the three fresh sample files.
The DLQ demo files are never touched.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd


def run():
    data_dir = os.path.join(PROJECT_ROOT, "data")

    # ------------------------------------------------------------------
    # sample.csv — standard column names, no timestamp
    # Contains a conflict pair (UID-001 and UID-002 map to same store+product,
    # both will be in the same batch if window is 5+ records) and one
    # duplicate update_id to demonstrate idempotency.
    # ------------------------------------------------------------------
    csv_rows = [
        ("S001", "Paracetamol 500mg",  100, "UID-001"),
        ("S001", "Paracetamol 500mg",  120, "UID-002"),   # conflict with UID-001
        ("S002", "Ibuprofen 400mg",     50, "UID-003"),
        ("S001", "Amoxicillin 250mg",   30, "UID-004"),
        ("S003", "Aspirin 100mg",      200, "UID-005"),
        ("S002", "Paracetamol 500mg",   80, "UID-006"),
        ("S003", "Metformin 500mg",    150, "UID-007"),
        ("S001", "Ibuprofen 400mg",     45, "UID-008"),
        ("S002", "Amoxicillin 250mg",   60, "UID-009"),
        ("S004", "Aspirin 100mg",      175, "UID-010"),
        ("S004", "Paracetamol 500mg",   90, "UID-011"),
        ("S003", "Ibuprofen 400mg",     55, "UID-012"),
        ("S004", "Metformin 500mg",    220, "UID-013"),
        ("S002", "Aspirin 100mg",       95, "UID-014"),
        ("S001", "Metformin 500mg",     70, "UID-015"),
        ("S003", "Amoxicillin 250mg",   40, "UID-016"),
        ("S004", "Ibuprofen 400mg",    130, "UID-017"),
        ("S002", "Metformin 500mg",    110, "UID-018"),
        ("S001", "Aspirin 100mg",       85, "UID-019"),
        ("S003", "Paracetamol 500mg",  160, "UID-020"),
        ("S001", "Paracetamol 500mg",  125, "UID-002"),   # duplicate update_id
        ("S002", "Ibuprofen 400mg",     55, "UID-021"),
    ]
    path_csv = os.path.join(data_dir, "sample.csv")
    pd.DataFrame(
        csv_rows, columns=["store_id", "product", "quantity", "update_id"]
    ).to_csv(path_csv, index=False)
    print(f"Regenerated  {path_csv}  ({len(csv_rows)} rows, no timestamp)")

    # ------------------------------------------------------------------
    # sample.xlsx — non-standard column names (tests schema mapping)
    # ------------------------------------------------------------------
    xlsx_rows = [
        ("S001", "Paracetamol 500mg", 102, "XL-001"),
        ("S001", "Ibuprofen 400mg",    48, "XL-002"),
        ("S002", "Amoxicillin 250mg",  62, "XL-003"),
        ("S002", "Aspirin 100mg",      98, "XL-004"),
        ("S003", "Metformin 500mg",   155, "XL-005"),
        ("S003", "Paracetamol 500mg", 162, "XL-006"),
        ("S004", "Ibuprofen 400mg",   133, "XL-007"),
        ("S004", "Amoxicillin 250mg",  32, "XL-008"),
        ("S001", "Aspirin 100mg",      88, "XL-009"),
        ("S002", "Metformin 500mg",   112, "XL-010"),
        ("S003", "Aspirin 100mg",     205, "XL-011"),
        ("S004", "Paracetamol 500mg",  93, "XL-012"),
    ]
    path_xlsx = os.path.join(data_dir, "sample.xlsx")
    pd.DataFrame(
        xlsx_rows, columns=["shop_id", "drug_name", "stock", "version"]
    ).to_excel(path_xlsx, index=False, engine="openpyxl")
    print(f"Regenerated  {path_xlsx}  ({len(xlsx_rows)} rows, non-standard column names)")

    # ------------------------------------------------------------------
    # sample.json — non-standard keys, two intentionally invalid rows
    # ------------------------------------------------------------------
    json_records = [
        {"branch": "S001", "medicine": "Paracetamol 500mg", "qty": 105, "id": "UID-J01"},
        {"branch": "S001", "medicine": "Amoxicillin 250mg", "qty":  35, "id": "UID-J02"},
        {"branch": "S002", "medicine": "Ibuprofen 400mg",   "qty":  60, "id": "UID-J03"},
        {"branch": "S002", "medicine": "Metformin 500mg",   "qty": 115, "id": "UID-J04"},
        {"branch": "S003", "medicine": "Aspirin 100mg",     "qty": 210, "id": "UID-J05"},
        {"branch": "S003", "medicine": "Paracetamol 500mg", "qty": 165, "id": "UID-J06"},
        {"branch": "S004", "medicine": "Amoxicillin 250mg", "qty":  28, "id": "UID-J07"},
        {"branch": "S004", "medicine": "Ibuprofen 400mg",   "qty": 140, "id": "UID-J08"},
        {"branch": "S001", "medicine": "Metformin 500mg",   "qty":  75, "id": "UID-J09"},
        {"branch": "S002", "medicine": "Aspirin 100mg",     "qty":  98, "id": "UID-J10"},
        # Invalid row 1: non-numeric quantity (dropped by validator — intentional demo)
        {"branch": "S003", "medicine": "Ibuprofen 400mg",   "qty": "not-a-number", "id": "UID-J11"},
        # Invalid row 2: empty store id (dropped by validator — intentional demo)
        {"branch": "",     "medicine": "Metformin 500mg",   "qty": 220, "id": "UID-J12"},
    ]
    path_json = os.path.join(data_dir, "sample.json")
    with open(path_json, "w", encoding="utf-8") as fh:
        json.dump({"updates": json_records}, fh, indent=4)
    print(f"Regenerated  {path_json}  ({len(json_records)} rows, 2 invalid for validation demo)")

    print()
    print("No timestamps in any file — the server stamps received_at automatically.")
    print("DLQ demo files (sample_dlq_demo.*) were not changed.")


if __name__ == "__main__":
    run()