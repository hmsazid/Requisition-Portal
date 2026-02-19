import re
import io
import datetime as dt
from typing import Optional, Dict, Any, List

import pandas as pd
import streamlit as st
from PIL import Image

import cv2
import numpy as np
import pytesseract


# -----------------------------
# Constants: REQUIRED TABLE COLS (DO NOT CHANGE ORDER)
# -----------------------------
COLUMNS = [
    "Link",
    "Vendor Name",
    "Department",
    "Payment Details",
    "Bill Number",
    "Payment Method",
    "Requested Amount",
    "Requested Currency",
    "Disburse Amount (USD)",
    "Charge/Fees (USD)",
    "Total Disbursed Amount (USD)",  # MUST STAY BLANK
    "Payment Proof",
    "Disbursement Date",
    "Disbursement Date (ApprovalMax)",  # user wants blank
]

# -----------------------------
# OCR Helpers
# -----------------------------
def preprocess_for_ocr(pil_img: Image.Image) -> np.ndarray:
    """Improve OCR reliability for UI screenshots."""
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )
    return thr

def ocr_text(pil_img: Image.Image) -> str:
    processed = preprocess_for_ocr(pil_img)
    config = "--psm 6"
    txt = pytesseract.image_to_string(processed, config=config)
    txt = txt.replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    return txt


# -----------------------------
# Parsing Helpers (heuristics)
# -----------------------------
def find_first(patterns: List[str], text: str, flags=0) -> Optional[str]:
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return m.group(1).strip()
    return None

def parse_amount_currency(text: str) -> Dict[str, Optional[str]]:
    currency = find_first(
        [
            r"(\bUSD\b|\bEUR\b|\bGBP\b|\bAED\b|\bBDT\b)",
        ],
        text,
        flags=re.MULTILINE,
    )

    amt = find_first(
        [
            r"Total\s*\(USD\)\s*:\s*([\d,]+\.\d{2})",
            r"Total\s*:\s*([\d,]+\.\d{2})\s*(?:USD|EUR|GBP|AED|BDT)\b",
            r"\b([\d,]+\.\d{2})\s*(USD|EUR|GBP|AED|BDT)\b",
        ],
        text,
        flags=re.IGNORECASE,
    )

    if not amt:
        m = re.search(r"\b([\d,]+\.\d{2})\s*(USD|EUR|GBP|AED|BDT)\b", text)
        if m:
            amt = m.group(1)

    if not currency:
        m = re.search(r"\b[\d,]+\.\d{2}\s*([A-Z]{3})\b", text)
        if m:
            currency = m.group(1)

    return {"amount": amt, "currency": currency}

def parse_vendor(text: str) -> Optional[str]:
    return find_first(
        [
            r"Bill\s+\d+\s+from\s+([A-Za-z0-9 &\-,\.\(\)]+)",
            r"Mailing address\s+([A-Za-z0-9 &\-,\.\(\)]+)",
        ],
        text,
        flags=re.IGNORECASE,
    )

def parse_department(text: str) -> Optional[str]:
    return find_first([r"Department\s+([A-Za-z0-9 &\-,]+)"], text, flags=re.IGNORECASE)

def parse_bill_number(text: str) -> Optional[str]:
    return find_first(
        [
            r"Bill number\s+(\d+)",
            r"\bBill\s+(\d{6,})\b",
        ],
        text,
        flags=re.IGNORECASE,
    )

def parse_class_payment_method(text: str) -> Optional[str]:
    if re.search(r"\bClass\s+Bank\b|\bBank\b", text, flags=re.IGNORECASE):
        return "Bank"
    if re.search(r"\bUSDT\b|\bUSDC\b|\bERC20\b|\bTRC20\b|\bTRX\b|\bWallet\b|\bAddress\b", text, flags=re.IGNORECASE):
        return "Alt"
    return None

def extract_description_block(text: str) -> str:
    m = re.search(
        r"Description\s*(.*?)(?:\n\s*Class\b|\n\s*Amount\b|\n\s*Amount\s+USD\b|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return m.group(1).strip() if m else ""

def extract_memo_block(text: str) -> str:
    m = re.search(r"\bMemo\b\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""

def clean_payment_details(desc: str, memo: str) -> str:
    parts = []
    if desc.strip():
        parts.append(desc.strip())
    if memo.strip():
        parts.append(memo.strip())
    out = "\n".join(parts).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out

def compute_fee_usd(payment_details: str) -> str:
    if not payment_details:
        return "–"
    if re.search(r"\bUSDT\b.*\bTRC20\b|\bUSDT\b.*\bTRX\b|\bUSDT\s*TRC20\b|\bUSDT\s*TRX\b", payment_details, flags=re.IGNORECASE):
        return "2"
    return "–"

def to_float_amount(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        return float(s.replace(",", "").strip())
    except Exception:
        return None

def fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "–"
    return f"{x:,.2f}"

def parse_row_from_text(
    text: str,
    link: str,
    proof: str,
    disbursement_date: str,
) -> Dict[str, Any]:
    vendor = parse_vendor(text) or "–"
    dept = parse_department(text) or "–"
    bill_no = parse_bill_number(text) or "–"

    ac = parse_amount_currency(text)
    req_amt = to_float_amount(ac["amount"])
    req_cur = (ac["currency"] or "–").upper()

    payment_method = parse_class_payment_method(text) or "–"

    desc = extract_description_block(text)
    memo = extract_memo_block(text)
    payment_details = clean_payment_details(desc, memo)
    payment_details_display = payment_details.replace("\n", "<br>") if payment_details else "–"

    disb_usd = None
    if req_amt is not None and req_cur == "USD":
        disb_usd = req_amt
    else:
        converted = find_first(
            [r"Amount\s+USD\s+([\d,]+\.\d{2})", r"Amount USD\s*([\d,]+\.\d{2})"],
            text,
            flags=re.IGNORECASE,
        )
        disb_usd = to_float_amount(converted) if converted else None

    fee = compute_fee_usd(payment_details)

    return {
        "Link": link if link else "–",
        "Vendor Name": vendor,
        "Department": dept,
        "Payment Details": payment_details_display,
        "Bill Number": bill_no,
        "Payment Method": payment_method if payment_method in ["Bank", "Alt"] else payment_method,
        "Requested Amount": fmt_money(req_amt) if req_amt is not None else "–",
        "Requested Currency": req_cur,
        "Disburse Amount (USD)": fmt_money(disb_usd) if disb_usd is not None else "–",
        "Charge/Fees (USD)": fee,
        "Total Disbursed Amount (USD)": "",  # ALWAYS BLANK
        "Payment Proof": proof if proof else "–",
        "Disbursement Date": disbursement_date,
        "Disbursement Date (ApprovalMax)": "",  # ALWAYS BLANK per your instruction
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Requisition Portal", layout="wide")

st.title("📘 Requisition Portal")
st.caption("Upload screenshots + links, extract into the strict master output table, and/or add rows manually.")

today_str = dt.date.today().strftime("%d %b %Y")

with st.sidebar:
    st.subheader("Inputs")
    st.text_input("Disbursement Date (auto)", value=today_str, disabled=True)
    st.markdown("---")
    st.info("Upload screenshots in the exact order you want rows to appear.")

uploads = st.file_uploader(
    "Upload bill screenshot(s)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

st.markdown("### Links (match screenshot order)")
link_inputs, proof_inputs = [], []

if uploads:
    for i, f in enumerate(uploads, start=1):
        cols = st.columns([1.2, 2, 2])
        with cols[0]:
            st.write(f"**#{i}** {f.name}")
        with cols[1]:
            link_inputs.append(st.text_input(f"ApprovalMax link for #{i}", key=f"link_{i}"))
        with cols[2]:
            proof_inputs.append(st.text_input(f"Payment proof link for #{i}", key=f"proof_{i}"))

extract_btn = st.button("🚀 Extract from Uploads", type="primary", disabled=not uploads)

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=COLUMNS)

if extract_btn and uploads:
    rows = []
    for idx, f in enumerate(uploads):
        img = Image.open(f)
        text = ocr_text(img)

        link = link_inputs[idx] if idx < len(link_inputs) else ""
        proof = proof_inputs[idx] if idx < len(proof_inputs) else ""

        rows.append(
            parse_row_from_text(
                text=text,
                link=link,
                proof=proof,
                disbursement_date=today_str,
            )
        )

    new_df = pd.DataFrame(rows, columns=COLUMNS)
    st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)

st.markdown("### Manual Add / Edit")
st.write("Add rows manually or edit extracted rows. Column order is locked.")

edited_df = st.data_editor(
    st.session_state.df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Payment Details": st.column_config.TextColumn(width="large"),
        "Total Disbursed Amount (USD)": st.column_config.TextColumn(disabled=True),
        "Disbursement Date (ApprovalMax)": st.column_config.TextColumn(disabled=True),
        "Disbursement Date": st.column_config.TextColumn(disabled=True),
    },
)

def enforce_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[COLUMNS]

    df["Total Disbursed Amount (USD)"] = ""
    df["Disbursement Date (ApprovalMax)"] = ""
    df["Disbursement Date"] = today_str

    for c in ["Link", "Payment Proof", "Charge/Fees (USD)"]:
        df[c] = df[c].replace({None: "–", "": "–"}).fillna("–")

    return df

final_df = enforce_rules(edited_df)
st.session_state.df = final_df

st.markdown("### Output (Strict Table)")
st.dataframe(final_df, use_container_width=True, hide_index=True)

csv_bytes = final_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="requisition_portal_output.csv", mime="text/csv")

xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
    final_df.to_excel(writer, index=False, sheet_name="Output")

st.download_button(
    "⬇️ Download Excel",
    data=xlsx_buf.getvalue(),
    file_name="requisition_portal_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown("---")
st.caption("v1 uses OCR heuristics. If you want near-perfect extraction, we can upgrade to a Vision-LLM extractor while keeping your exact output rules.")
