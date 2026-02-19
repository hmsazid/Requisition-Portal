import re
import io
import datetime as dt
from typing import Optional, Dict, Any, List

import pandas as pd
import streamlit as st
from PIL import Image
import pytesseract


# =====================================================
# REQUIRED MASTER COLUMN ORDER (DO NOT CHANGE)
# =====================================================
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
    "Total Disbursed Amount (USD)",  # ALWAYS BLANK
    "Payment Proof",
    "Disbursement Date",
    "Disbursement Date (ApprovalMax)",  # ALWAYS BLANK
]


# =====================================================
# OCR FUNCTION
# =====================================================
def ocr_text(image: Image.Image) -> str:
    image = image.convert("RGB")
    text = pytesseract.image_to_string(image, config="--psm 6")

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def find_first(patterns: List[str], text: str, flags=0) -> Optional[str]:
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return m.group(1).strip()
    return None


def parse_vendor(text: str) -> str:
    return find_first(
        [
            r"Bill\s+\d+\s+from\s+([A-Za-z0-9 &\-,\.\(\)]+)",
            r"Mailing address\s+([A-Za-z0-9 &\-,\.\(\)]+)",
        ],
        text,
        flags=re.IGNORECASE,
    ) or "–"


def parse_department(text: str) -> str:
    return find_first(
        [r"Department\s+([A-Za-z0-9 &\-,]+)"],
        text,
        flags=re.IGNORECASE,
    ) or "–"


def parse_bill_number(text: str) -> str:
    return find_first(
        [
            r"Bill number\s+(\d+)",
            r"\bBill\s+(\d{6,})\b",
        ],
        text,
        flags=re.IGNORECASE,
    ) or "–"


def parse_amount_currency(text: str) -> Dict[str, Optional[str]]:
    amount = find_first(
        [
            r"Total\s*\(USD\)\s*:\s*([\d,]+\.\d{2})",
            r"\b([\d,]+\.\d{2})\s*(USD|EUR|GBP|AED|BDT)\b",
        ],
        text,
        flags=re.IGNORECASE,
    )

    currency = find_first(
        [r"\b(USD|EUR|GBP|AED|BDT)\b"],
        text,
        flags=re.IGNORECASE,
    )

    return {"amount": amount, "currency": currency}


def parse_payment_method(text: str) -> str:
    if re.search(r"\bUSDT\b|\bUSDC\b|\bERC20\b|\bTRC20\b|\bTRX\b|\bWallet\b", text, re.IGNORECASE):
        return "Alt"
    if re.search(r"\bBank\b|\bIBAN\b|\bSWIFT\b", text, re.IGNORECASE):
        return "Bank"
    return "–"


def extract_description(text: str) -> str:
    desc = find_first(
        [r"Description\s*(.*?)(?:\n\s*Class\b|\n\s*Amount\b|$)"],
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ) or ""

    memo = find_first(
        [r"\bMemo\b\s*(.*)$"],
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ) or ""

    combined = "\n".join([d for d in [desc, memo] if d.strip()])
    combined = re.sub(r"\n{3,}", "\n\n", combined).strip()

    return combined if combined else "–"


def compute_fee(payment_details: str) -> str:
    if re.search(r"\bUSDT\b.*\bTRC20\b|\bUSDT\b.*\bTRX\b|\bUSDT\s*TRC20\b|\bUSDT\s*TRX\b", payment_details, re.IGNORECASE):
        return "2"
    return "–"


def format_money(val: Optional[str]) -> str:
    if not val:
        return "–"
    try:
        return f"{float(val.replace(',', '')):,.2f}"
    except:
        return "–"


# =====================================================
# PARSE SINGLE ROW
# =====================================================
def parse_row(text: str, link: str, proof: str, disb_date: str) -> Dict[str, Any]:

    vendor = parse_vendor(text)
    department = parse_department(text)
    bill_no = parse_bill_number(text)

    ac = parse_amount_currency(text)
    req_amount = format_money(ac["amount"])
    currency = (ac["currency"] or "–").upper()

    payment_method = parse_payment_method(text)
    payment_details = extract_description(text).replace("\n", "<br>")

    disburse_usd = req_amount if currency == "USD" else "–"
    fee = compute_fee(payment_details)

    return {
        "Link": link if link else "–",
        "Vendor Name": vendor,
        "Department": department,
        "Payment Details": payment_details,
        "Bill Number": bill_no,
        "Payment Method": payment_method,
        "Requested Amount": req_amount,
        "Requested Currency": currency,
        "Disburse Amount (USD)": disburse_usd,
        "Charge/Fees (USD)": fee,
        "Total Disbursed Amount (USD)": "",
        "Payment Proof": proof if proof else "–",
        "Disbursement Date": disb_date,
        "Disbursement Date (ApprovalMax)": "",
    }


# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="Requisition Portal", layout="wide")

st.title("📘 Requisition Portal")

today = dt.date.today().strftime("%d %b %Y")

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=COLUMNS)

st.sidebar.text_input("Disbursement Date", value=today, disabled=True)

uploads = st.file_uploader(
    "Upload Bill Screenshot(s)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

link_inputs = []
proof_inputs = []

if uploads:
    st.subheader("Links & Proof (match screenshot order)")
    for i, file in enumerate(uploads, start=1):
        col1, col2, col3 = st.columns([1, 2, 2])
        with col1:
            st.write(f"**#{i}** {file.name}")
        with col2:
            link_inputs.append(st.text_input(f"ApprovalMax Link #{i}", key=f"link_{i}"))
        with col3:
            proof_inputs.append(st.text_input(f"Payment Proof #{i}", key=f"proof_{i}"))

if st.button("🚀 Extract Data", type="primary") and uploads:

    rows = []

    for idx, file in enumerate(uploads):
        image = Image.open(file)
        text = ocr_text(image)

        link = link_inputs[idx] if idx < len(link_inputs) else ""
        proof = proof_inputs[idx] if idx < len(proof_inputs) else ""

        row = parse_row(text, link, proof, today)
        rows.append(row)

    new_df = pd.DataFrame(rows, columns=COLUMNS)
    st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)

# =====================================================
# MANUAL EDIT SECTION
# =====================================================
st.subheader("Manual Add / Edit")

edited_df = st.data_editor(
    st.session_state.df,
    use_container_width=True,
    num_rows="dynamic",
)

# Enforce Rules
edited_df = edited_df[COLUMNS]
edited_df["Total Disbursed Amount (USD)"] = ""
edited_df["Disbursement Date"] = today
edited_df["Disbursement Date (ApprovalMax)"] = ""

st.session_state.df = edited_df

# =====================================================
# OUTPUT
# =====================================================
st.subheader("Final Output")

st.dataframe(st.session_state.df, use_container_width=True, hide_index=True)

csv = st.session_state.df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv, "requisition_output.csv", "text/csv")

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    st.session_state.df.to_excel(writer, index=False, sheet_name="Output")

st.download_button(
    "⬇️ Download Excel",
    excel_buffer.getvalue(),
    "requisition_output.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
