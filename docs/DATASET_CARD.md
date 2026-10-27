# Dataset Card: LocalShield Multilingual PII Dataset

## 1. Summary
The dataset contains structured and unstructured text in English, Hindi (Devanagari script), Romanized Hindi (Hinglish), and customer support conversations.

## 2. Split Methodology & Leakage Audit
To eliminate data contamination and template leakage:
- **Template-Level Isolation**: Templates used in the training split were partitioned from validation and test splits.
- **Disjoint Negatives**: Negative examples were separated into disjoint pools across splits.
- **Audit Verification**:
  - `train_val_text_overlap`: 0
  - `train_test_text_overlap`: 0
  - `val_test_text_overlap`: 0

## 3. Sample Counts & Entity Distributions
| Split | Samples | PERSON | ADDRESS | ORGANIZATION | LOCATION |
|---|---|---|---|---|---|
| Train | 1,350 | 1,200 | 382 | 487 | 495 |
| Val | 340 | 300 | 120 | 154 | 99 |
| Test | 340 | 300 | 131 | 149 | 110 |
| **Total** | **2,030** | **1,800** | **633** | **790** | **704** |

## 4. Privacy & Synthetic Guarantees
- All identifiers (Aadhaar, PAN, Credit Cards, API Keys) are synthetic and do not correspond to real individuals.
- Aadhaar numbers use valid Verhoeff check digits generated using synthetic prefixes.
- Credit cards use valid Luhn checksums on test prefixes.
