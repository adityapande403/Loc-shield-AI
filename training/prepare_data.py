"""Dataset preparation, synthetic generation, auditing, and leak-free template-level splitting.

Produces train, val, and frozen held-out test sets covering English, Hindi (Devanagari),
Romanized Hindi (Hinglish), code-mixed customer support text, and difficult negatives.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Comprehensive multilingual entity dictionaries
PERSON_NAMES_EN = [
    "Rahul Sharma", "Priya Patel", "Amit Kumar", "Neha Singh", "Vikram Malhotra",
    "Ananya Iyer", "Rajesh Gupta", "Sneha Mukherjee", "Arjun Reddy", "Pooja Verma",
    "Siddharth Rao", "Deepika Padukone", "Suresh Raina", "Meera Nambiar", "Karan Johar",
    "Sunita Devi", "Manish Sisodia", "Tanvi Joshi", "Aditya Deshmukh", "Swati Kulkarni",
    "David Miller", "Sarah Jenkins", "Michael Chang", "Emily Watson", "Alex Rivera"
]

PERSON_NAMES_HI = [
    "राहुल शर्मा", "प्रिया पटेल", "अमित कुमार", "नेहा सिंह", "विक्रम मल्होत्रा",
    "अनन्या अय्यर", "राजेश गुप्ता", "स्नेहा मुखर्जी", "अर्जुन रेड्डी", "पूजा वर्मा",
    "सिद्धार्थ राव", "सुनीता देवी", "मनीष सिसोदिया", "तन्वी जोशी", "आदित्य देशमुख"
]

LOCATIONS_EN = [
    "Bengaluru", "Mumbai", "New Delhi", "Hyderabad", "Pune", "Chennai", "Kolkata",
    "Ahmedabad", "Jaipur", "Lucknow", "Indore", "Chandigarh", "Kochi", "Noida",
    "London", "New York", "Singapore", "Dubai", "San Francisco"
]

LOCATIONS_HI = [
    "बेंगलुरु", "मुंबई", "नई दिल्ली", "हैदराबाद", "पुणे", "चेन्नई", "कोलकाता",
    "अहमदाबाद", "जयपुर", "लखनऊ", "इंदौर", "चंडीगढ़", "कोच्चि", "नोएडा"
]

ORGANIZATIONS_EN = [
    "Tata Consultancy Services", "Infosys", "Wipro", "HDFC Bank", "State Bank of India",
    "Reliance Industries", "ICICI Bank", "Flipkart", "Zomato", "Swiggy",
    "Google", "Microsoft", "Amazon", "Apple", "Paytm", "Delhi University", "AIIMS Delhi"
]

ORGANIZATIONS_HI = [
    "टाटा कंसल्टेंसी सर्विसेज", "इंफोसिस", "विप्रो", "एचडीएफसी बैंक", "भारतीय स्टेट बैंक",
    "रिलायंस इंडस्ट्रीज", "आईसीआईसीआई बैंक", "फ्लिपकार्ट", "जोमैटो", "स्विगी"
]

ADDRESSES_EN = [
    "Flat 402, Shanti Heights, Sector 62, Noida, UP 201301",
    "12/A MG Road, Near Metro Station, Indiranagar, Bengaluru 560038",
    "Plot 88, Jubilee Hills, Road No 36, Hyderabad 500033",
    "Shop 14, Linking Road, Bandra West, Mumbai 400050",
    "House 25, Park Street, Near South City Mall, Kolkata 700016",
    "742 Evergreen Terrace, Springfield, OR 97477",
    "10 Downing Street, Westminster, London SW1A 2AA",
    "B-104, Sunrise Apartments, Viman Nagar, Pune 411014",
    "G-5, Connaught Place, Inner Circle, New Delhi 110001",
    "Villa 9, Palm Meadows, Whitefield, Bengaluru 560066"
]

ADDRESSES_HI = [
    "फ्लैट ४०२, शांति हाइट्स, सेक्टर ६२, नोएडा",
    "१२/ए एमजी रोड, मेट्रो स्टेशन के पास, इंदिरानगर, बेंगलुरु",
    "प्लाट ८८, जुबली हिल्स, रोड नंबर ३६, हैदराबाद",
    "दुकान १४, लिंकिंग रोड, बांद्रा वेस्ट, मुंबई",
    "मकान २५, पार्क स्ट्रीट, साउथ सिटी मॉल के पास, कोलकाता",
    "बी-१०४, सनराइज अपार्टमेंट्स, विमान नगर, पुणे"
]

# Template pools partitioned by split to GUARANTEE template-level isolation
# (No template in train will ever appear in val or test)

TRAIN_TEMPLATES = [
    # English
    ("Customer {person} reported an issue from {location} regarding account at {org}.", ["person", "location", "org"]),
    ("Please dispatch the shipment to {person} at {address}.", ["person", "address"]),
    ("Meeting scheduled with {person} from {org} at our {location} branch.", ["person", "org", "location"]),
    ("Invoice billed to {person}, residing at {address}.", ["person", "address"]),
    ("The senior manager {person} has joined {org} in {location}.", ["person", "org", "location"]),
    ("Kindly verify the identity documents for {person} before onboarding to {org}.", ["person", "org"]),
    ("Delivery partner contacted {person} regarding parcel delivery at {address}.", ["person", "address"]),
    ("Support ticket #4920 opened by {person} from {location}.", ["person", "location"]),
    ("Board of directors announced that {person} will lead {org}.", ["person", "org"]),
    ("Property registered under the name of {person} located at {address}.", ["person", "address"]),
    
    # Romanized Hindi (Hinglish)
    ("Mera naam {person} hai aur main {location} me rehta hu.", ["person", "location"]),
    ("Kripya ye document {person} ko forward karein jo {org} me kaam karte hain.", ["person", "org"]),
    ("Courier parcel {person} ke address {address} par bhej diya gaya hai.", ["person", "address"]),
    ("Aapka loan application {person} ke naam se {org} me submit ho gaya hai.", ["person", "org"]),
    ("Main {person} baat kar raha hu {location} se.", ["person", "location"]),
    ("Hamari meeting {person} ke saath {location} office me fix hui hai.", ["person", "location"]),
    
    # Hindi (Devanagari)
    ("ग्राहक {person_hi} ने {location_hi} से संपर्क किया है।", ["person_hi", "location_hi"]),
    ("कृपया यह पार्सल {person_hi} को {address_hi} पर भेजें।", ["person_hi", "address_hi"]),
    ("श्री {person_hi} {org_hi} के मुख्य कार्यकारी अधिकारी हैं।", ["person_hi", "org_hi"]),
    ("मेरा नाम {person_hi} है और मैं {location_hi} में रहता हूँ।", ["person_hi", "location_hi"]),
    ("दस्तावेज़ सत्यापन के लिए {person_hi} {org_hi} कार्यालय उपस्थित हुए।", ["person_hi", "org_hi"]),
    ("{person_hi} का नया पता {address_hi} है।", ["person_hi", "address_hi"]),
]

VAL_TEMPLATES = [
    # English
    ("Contact {person} at {org} regarding the upcoming contract in {location}.", ["person", "org", "location"]),
    ("Shipment addressed to {person} at {address} is out for delivery.", ["person", "address"]),
    ("Employee verification request for {person} working at {org}.", ["person", "org"]),
    ("User {person} logged in from {location}.", ["person", "location"]),
    
    # Romanized Hindi
    ("{person} ne bataya ki unka office {location} me {org} ke paas hai.", ["person", "location", "org"]),
    ("Payment confirmation slip {person} ko {address} par deliver karein.", ["person", "address"]),
    
    # Hindi (Devanagari)
    ("{person_hi} ने {org_hi} में खाता खोलने के लिए आवेदन किया।", ["person_hi", "org_hi"]),
    ("{person_hi} का निवास स्थान {address_hi} में स्थित है।", ["person_hi", "address_hi"]),
]

TEST_TEMPLATES = [
    # Completely independent frozen test templates!
    ("CONFIDENTIAL: The audit of {org} was conducted by external consultant {person} in {location}.", ["org", "person", "location"]),
    ("Return goods notification: Customer {person} returned the package from {address}.", ["person", "address"]),
    ("Executive briefing prepared for {person} at headquarters of {org}.", ["person", "org"]),
    ("Incident report logged by security officer {person} at {location} facility.", ["person", "location"]),
    ("Legal notice served to {person} at registered residence {address}.", ["person", "address"]),
    
    # Hinglish test
    ("Kya aap {person} ko jante hain jo {org} me {location} branch sambhalte hain?", ["person", "org", "location"]),
    ("Order cancel karne ka anurodh {person} dwara {address} se aaya.", ["person", "address"]),
    
    # Devanagari test
    ("गोपनीय सूचना: {person_hi} ने {location_hi} में {org_hi} की नई शाखा का उद्घाटन किया।", ["person_hi", "location_hi", "org_hi"]),
    ("अदालती समन {person_hi} को उनके पते {address_hi} पर तामील किया गया।", ["person_hi", "address_hi"]),
    ("नमस्ते, मेरा नाम {person_hi} है और मैं {org_hi} का प्रतिनिधि हूँ।", ["person_hi", "org_hi"]),
]

# Difficult Negatives (Non-PII sentences)
TRAIN_NEGATIVES = [
    "The quarterly financial earnings report will be published on Friday morning.",
    "System memory utilization reached 84% during peak database query execution.",
    "Error 404: The requested resource could not be found on this application server.",
    "Please ensure all unit tests and integration benchmarks pass before merging code.",
    "The quick brown fox jumps over the lazy dog in the sunny park.",
    "Natural language processing models require sufficient training data and proper tokenization.",
    "सभी सर्वर सामान्य रूप से कार्य कर रहे हैं और कोई व्यवधान नहीं है।",
    "डेटाबेस बैकअप प्रक्रिया सफलतापूर्वक पूर्ण हो गई है।",
]

VAL_NEGATIVES = [
    "कृपया निर्देशों को ध्यान से पढ़ें और नियमों का पालन करें।",
    "Aaj mausam bahut accha hai aur baarish hone ki sambhavna hai.",
    "System log: CPU temperature normalized after thread pool throttling.",
    "The standard operating procedure has been revised for this quarter.",
]

TEST_NEGATIVES = [
    "Machine learning transformers utilize self-attention mechanisms across layers.",
    "यह सॉफ्टवेयर केवल स्थानीय रूप से कार्य करता है और कोई बाहरी कॉल नहीं करता।",
    "Network connection latency measured at 12 milliseconds over loopback interface.",
    "Continuous integration pipelines automatically validate pull requests against main.",
]


def generate_sample(template: str, slot_keys: List[str], rng: random.Random) -> Dict[str, Any]:
    """Instantiate a template and compute exact character offsets for each entity span."""
    slots: Dict[str, str] = {}
    entities: List[Dict[str, Any]] = []

    # Map slot names to values and entity types
    for key in slot_keys:
        if key == "person":
            val = rng.choice(PERSON_NAMES_EN)
            etype = "PERSON"
        elif key == "person_hi":
            val = rng.choice(PERSON_NAMES_HI)
            etype = "PERSON"
        elif key == "location":
            val = rng.choice(LOCATIONS_EN)
            etype = "LOCATION"
        elif key == "location_hi":
            val = rng.choice(LOCATIONS_HI)
            etype = "LOCATION"
        elif key == "org":
            val = rng.choice(ORGANIZATIONS_EN)
            etype = "ORGANIZATION"
        elif key == "org_hi":
            val = rng.choice(ORGANIZATIONS_HI)
            etype = "ORGANIZATION"
        elif key == "address":
            val = rng.choice(ADDRESSES_EN)
            etype = "ADDRESS"
        elif key == "address_hi":
            val = rng.choice(ADDRESSES_HI)
            etype = "ADDRESS"
        else:
            continue
        slots[key] = (val, etype)

    # Reconstruct text by scanning template tokens
    # e.g. "Customer {person} reported from {location}."
    import re
    pattern = re.compile(r"\{([a-zA-Z0-9_]+)\}")
    last_idx = 0
    built_text = []
    
    for match in pattern.finditer(template):
        slot_name = match.group(1)
        # Append text preceding the placeholder
        built_text.append(template[last_idx:match.start()])
        curr_offset = len("".join(built_text))
        
        val, etype = slots[slot_name]
        start = curr_offset
        end = curr_offset + len(val)
        built_text.append(val)
        
        entities.append({
            "start": start,
            "end": end,
            "label": etype,
            "text": val
        })
        last_idx = match.end()

    built_text.append(template[last_idx:])
    final_text = "".join(built_text)

    # Verification: assert text slice matches
    for ent in entities:
        assert final_text[ent["start"]:ent["end"]] == ent["text"], f"Offset mismatch in '{final_text}'"

    return {
        "text": final_text,
        "entities": entities,
        "is_negative": False
    }


def build_split(
    templates: List[Tuple[str, List[str]]],
    num_samples: int,
    seed: int,
    negatives_pool: List[str],
    include_negatives: int = 0
) -> List[Dict[str, Any]]:
    """Generate dataset split with specified template pool and difficult negatives."""
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []

    # 1. Generate positive samples across templates
    for i in range(num_samples):
        tmpl, slot_keys = rng.choice(templates)
        sample = generate_sample(tmpl, slot_keys, rng)
        samples.append(sample)

    # 2. Add negative samples from the disjoint negative pool
    for i in range(include_negatives):
        neg_text = rng.choice(negatives_pool)
        samples.append({
            "text": neg_text,
            "entities": [],
            "is_negative": True
        })

    rng.shuffle(samples)
    return samples


def audit_dataset(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Perform leakage checks, distribution statistics, and quality audit."""
    train_texts = {s["text"] for s in train_data}
    val_texts = {s["text"] for s in val_data}
    test_texts = {s["text"] for s in test_data}

    # Overlap / Leakage checks
    train_val_overlap = len(train_texts.intersection(val_texts))
    train_test_overlap = len(train_texts.intersection(test_texts))
    val_test_overlap = len(val_texts.intersection(test_texts))

    def count_entities(data: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in data:
            for ent in s["entities"]:
                lbl = ent["label"]
                counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    audit_report = {
        "sample_counts": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
            "total": len(train_data) + len(val_data) + len(test_data)
        },
        "leakage_audit": {
            "train_val_text_overlap": train_val_overlap,
            "train_test_text_overlap": train_test_overlap,
            "val_test_text_overlap": val_test_overlap,
            "template_level_isolation_enforced": True,
        },
        "entity_distributions": {
            "train": count_entities(train_data),
            "val": count_entities(val_data),
            "test": count_entities(test_data),
        }
    }
    return audit_report


def main():
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    eval_dir = data_dir / "evaluation"
    processed_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("Generating reproducible synthetic and multilingual datasets...")
    # Generate splits
    train_samples = build_split(TRAIN_TEMPLATES, num_samples=1200, seed=42, negatives_pool=TRAIN_NEGATIVES, include_negatives=150)
    val_samples = build_split(VAL_TEMPLATES, num_samples=300, seed=123, negatives_pool=VAL_NEGATIVES, include_negatives=40)
    test_samples = build_split(TEST_TEMPLATES, num_samples=300, seed=999, negatives_pool=TEST_NEGATIVES, include_negatives=40)

    # Save to JSONL
    def write_jsonl(filepath: Path, data: List[Dict[str, Any]]):
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "val.jsonl"
    test_path = eval_dir / "test.jsonl"

    write_jsonl(train_path, train_samples)
    write_jsonl(val_path, val_samples)
    write_jsonl(test_path, test_samples)

    # Run and save audit
    audit = audit_dataset(train_samples, val_samples, test_samples)
    audit_file = eval_dir / "dataset_audit_report.json"
    audit_file.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Data generation complete!")
    print(f"Train samples: {len(train_samples)} -> {train_path}")
    print(f"Val samples:   {len(val_samples)} -> {val_path}")
    print(f"Test samples:  {len(test_samples)} -> {test_path}")
    print(f"Audit report saved -> {audit_file}")
    print("Audit Summary:", json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
