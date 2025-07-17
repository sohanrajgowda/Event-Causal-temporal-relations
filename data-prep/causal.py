import os
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd

# Unzip the dataset

extract_path = "C:\\Users\\sohan\\Documents\\event-causal-temporal\\alldata\\Causal-TimeBank-TimeML"


# Function to extract causal and non-causal event pairs
def extract_event_pairs_from_tml_etree(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    events = {}
    for event in root.iter("EVENT"):
        eid = event.attrib.get("eid")
        if eid:
            events[eid] = event.text

    event_instances = {}
    for ei in root.iter("MAKEINSTANCE"):
        eiid = ei.attrib.get("eiid")
        eid = ei.attrib.get("eventID")
        if eiid and eid:
            event_instances[eiid] = eid

    eiid_to_trigger = {
        eiid: events[eid]
        for eiid, eid in event_instances.items()
        if eid in events and events[eid] is not None
    }

    # Extract positive pairs from CLINKs
    positive_pairs = []
    clinks = root.findall(".//CLINK")
    clink_set = set()
    for cl in clinks:
        src = cl.attrib.get("eventInstanceID")
        tgt = cl.attrib.get("relatedToEventInstance")
        if src in eiid_to_trigger and tgt in eiid_to_trigger:
            clink_set.add((src, tgt))
            positive_pairs.append({
                'text1': eiid_to_trigger[src],
                'text2': eiid_to_trigger[tgt],
                'label': 1
            })

    # Generate negative pairs
    eiids = list(eiid_to_trigger.keys())
    negative_pairs = []
    for i in range(len(eiids)):
        for j in range(i+1, len(eiids)):
            src, tgt = eiids[i], eiids[j]
            if (src, tgt) not in clink_set and (tgt, src) not in clink_set:
                negative_pairs.append({
                    'text1': eiid_to_trigger[src],
                    'text2': eiid_to_trigger[tgt],
                    'label': 0
                })
            if len(negative_pairs) >= len(positive_pairs):
                break
        if len(negative_pairs) >= len(positive_pairs):
            break

    return positive_pairs, negative_pairs

# Process all TML files
tml_files = [os.path.join(extract_path, f) for f in os.listdir(extract_path) if f.endswith('.tml')]
all_event_pairs = []

for file_path in tml_files:
    pos, neg = extract_event_pairs_from_tml_etree(file_path)
    all_event_pairs.extend(pos + neg)

# Format DataFrame
df = pd.DataFrame(all_event_pairs)
df['input_text'] = df.apply(lambda row: f"{row['text1']} [SEP] {row['text2']}", axis=1)
final_df = df[['input_text', 'label']]
final_df.to_csv("causal_classification_dataset.csv", index=False)
