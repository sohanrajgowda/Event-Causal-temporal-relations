import torch.nn as nn
from torch_geometric.nn import HypergraphConv
from transformers import BertModel
import torcfrom causal_model import HGATCausalClassifier
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import torch
import h.nn.functional as F

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import stanza
stanza.download("en")
nlp = stanza.Pipeline("en")

def build_hyperedge_index_from_text(text):
    doc = nlp(text)
    node_to_edge = []
    edge_id = 0

    for sent in doc.sentences:
        for word in sent.words:
            if word.head != 0:  # Ignore ROOT
                head_idx = word.head - 1
                child_idx = word.id - 1
                node_to_edge.append((head_idx, edge_id))
                node_to_edge.append((child_idx, edge_id))
                edge_id += 1

    if node_to_edge:
        source_nodes, hyperedges = zip(*node_to_edge)
        hyperedge_index = torch.tensor([list(source_nodes), list(hyperedges)], dtype=torch.long)
    else:
        hyperedge_index = torch.zeros((2, 1), dtype=torch.long)

    return hyperedge_index


class CausalDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        import pandas as pd

        df = pd.read_csv(csv_path)

        self.labels = df["label"].tolist()
        self.sentences = df["input_text"].tolist()

        self.encodings = tokenizer(self.sentences, truncation=True, padding=True)

        # Create dummy or real hyperedge indices for now (update with actual graph logic)
        self.hyperedge_indices = [build_hyperedge_index_from_text(sent) for sent in self.sentences]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "hyperedge_index": self.hyperedge_indices[idx],
        }


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]
    hyperedge_indices = [item['hyperedge_index'] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    # Combine hyperedge_index tensors and shift node indices
    hyperedge_index_combined = []
    node_offset = 0
    for i, edge_index in enumerate(hyperedge_indices):
        edge_index = edge_index.clone()
        edge_index[0, :] += node_offset  # shift node indices
        hyperedge_index_combined.append(edge_index)
        node_offset += input_ids.size(1)  # or len(batch[i]['input_ids'])

    # Concatenate all hyperedge indices
    hyperedge_index = torch.cat(hyperedge_index_combined, dim=1)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels,
        "hyperedge_index": hyperedge_index,
    }


# Initialize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = CausalDataset("causal_classification_dataset.csv", tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True,collate_fn=collate_fn)


model = HGATCausalClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()


# Training loop
def train_model(model, dataloader, optimizer, criterion, device, epochs=5):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            print(batch['hyperedge_index'].shape)
            hyperedge_index = batch['hyperedge_index'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, hyperedge_index)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model

model=train_model(model,dataloader,optimizer,criterion,device='cuda' if torch.cuda.is_available() else 'cpu',epochs=5)
