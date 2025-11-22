import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset



class BaseAntibodyDataset(Dataset):
    def __init__(self, embed_file, desc_file, labels_file=None, which=[0], fold=slice(None)):
        # Load embeddings & descriptors
        self.features = np.load(embed_file)[fold]
        self.desc = np.load(desc_file)[fold]

        # Load selected label columns
        if labels_file:
            self.labels = np.load(labels_file)[fold][:, which]
        else:
            self.labels = np.ones(len(self.desc))
        if self.labels.ndim == 1:
            self.labels = self.labels[:, None]

        # --- Filter out samples with ANY NaNs across the selected targets ---
        valid_mask = ~np.isnan(self.labels).any(axis=1)
        self.features = self.features[valid_mask]
        self.desc = self.desc[valid_mask]
        self.labels = self.labels[valid_mask]

        print(f"Loaded {len(self.features)} valid samples (filtered NaNs across {len(which)} targets)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float),
            torch.tensor(self.desc[idx], dtype=torch.float),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )


class WarmupDataset(Dataset):
    """
    Warmup dataset:
    - Uses AbLang embeddings (NxD)
    - Uses zero descriptors of same shape as desc_file
    - Supports multitask labels via 'which'
    - Filters out samples with NaNs in *any* selected label column
    """
    def __init__(self, embed_file, desc_file, labels_file, which=[0], fold=slice(None)):
        # 1. Load embeddings
        self.features = np.load(embed_file)[fold]

        # 2. Create zero descriptors same shape as template
        desc_template = np.load(desc_file)
        self.desc = np.zeros_like(desc_template)[fold]

        # 3. Load label columns
        self.labels = np.load(labels_file)[fold][:, which]
        if self.labels.ndim == 1:
            self.labels = self.labels[:, None]

        # 4. Filter out any sample with NaN in any selected column
        valid_mask = ~np.isnan(self.labels).any(axis=1)
        self.features = self.features[valid_mask]
        self.desc = self.desc[valid_mask]
        self.labels = self.labels[valid_mask]

        print(f"Loaded {len(self.features)} warmup samples (filtered NaNs across {len(which)} targets)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float),
            torch.tensor(self.desc[idx], dtype=torch.float),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )


def collate_batch(batch):
    embeds, descs, labels = zip(*batch)
    embeds = torch.stack(embeds)
    descs = torch.stack(descs)
    labels = torch.stack(labels)
    return embeds, descs, labels