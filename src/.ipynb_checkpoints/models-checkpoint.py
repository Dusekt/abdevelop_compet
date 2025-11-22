import torch.nn as nn


class AntibodyModel2(nn.Module):
    def __init__(self, embed_dim=480, hidden_dim=256, dsp_dim=30, output_dim=1):
        super().__init__()

        # store dims for easy replace_head
        self._embed_dim = embed_dim
        self._dsp_dim = dsp_dim
        self._hidden_dim = hidden_dim
        self._fusion_dim = hidden_dim // 4 + dsp_dim // 4  # after concatenation e+d -> hidden_dim

        # Embedding branch -> outputs hidden_dim//4
        self.ablang_branch = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
        )

        # Descriptor branch -> outputs dsp_dim//4
        self.desc_branch = nn.Sequential(
            nn.Linear(dsp_dim, dsp_dim // 2),
            nn.LayerNorm(dsp_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dsp_dim // 2, dsp_dim // 4),
            nn.LayerNorm(dsp_dim // 4),
            nn.GELU(),
        )

        # Fusion trunk is simply concatenation into fusion_dim (hidden_dim)
        # and a small trunk could be added here if desired; for simplicity we
        # keep the trunk minimal and place final layers into self.out
        # Final prediction head (easy to replace)
        self.out = nn.Sequential(
            nn.Linear(self._fusion_dim, self._fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self._fusion_dim // 2, self._fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self._fusion_dim // 4, self._fusion_dim // 6),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self._fusion_dim // 6, output_dim),
        )

        # initialize weights for all linear modules
        self._init_weights()

    def forward(self, embeds, descs):
        e = self.ablang_branch(embeds)           # (B, hidden_dim//2)
        d = self.desc_branch(descs)              # (B, hidden_dim//2)
        x = torch.cat([e, d], dim=-1)            # (B, hidden_dim)
        return self.out(x)                       # (B, output_dim)

    def replace_head(self, output_dim=1):
        """Replace only the final prediction head (self.out) and reinitialize it."""
        # create a fresh self.out with same input fusion dim
        self.out = nn.Sequential(
            nn.Linear(self._fusion_dim, self._fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self._fusion_dim // 2, self._fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self._fusion_dim // 4, self._fusion_dim // 6),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self._fusion_dim // 6, output_dim),
        )
        # initialize new head weights
        for m in self.out.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def freeze_ablang(self, freeze=True):
        for p in self.ablang_branch.parameters():
            p.requires_grad = not freeze
    def freeze_dsp(self, freeze=True):
        for p in self.desc_branch.parameters():
            p.requires_grad = not freeze

class AntibodyModel3(nn.Module):
    def __init__(self, embed_dim=480, hidden_dim=256, dsp_dim=30, output_dim=1):
        super().__init__()

        # store dims for easy replace_head
        self._embed_dim = embed_dim
        self._dsp_dim = dsp_dim
        self._hidden_dim = hidden_dim
        self._fusion_dim = hidden_dim // 4 + dsp_dim // 4  # after concatenation e+d -> hidden_dim

        # Embedding branch -> outputs hidden_dim//4
        self.ablang_branch = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
        )

        # Descriptor branch -> outputs dsp_dim//4
        self.desc_branch = nn.Sequential(
            nn.Linear(dsp_dim, dsp_dim // 2),
            nn.LayerNorm(dsp_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dsp_dim // 2, dsp_dim // 4),
            nn.LayerNorm(dsp_dim // 4),
            nn.GELU(),
        )

        # Fusion trunk is simply concatenation into fusion_dim (hidden_dim)
        # and a small trunk could be added here if desired; for simplicity we
        # keep the trunk minimal and place final layers into self.out
        # Final prediction head (easy to replace)
        self.con = nn.Sequential(
            nn.Linear(self._fusion_dim, self._fusion_dim),
            nn.GELU()
        )
        
        self.out = nn.Sequential(
            nn.Linear(self._fusion_dim, self._fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self._fusion_dim // 2, self._fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self._fusion_dim // 4, self._fusion_dim // 6),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self._fusion_dim // 6, output_dim)
        )

        # initialize weights for all linear modules
        self._init_weights()

    def forward(self, embeds, descs):
        e = self.ablang_branch(embeds)           # (B, hidden_dim//2)
        d = self.desc_branch(descs)              # (B, hidden_dim//2)
        x = torch.cat([e, d], dim=-1)            # (B, hidden_dim)
        x = self.con(x)
        return self.out(x)                       # (B, output_dim)

    def replace_head(self, output_dim=1):
        """Replace only the final prediction head (self.out) and reinitialize it."""
        # create a fresh self.out with same input fusion dim
        self.out = nn.Sequential(
            nn.Linear(self._fusion_dim, self._fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self._fusion_dim // 2, self._fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self._fusion_dim // 4, self._fusion_dim // 6),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self._fusion_dim // 6, output_dim)
        )
        # initialize new head weights
        for m in self.out.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def freeze_ablang(self, freeze=True):
        for p in self.ablang_branch.parameters():
            p.requires_grad = not freeze
    def freeze_dsp(self, freeze=True):
        for p in self.desc_branch.parameters():
            p.requires_grad = not freeze

class AntibodyModel4(nn.Module):
    def __init__(self, embed_dim=480, hidden_dim=256, dsp_dim=30, output_dim=1):
        super().__init__()

        # store dims for easy replace_head
        self._embed_dim = embed_dim
        self._dsp_dim = dsp_dim
        self._hidden_dim = hidden_dim
        self._fusion_dim = hidden_dim // 4 + dsp_dim // 4  # after concatenation e+d -> hidden_dim

        # Embedding branch -> outputs hidden_dim//4
        self.ablang_branch = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU()
        )

        # Descriptor branch -> outputs dsp_dim//4
        self.desc_branch = nn.Sequential(
            nn.Linear(dsp_dim, dsp_dim // 2),
            nn.LayerNorm(dsp_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dsp_dim // 2, dsp_dim // 4),
            nn.LayerNorm(dsp_dim // 4),
            nn.GELU()
        )

        # Fusion trunk is simply concatenation into fusion_dim (hidden_dim)
        # and a small trunk could be added here if desired; for simplicity we
        # keep the trunk minimal and place final layers into self.out
        # Final prediction head (easy to replace)
        self.con = nn.Sequential(
            nn.Linear(self._fusion_dim, self._fusion_dim),
            nn.GELU()
        )
        
        self.out = nn.Sequential(
            nn.Linear(self._fusion_dim, self._fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self._fusion_dim // 2, self._fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self._fusion_dim // 4, self._fusion_dim // 6),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self._fusion_dim // 6, output_dim)
        )

        # initialize weights for all linear modules
        self._init_weights()

    def forward(self, embeds, descs):
        e = self.ablang_branch(embeds)           # (B, hidden_dim//2)
        d = self.desc_branch(descs)              # (B, hidden_dim//2)
        x = torch.cat([e, d], dim=-1)            # (B, hidden_dim)
        x = self.con(x)
        return self.out(x)                       # (B, output_dim)

    def replace_head(self, output_dim=1):
        """Replace only the final prediction head (self.out) and reinitialize it."""
        # create a fresh self.out with same input fusion dim
        self.out = nn.Sequential(
            nn.Linear(self._fusion_dim, self._fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self._fusion_dim // 2, self._fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self._fusion_dim // 4, self._fusion_dim // 6),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self._fusion_dim // 6, output_dim)
        )
        # initialize new head weights
        for m in self.out.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def freeze_ablang(self, freeze=True):
        for p in self.ablang_branch.parameters():
            p.requires_grad = not freeze
    def freeze_dsp(self, freeze=True):
        for p in self.desc_branch.parameters():
            p.requires_grad = not freeze


class AntibodyModel5(nn.Module):
    def __init__(self, embed_dim=480, hidden_dim=256, dsp_dim=30, output_dim=1):
        super().__init__()

        # store dims for easy replace_head
        self._embed_dim = embed_dim
        self._dsp_dim = dsp_dim
        self._hidden_dim = hidden_dim
        self._fusion_dim = hidden_dim // 4 +  dsp_dim // 2 # after concatenation e+d -> hidden_dim

        # Embedding branch -> outputs hidden_dim//2
        self.ablang_branch = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
        )

        # Descriptor branch -> outputs hidden_dim//2
        self.desc_branch = nn.Sequential(
            nn.Linear(dsp_dim, dsp_dim // 2),
            nn.LayerNorm(dsp_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dsp_dim // 2, dsp_dim // 2),
            nn.LayerNorm(dsp_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # Fusion trunk is simply concatenation into fusion_dim (hidden_dim)
        # and a small trunk could be added here if desired; for simplicity we
        # keep the trunk minimal and place final layers into self.out
        # Final prediction head (easy to replace)
        self.out = nn.Sequential(
            nn.Linear(self._fusion_dim, self._fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self._fusion_dim // 2, output_dim),
        )

        # initialize weights for all linear modules
        self._init_weights()

    def forward(self, embeds, descs):
        e = self.ablang_branch(embeds)           # (B, hidden_dim//2)
        d = self.desc_branch(descs)              # (B, hidden_dim//2)
        x = torch.cat([e, d], dim=-1)            # (B, hidden_dim)
        return self.out(x)                       # (B, output_dim)

    def replace_head(self, output_dim=1):
        """Replace only the final prediction head (self.out) and reinitialize it."""
        # create a fresh self.out with same input fusion dim
        self.out = nn.Sequential(
            nn.Linear(self._fusion_dim, self._fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self._fusion_dim // 2, output_dim),
        )
        # initialize new head weights
        for m in self.out.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def freeze_ablang(self, freeze=True):
        for p in self.ablang_branch.parameters():
            p.requires_grad = not freeze
    def freeze_dsp(self, freeze=True):
        for p in self.desc_branch.parameters():
            p.requires_grad = not freeze