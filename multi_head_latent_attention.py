from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as f


class Multi_Head_Latent_Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.qk_nope_head_dim = qk_rope_head_dim
        self.q_head_dim = qk_rope_head_dim + self.qk_nope_head_dim

        # key, value "A" projection with multi query attention style
        # part of lora rank decomposition
        self.down_projection = nn.Linear(
            hidden_size, kv_lora_rank + self.qk_rope_head_dim, bias=False
        )

        # key, value "B" projection (part of lora rank decomposition)
        self.up_projection = nn.Linear(
            self.kv_lora_rank,
            num_heads * (self.qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        # Query standard projection
        self.q_projection = nn.Linear(
            hidden_size, num_heads * self.q_head_dim, bias=False
        )

        self.output_projection = nn.Linear(
            num_heads * v_head_dim, hidden_size, bias=False
        )

        self.scale = qk_rope_head_dim**-(0.5)

    def forward(
        self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_state.shape

        # down projection
        kv_down = self.down_projection(hidden_state)

        # kv_compressed -> (batch_size, seq_len, kv_lora_rank)
        # k_rope -> (batch_size, seq_len, qk_rope_head_dim)
        kv_compressed, k_rope = kv_down.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # kv_up (batch_size, seq_len, kv_lora_rank) -> (batch_size, seq_len, num_heads * (qk_nope_head_dim + v_head_dim))
        kv_up = self.up_projection(kv_compressed)
        kv = kv_up.view(
            batch_size,
            seq_len,
            self.num_heads,
            (self.qk_nope_head_dim + self.v_head_dim),
        )
        kv = kv.transpose(1, 2)
        # k_nope -> (batch_size, num_heads, seq_len, qk_nope_head_dim)
        # v -> (batch_size, num_heads, seq_len, v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q = self.q_projection(
            hidden_state
        )  # (batch, seq, hidden_size) -> (batch, seq, num_heads * q_head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        q = q.transpose(1, 2)
        q_rope, q_nope = q.split([self.qk_rope_head_dim, self.qk_nope_head_dim], dim=-1)
        q_full = torch.cat([q_rope, q_nope], dim=-1)

        # k_rope -> (batch_size, seq_len, qk_rope_head_dim)
        # k_nope -> (batch_size, num_heads, seq_len, qk_nope_head_dim)
        k_rope_expanded = k_rope.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        k_full = torch.cat([k_rope_expanded, k_nope], dim=-1)

        # Calculating scores
        attention_scores = torch.matmul(q_full, k_full.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        contextual = f.softmax(attention_scores, dim=-1)
        contextual = torch.matmul(contextual, v)
        contextual = contextual.transpose(1, 2).contiguous()
        contextual = contextual.view(
            batch_size, seq_len, self.num_heads * self.v_head_dim
        )

        output = self.output_projection(contextual)

        return output


print("testing Multi_Head_Latent_Attention...")
test_input = torch.randn(2, 10, 512)
print(f"test_input shape : {test_input.shape}")
mhla = Multi_Head_Latent_Attention(
    hidden_size=test_input.shape[2],
    num_heads=4,
    kv_lora_rank=5,
    qk_rope_head_dim=4,
    v_head_dim=4,
)
test_output = mhla(test_input)
print(f"test_output shape : {test_output.shape}")
print(f"number of parameters : {sum(p.numel() for p in mhla.parameters()):,}")
