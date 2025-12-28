from typing import Optional
import torch
import torch.nn as nn

from multi_head_latent_attention import Multi_Head_Latent_Attention as mhla
from Mixture_Of_Experts import Mixture_Of_Experts as moe
from RMS_Norm import RMSNorm


class DS_Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        v_head_dim,
        intermediate_dim,
        num_experts,
        num_experts_per_token,
    ):
        super().__init__()

        self.attn_rms = RMSNorm(hidden_size=hidden_size)
        self.mhla_attn = mhla(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
        )

        self.moe_rms = RMSNorm(hidden_size=hidden_size)
        self.moe = moe(
            hidden_size=hidden_size,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )

    def forward(
        self, hidden_state, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.attn_rms(hidden_state)
        hidden_state = self.mhla_attn(hidden_state, attention_mask)
        hidden_state = residual + hidden_state

        residual = hidden_state
        hidden_state = self.moe_rms(hidden_state)
        hidden_state = self.moe(hidden_state)
        hidden_state = residual + hidden_state

        return hidden_state


print("- Testing DeepSeek V3 Transformer Block...")
block = DS_Block(
    hidden_size=512,
    num_heads=8,
    kv_lora_rank=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
    intermediate_dim=1024,
    num_experts=8,
    num_experts_per_token=2,
)

test_input = torch.randn(2, 10, 512)
output = block(test_input)

print(f"- Input: {test_input.shape}")
print(f"- Output: {output.shape}")
print(f"- Parameters: {sum(p.numel() for p in block.parameters()):,}")
