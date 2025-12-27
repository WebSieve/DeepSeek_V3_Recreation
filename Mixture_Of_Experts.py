import torch
import torch.nn as nn
import torch.nn.functional as f


"""
- calculate affinity to each expert
- select top_k experts with highest affinity
- send tokens to those k experts
- combine experts output
"""


class Mixture_Of_Experts(nn.Module):
    """
    Top K Routing
    Auxiliary loss free load balancing
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_dim: int,
        num_experts: int,
        num_experts_per_token: int,
    ):
        """
        hidden_size : embedding dimension
        intermediate_dim : 2.4x in deepseek i guess
        num_experts : total number of experts
        num_experts_per_token : number of experts activated per token
        """

        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

        # Router (Calculating Affinity)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # shared experts (always activated)
        self.shared_expert_gate = nn.Linear(hidden_size, intermediate_dim, bias=False)
        self.shared_expert_up = nn.Linear(hidden_size, intermediate_dim, bias=False)
        self.shared_expert_down = nn.Linear(intermediate_dim, hidden_size, bias=False)

        # Router experts (Sparsely activated)
        self.experts = nn.ModuleList(
            [self._create_expert() for _ in range(num_experts)]
        )

    def _create_expert(self) -> nn.ModuleDict:
        """
        Creating a single expert network
        Each expert is a feed-forward network with gating
        """

        expert = nn.ModuleDict(
            {
                "gate_projection": nn.Linear(
                    self.hidden_size, self.intermediate_dim, bias=False
                ),
                "up_projection": nn.Linear(
                    self.hidden_size, self.intermediate_dim, bias=False
                ),
                "down_projection": nn.Linear(
                    self.intermediate_dim, self.hidden_size, bias=False
                ),
            }
        )
        return expert

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Applying Mixture Of Experts

        Args:
            hidden_states : (batch_size, seq_len, hidden_size)

        Returns:
            output : (batch_size, seq_len, hidden_size)

        """

        batch_size, seq_len, hidden_size = hidden_states.shape

        # flattening the hidden states (batch_size * seq_len, hidden_size)
        hs_flat = hidden_states.view(-1, hidden_size)

        # Calculating Affinity scores for each expert
        router_logits = self.gate(hs_flat)

        # Selecting top K experts with the highest affinity
        routing_weights, selected_experts = torch.topk(
            router_logits, self.num_experts_per_token, dim=-1
        )
        routing_weights = f.softmax(routing_weights, dim=-1)

        # Passing through shared experts
        shared_gate = f.silu(self.shared_expert_gate(hs_flat))
        shared_up = self.shared_expert_up(hs_flat)
        shared_output = self.shared_expert_down(shared_gate * shared_up)

        # Initializing the output Tensor
        routed_output = torch.zeros_like(hs_flat)

        for i in range(batch_size * seq_len):
            token_experts = selected_experts[i]
            token_weights = routing_weights[i]
            for expert_idx, weight in zip(token_experts, token_weights):
                expert = self.experts[expert_idx]
                gate = f.silu(expert["gate_projection"](hs_flat[i]))
                up = expert["up_projection"](hs_flat[i])
                expert_output = expert["down_projection"](gate * up)

                routed_output[i] += weight * expert_output

        final_output = shared_output + routed_output
        final_output = final_output.view(batch_size, seq_len, hidden_size)
        return final_output


print("- Testing Mixture_Of_Experts...")
test_input = torch.randn(2, 10, 512)
print(f"- shape of test_input : {test_input.shape}")
moe = Mixture_Of_Experts(
    hidden_size=test_input.shape[2],
    intermediate_dim=test_input.shape[2] * 2,
    num_experts=4,
    num_experts_per_token=2,
)
test_output = moe(test_input)
print(f"- shape of test_input : {test_output.shape}")
print(f"- Number of Parameters : {sum(p.numel() for p in moe.parameters()):,}")
