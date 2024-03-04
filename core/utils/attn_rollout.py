import torch
import torch.nn as nn


class AttentionRollout:
    def __init__(
            self, 
            model: nn.Module, 
            attention_layer_name='attn',
            head_fusion="mean",
            discard_ratio=0.9
        ):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

        # register forward hook `self.get_attention` to get attention weights
        for name, module in self.model.named_modules():
            if name.endswith(attention_layer_name):
                module.register_forward_hook(self.get_attention)

        # list storing attention weights of each layer
        self.attentions = []
    
    """
    @staticmethod
    def rollout(attentions, discard_ratio, head_fusion):
        # attention, size: (N, n_heads, seq_len, seq_len)
        N, n_heads, seq_len, _ = attentions[0].size()
        result = torch.eye(seq_len)
        # result, size: (N, seq_len, seq_len)

        with torch.no_grad():
            for attention in attentions:
            # for attention in attentions[::-1]:
                # attention, size: (N, n_heads, seq_len, seq_len)
                if head_fusion == "mean":
                    attention_heads_fused = attention.mean(axis=1)
                elif head_fusion == "max":
                    attention_heads_fused = attention.max(axis=1)[0]
                elif head_fusion == "min":
                    attention_heads_fused = attention.min(axis=1)[0]
                else:
                    raise "Attention head fusion type Not supported"
                # attention_fused, size: (N, seq_len, seq_len)

                '''
                def set_smallest_k_to_zero(input_tensor, k):
                    values, indices = torch.topk(input_tensor, k, largest=False)
                    rows = torch.arange(input_tensor.size(0)).unsqueeze(1)
                    input_tensor[rows, indices] = 0
                    return input_tensor
                
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                set_smallest_k_to_zero(flat, int(flat.size(-1)*discard_ratio))
                '''
                
                '''
                # Drop the lowest attentions, but don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                # flat, size: (N, seq_len*seq_len)
                _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
                indices = indices[indices != 0]
                # flat[:, indices] = 0
                # flat, size: (N, seq_len, seq_len)
                '''

                # Stack the identity matrices along a new dimension
                I = torch.stack([torch.eye(seq_len) for _ in range(N)])
                # I, size: (N, seq_len, seq_len)
                # I = torch.eye(attention_heads_fused.size(-1))

                a = (attention_heads_fused + 1.0*I)/2
                # a = attention_heads_fused
                # a, size: (N, seq_len, seq_len)
                a = a / a.sum(dim=-1, keepdim=True)
                # a, size: (N, seq_len, seq_len)

                result = torch.matmul(a, result)
                # result, size: (N, seq_len, seq_len)
        # Look at the total attention between the class token, and the image patches
        mask = result[:, 0, 1:]
        # mask, size: (N, seq_len-1)
        return mask
    """

    @staticmethod
    def rollout(attentions, discard_ratio, head_fusion):
        # attention, size: (N, n_heads, seq_len, seq_len)
        N, n_heads, seq_len, _ = attentions[0].size()
        result = torch.eye(seq_len)
        # result, size: (N, seq_len, seq_len)

        with torch.no_grad():
            for attention in attentions:
                # attention, size: (N, n_heads, seq_len, seq_len)
                if head_fusion == "mean":
                    attention_heads_fused = attention.mean(axis=1)
                elif head_fusion == "max":
                    attention_heads_fused = attention.max(axis=1)[0]
                elif head_fusion == "min":
                    attention_heads_fused = attention.min(axis=1)[0]
                else:
                    raise "Attention head fusion type Not supported"
                # attention_fused, size: (N, seq_len, seq_len)

                # Stack the identity matrices along a new dimension
                I = torch.stack([torch.eye(seq_len) for _ in range(N)])
                # I, size: (N, seq_len, seq_len)
                # I = torch.eye(attention_heads_fused.size(-1))

                a = (attention_heads_fused + 1.0*I)/2
                # a = attention_heads_fused
                # a, size: (N, seq_len, seq_len)
                a = a / a.sum(dim=-1, keepdim=True)
                # a, size: (N, seq_len, seq_len)

                result = torch.matmul(a, result)
                # result, size: (N, seq_len, seq_len)
        # Look at the total attention between the class token, and the image patches
        mask = result[:, 0, 1:]
        # mask, size: (N, seq_len-1)
        return mask

    def get_attention(self, module, input, output):
        attn_output, attn_weights = output
        self.attentions.append(attn_weights.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
        attn_dict = {i: attn for i, attn in enumerate(self.attentions)}
        # {0: (N, l, l), 1: (N, l, l), 2: (N, l, l)}
        return attn_dict
        # return self.rollout(self.attentions, self.discard_ratio, self.head_fusion)
