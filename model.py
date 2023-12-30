import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import math
from pathlib import Path
import argparse
from transformers import AutoTokenizer
from inference import generate


@dataclass
class ModelArgs:
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    num_key_value_heads: int = 16
    max_position_embeddings: int = 16384
    rms_norm_eps: float = 1e-6
    intermediate_size: int = 5504
    rope_theta: float = 100000
    rope_scaling_factor: int = 4.0
    vocab_size: int = 32256


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5, device="cpu"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims, device=device))
        self.eps = eps

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.pow(x, 2).mean(dim=-1, keepdims=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self.norm(x)


class LlamaLinearScalingRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000, linear_scale=4.0, device: str = "cpu"):
        super().__init__()
        self.dim = dim
        self.base = base
        self.device = device
        self.linear_scale = linear_scale

    @staticmethod
    def compute_frequency_theta(x: torch.Tensor, dim: int, offset: int = 0, base: int = 10000,
                                linear_scale=4.0,
                                device="cpu", dtype=torch.float32):
        # x.shape = B * NH, L, D
        D = dim // 2
        N = x.shape[1] + offset

        dim_pos = torch.arange(0, D, device=device, dtype=dtype)
        pos = torch.arange(offset, N, device=device, dtype=dtype) / linear_scale

        dim_freq = torch.exp(-dim_pos * (math.log(base) / D))
        m_theta = pos.reshape(-1, 1) * dim_freq.reshape(1, -1)
        return torch.cos(m_theta), torch.sin(m_theta)

    def __call__(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        B, NH, L, _ = x.shape
        x = x.reshape(B * NH, L, -1)

        cos_m_theta, sin_m_theta = (LlamaLinearScalingRotaryEmbedding.
                                    compute_frequency_theta(x,
                                                            self.dim,
                                                            offset,
                                                            self.base,
                                                            self.linear_scale,
                                                            self.device))
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2: self.dim]
        rx1 = x1 * cos_m_theta - x2 * sin_m_theta
        rx2 = x1 * sin_m_theta + x2 * cos_m_theta
        if self.dim < x.shape[-1]:
            rx = torch.concatenate([rx1, rx2, x[..., self.dim:]], dim=-1)
        else:
            rx = torch.concatenate([rx1, rx2], dim=-1)
        return rx.reshape(B, NH, L, -1)


class LlamaAttention(nn.Module):
    def __init__(self, args: ModelArgs, device="cpu"):
        super().__init__()
        self.args = args
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim,
                                bias=False, device=device)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim,
                                bias=False, device=device)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim,
                                bias=False, device=device)
        self.o_proj = nn.Linear(args.num_key_value_heads * self.head_dim, args.hidden_size,
                                bias=False, device=device)

        self.repeats = args.num_attention_heads // args.num_key_value_heads
        self.scale = self.head_dim ** -0.5

        self.rotary_emb = LlamaLinearScalingRotaryEmbedding(dim=self.head_dim, base=args.rope_theta,
                                                            linear_scale=args.rope_scaling_factor,
                                                            device=device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor],
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]]) \
            -> Tuple[torch.Tensor, Union[Tuple[torch.Tensor, torch.Tensor], None]]:
        B, L, H = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.args.num_attention_heads, -1).transpose(1, 2)
        keys = keys.reshape(B, L, self.args.num_key_value_heads, -1).transpose(1, 2)
        values = values.reshape(B, L, self.args.num_key_value_heads, -1).transpose(1, 2)

        def repeat(inp: torch.Tensor) -> torch.Tensor:  # B, L, NK, H
            batch, n_kv, seq_length, head_dim = inp.shape
            inp = inp[:, None, :, :, :].expand(batch, self.repeats, n_kv, seq_length, head_dim)
            return inp.reshape(batch, self.repeats * n_kv, seq_length, -1)

        keys, values = map(repeat, (keys, values))  # B, NH, L, D

        if cache is not None:
            key_cache, value_cache = cache
            keys = self.rotary_emb(keys, key_cache.shape[2])
            queries = self.rotary_emb(queries, key_cache.shape[2])
            keys = torch.concatenate((key_cache, keys), dim=2)
            values = torch.concatenate((value_cache, values), dim=2)
        else:
            keys = self.rotary_emb(keys)
            queries = self.rotary_emb(queries)

        scores = (self.scale * queries) @ keys.transpose(2, 3)
        if mask is not None:
            scores += mask
        scores = torch.softmax(scores, dim=-1)

        v_hat = (scores @ values).transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(v_hat), (keys, values)


class LlamaMLP(nn.Module):
    def __init__(self, args: ModelArgs, device='cpu'):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False,
                                   device=device)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False,
                                   device=device)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False,
                                 device=device)
        self.act_fn = nn.SiLU()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, device='cpu'):
        super().__init__()

        self.self_attn = LlamaAttention(args, device)
        self.mlp = LlamaMLP(args, device)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps, device=device)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps,
                                                device=device)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs, device='cpu'):
        super().__init__()
        self.device = device

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size, device=device)
        self.layers = nn.ModuleList([LlamaDecoderLayer(args=args, device=device) for _ in range(
            args.num_hidden_layers)])
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps, device=device)

    @staticmethod
    def create_causal_mask(seq_length: int, device="cpu"):
        pos = torch.arange(0, seq_length, device=device)
        mask = pos[:, None] * pos[None]
        return mask * -1e-9

    def forward(self, x: torch.Tensor, cache=None):
        x = self.embed_tokens(x)

        mask = None
        if x.shape[1] > 1:
            mask = LlamaModel.create_causal_mask(x.shape[1], device=self.device)
            mask = mask.to(x.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            x, cache[e] = layer(x, mask, cache[e])
        x = self.norm(x)
        return x, cache


class DeepSeekCoder(nn.Module):
    def __init__(self, args: ModelArgs, device='cpu') -> None:
        super().__init__()
        self.model = LlamaModel(args, device=device)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False, device=device)

    def forward(self, x: torch.Tensor, cache=None):
        x, cache = self.model(x, cache=cache)
        return self.lm_head(x), cache


def load_weights_and_tokenizer(path: str, device: str = "mps"):
    model_ = DeepSeekCoder(ModelArgs(), device=device)
    path_ = Path(path) / 'deepseek-weights.pt'
    path = Path(path)
    if not path_.exists():
        raise Exception('model weights does not exist in directory ' + str(path))
    print('[INFO] Updating model with weights')
    _model_ = torch.load(str(path / "deepseek-weights.pt"))
    model_.load_state_dict(_model_)
    tokenizer_ = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        trust_remote_code=True)
    return model_, tokenizer_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Deepseek Coder Language Model")
    parser.add_argument("--model-path",
                        type=str,
                        default="weights",
                        help="The path to load model weights from")
    parser.add_argument(
        "--prompt",
        "-p",
        help="Prompt input for model to start generation",
        default="write a react native snippet to display the text 'hello world'",
        type=str,
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=500,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temp",
        "-t",
        help="The sampling temperature.",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--device",
        "-d",
        help="Device to run inference on.",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--seed",
        "-s",
        help="PRNG seed.",
        type=int,
        default=0,
    )

    args_ = parser.parse_args()
    torch.manual_seed(args_.seed)
    model, tokenizer = load_weights_and_tokenizer(args_.model_path, args_.device)

    prompt = tokenizer(
        args_.prompt,
        return_attention_mask=False,
    )["input_ids"]
    prompt = torch.as_tensor(prompt, dtype=torch.int32, device=args_.device)

    print("[INFO] Starting Generation", flush=True)
    print(args_.prompt, end="", flush=True)

    tokens = []
    for token, _ in zip(generate(prompt, model, temp=args_.temp), range(args_.max_tokens)):
        tokens.append(token)

        if (len(tokens) % 10) == 0:
            eos_index = next(
                (i for i, t in enumerate(tokens) if t.item() == tokenizer.eos_token_id),
                None,
            )

            if eos_index is not None:
                tokens = tokens[:eos_index]

            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []
            if eos_index is not None:
                break

    s = tokenizer.decode([t.item() for t in tokens])
    print(s, flush=True)
