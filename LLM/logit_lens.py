"""
Logit Lens Analysis & Visualization.

This module demonstrates how a Transformer language model refines its next-token
predictions layer by layer by projecting intermediate residual stream states
directly into vocabulary space ("Logit Lens").
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

try:
    from transformer_lens import HookedTransformer
    from transformer_lens.utils import get_act_name
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def get_act_name(name: str, layer: int) -> str:
        """Return a standardized activation hook name for a given layer."""
        return f"blocks.{layer}.hook_{name}"

    class _ModelConfig:
        """Minimal config object compatible with HookedTransformer.cfg."""

        def __init__(self, hf_config):
            self.n_layers = hf_config.num_hidden_layers
            self.d_model = hf_config.hidden_size
            self.d_vocab = hf_config.vocab_size
            self.use_bias = True

    class HookedTransformer:
        """
        Lightweight compatibility wrapper around HuggingFace GPT-2 CausalLM
        that exposes the HookedTransformer interface required for Logit Lens.
        """

        def __init__(self, hf_model, tokenizer):
            self.hf_model = hf_model
            self.tokenizer = tokenizer
            self.cfg = _ModelConfig(hf_model.config)
            self.device = next(hf_model.parameters()).device

        @classmethod
        def from_pretrained(cls, model_name: str = "gpt2", **kwargs) -> "HookedTransformer":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            hf_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            hf_model.eval()
            return cls(hf_model, tokenizer)

        @property
        def ln_final(self) -> torch.nn.Module:
            return self.hf_model.transformer.ln_f

        @property
        def W_U(self) -> torch.Tensor:
            # HuggingFace GPT2 lm_head.weight has shape [vocab_size, d_model];
            # TransformerLens W_U has shape [d_model, vocab_size].
            return self.hf_model.lm_head.weight.T

        @property
        def b_U(self) -> torch.Tensor:
            if self.hf_model.lm_head.bias is not None:
                return self.hf_model.lm_head.bias
            return torch.zeros(self.cfg.d_vocab, device=self.device)

        def to_tokens(self, prompt: str) -> torch.Tensor:
            return self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        def to_str_tokens(self, prompt: str) -> List[str]:
            token_ids = self.to_tokens(prompt)[0].tolist()
            return [self.tokenizer.decode([tid]) for tid in token_ids]

        def generate(self, prompt: str, max_new_tokens: int = 15) -> str:
            input_ids = self.to_tokens(prompt)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                output_ids = self.hf_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        def run_with_cache(
            self, tokens: torch.Tensor, remove_batch_dim: bool = True
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            cache: Dict[str, torch.Tensor] = {}
            hooks = []

            def _make_hook(layer_idx: int):
                def _hook(_module, _inp, output):
                    # GPT2Block returns a tuple where output[0] is the hidden state (resid_post)
                    hidden = output[0] if isinstance(output, tuple) else output
                    detached = hidden.detach()
                    if remove_batch_dim:
                        detached = detached.squeeze(0)
                    cache[get_act_name("resid_post", layer_idx)] = detached

                return _hook

            for layer in range(self.cfg.n_layers):
                h = self.hf_model.transformer.h[layer].register_forward_hook(_make_hook(layer))
                hooks.append(h)

            try:
                with torch.no_grad():
                    outputs = self.hf_model(tokens)
            finally:
                for h in hooks:
                    h.remove()

            logits = outputs.logits.squeeze(0) if remove_batch_dim else outputs.logits
            return logits, cache


@dataclass
class LogitLensResult:
    """Dataclass to store the results of the logit lens analysis."""

    prompt: str
    tokens: List[str]
    logits_per_layer: torch.Tensor      # Shape: [n_layers, seq_len, vocab_size]
    probs_per_layer: torch.Tensor       # Shape: [n_layers, seq_len, vocab_size]
    top_tokens: List[List[str]]         # Shape: [n_layers, top_k] at target position
    top_probs: List[List[float]]        # Shape: [n_layers, top_k] at target position


def load_model(model_name: str = "gpt2") -> HookedTransformer:
    """
    Load a pre-trained HookedTransformer model (defaults to GPT-2 Small).

    Args:
        model_name (str): HuggingFace / TransformerLens model identifier.
    Returns:
        HookedTransformer: Loaded model in evaluation mode.
    """
    print(f"Loading model '{model_name}'...")
    model = HookedTransformer.from_pretrained(model_name)
    return model


def generate_tokens(model: HookedTransformer, prompt: str, max_new_tokens: int = 15) -> List[str]:
    """
    Generate tokens from the prompt using the model.

    Args:
        model (HookedTransformer): Loaded transformer model.
        prompt (str): The input prompt to generate tokens from.
        max_new_tokens (int): Maximum number of new tokens to generate.
    Returns:
        List[str]: A list of string tokens from the generated text.
    """
    generated_text = model.generate(prompt, max_new_tokens=max_new_tokens)
    return model.to_str_tokens(generated_text)


def run_model(model: HookedTransformer) -> List[str]:
    """
    Run the model on a few-shot prompt and generate answer tokens.

    Args:
        model (HookedTransformer): Loaded transformer model.
    Returns:
        List[str]: A list of generated tokens.
    """
    few_shot_prompt = (
        "The following is a conversation between a human and an AI assistant. "
        "The assistant is helpful, creative, clever, and very friendly.\n\n"
        "Human: Hello, who are you?\n"
        "AI: I am an AI assistant. How can I help you today?\n"
        "Human: What is the capital of Germany?\n"
        "AI: The capital of Germany is Berlin.\n"
        "Human: What is the capital of Japan?\n"
        "AI: The capital of Japan is Tokyo."
    )
    my_query = "What is the capital of France?"
    full_prompt = f"{few_shot_prompt}\nHuman: {my_query}\nAI:"
    return generate_tokens(model, full_prompt, max_new_tokens=8)


def get_logits(
    model: HookedTransformer,
    prompt: str,
    position: int = -1,
    top_k: int = 5,
    apply_ln: bool = True,
) -> LogitLensResult:
    """
    Extract intermediate residual stream activations across all layers and project
    them into vocabulary logits and probabilities (Logit Lens).

    Pipeline:
        Hidden State (resid_post) -> Final LayerNorm -> Unembedding (W_U) -> Softmax -> Probabilities

    Args:
        model (HookedTransformer): The transformer model to analyze.
        prompt (str): Input text prompt to analyze.
        position (int): Token position index to inspect for top-k summary (default: -1, last token).
        top_k (int): Number of top predicted tokens to record per layer.
        apply_ln (bool): Whether to apply the final layer normalization before unembedding.
    Returns:
        LogitLensResult: Dataclass containing per-layer logits, probabilities, and top-k predictions.
    """
    token_ids = model.to_tokens(prompt)
    str_tokens = model.to_str_tokens(prompt)
    n_layers = model.cfg.n_layers

    with torch.no_grad():
        _, cache = model.run_with_cache(token_ids, remove_batch_dim=True)

        # Stack intermediate residual streams: [n_layers, seq_len, d_model]
        residual_streams = torch.stack(
            [cache[get_act_name("resid_post", layer)] for layer in range(n_layers)],
            dim=0,
        )

        # Apply final LayerNorm across the d_model dimension before unembedding
        if apply_ln:
            residual_streams = model.ln_final(residual_streams)

        # Project hidden states to vocabulary space:
        # residual_streams: [n_layers, seq_len, d_model], W_U: [d_model, vocab_size]
        logits = torch.einsum("lsd,dv->lsv", residual_streams, model.W_U)

        if getattr(model.cfg, "use_bias", False) and model.b_U is not None:
            logits = logits + model.b_U[None, None, :]

        # Convert logits to probabilities across the vocabulary dimension
        probs = F.softmax(logits, dim=-1)

        # Extract top-k predictions for the specified token position across all layers
        pos_probs = probs[:, position, :]
        top_probs_vals, top_indices = torch.topk(pos_probs, k=top_k, dim=-1)
        top_tokens = [
            [model.tokenizer.decode([idx.item()]) for idx in layer_indices]
            for layer_indices in top_indices
        ]

    return LogitLensResult(
        prompt=prompt,
        tokens=str_tokens,
        logits_per_layer=logits,
        probs_per_layer=probs,
        top_tokens=top_tokens,
        top_probs=top_probs_vals.tolist(),
    )


def display_results(results: LogitLensResult) -> pd.DataFrame:
    """
    Display a formatted table of the top-k predicted next tokens for each layer.

    Args:
        results (LogitLensResult): The results of the logit lens analysis.
    Returns:
        pd.DataFrame: Summary table of layer-by-layer top predictions.
    """
    n_layers = len(results.top_tokens)
    top_k = len(results.top_tokens[0]) if n_layers > 0 else 0

    rows = []
    for layer_idx in range(n_layers):
        row = {"Layer": f"Layer {layer_idx:2d}"}
        for k in range(top_k):
            tok = repr(results.top_tokens[layer_idx][k])
            prob = results.top_probs[layer_idx][k]
            row[f"Top-{k + 1}"] = f"{tok} ({prob * 100:5.1f}%)"
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print(f"Logit Lens Next-Token Predictions for Prompt: {results.prompt!r}")
    print(f"Tokenized Prompt: {results.tokens}")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")
    return df


def plot_logit_lens_heatmap(
    model: HookedTransformer,
    results: LogitLensResult,
    save_path: Optional[str] = None,
    show: bool = True,
) -> str:
    """
    Plot a 2D Logit Lens heatmap across all layers (Y-axis) and input token
    positions (X-axis). Each cell displays the top-1 predicted next token and
    is colored by its probability.

    Args:
        model (HookedTransformer): The transformer model.
        results (LogitLensResult): Computed LogitLensResult containing probs_per_layer.
        save_path (Optional[str]): File path to save the output PNG image.
        show (bool): Whether to display the plot interactively.
    Returns:
        str: Path where the visualization image was saved.
    """
    probs = results.probs_per_layer  # [n_layers, seq_len, vocab_size]
    n_layers, n_tokens, _ = probs.shape

    top1_probs, top1_indices = probs.max(dim=-1)
    top1_probs_np = top1_probs.detach().cpu().numpy()

    top1_tokens = [
        [model.tokenizer.decode([idx.item()]).replace("\n", "\\n") for idx in layer_indices]
        for layer_indices in top1_indices
    ]

    fig_width = max(10, n_tokens * 1.4)
    fig_height = max(6, n_layers * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(top1_probs_np, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Top-1 Next-Token Probability", fontsize=11)

    # Annotate each (layer, position) cell with the decoded top-1 token and probability
    for layer_idx in range(n_layers):
        for pos_idx in range(n_tokens):
            prob_val = top1_probs_np[layer_idx, pos_idx]
            tok_label = top1_tokens[layer_idx][pos_idx]
            text_color = "black" if prob_val > 0.55 else "white"
            ax.text(
                pos_idx,
                layer_idx,
                f"'{tok_label}'\n({prob_val:.2f})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
                fontweight="bold",
            )

    x_labels = [f"{i}: {repr(t)}" for i, t in enumerate(results.tokens)]
    y_labels = [f"Layer {l}" for l in range(n_layers)]

    ax.set_xticks(np.arange(n_tokens))
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.invert_yaxis()  # Put Layer 0 at the bottom and final layer at the top

    ax.set_xlabel("Input Prompt Token Position", fontsize=12, labelpad=8)
    ax.set_ylabel("Transformer Layer (Residual Stream)", fontsize=12, labelpad=8)
    ax.set_title(
        f"Logit Lens Heatmap: Top-1 Predicted Next Token Across Layers\nPrompt: {results.prompt!r}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    fig.tight_layout()

    if save_path is None:
        out_dir = Path(__file__).resolve().parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(out_dir / "logit_lens_heatmap.png")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved Logit Lens heatmap image to: {save_path}")

    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    return save_path


def plot_token_probability_line(
    model: HookedTransformer,
    prompt: str,
    target_tokens: List[str],
    apply_ln: bool = True,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
    show: bool = True,
) -> str:
    """
    Plot the probability trajectory of specific target tokens across layers at the
    final token position of the prompt.

    Args:
        model (HookedTransformer): The transformer model to analyze.
        prompt (str): The input prompt to feed into the model.
        target_tokens (List[str]): A list of candidate next-token strings to track.
        apply_ln (bool): Whether to apply final LayerNorm before unembedding.
        figsize (Tuple[int, int]): Size of the figure for plotting.
        save_path (Optional[str]): File path to save the output PNG image.
        show (bool): Whether to display the plot interactively.
    Returns:
        str: Path where the visualization image was saved.
    """
    results = get_logits(model, prompt=prompt, position=-1, apply_ln=apply_ln)
    probs_last_pos = results.probs_per_layer[:, -1, :]  # [n_layers, vocab_size]
    n_layers = probs_last_pos.shape[0]
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=figsize)
    markers = ["o", "s", "^", "D", "v", "P"]

    max_prob = 0.0
    for idx, token_str in enumerate(target_tokens):
        encoded = model.tokenizer.encode(token_str, add_special_tokens=False)
        if not encoded:
            continue
        token_id = encoded[0]
        decoded_label = model.tokenizer.decode([token_id])
        token_probs = probs_last_pos[:, token_id].detach().cpu().numpy()
        max_prob = max(max_prob, float(token_probs.max()))

        marker = markers[idx % len(markers)]
        ax.plot(
            layers,
            token_probs,
            marker=marker,
            linewidth=2.2,
            markersize=6,
            label=f"Token {decoded_label!r} (id={token_id})",
        )

    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers])
    upper_ylim = min(1.05, max(0.25, max_prob * 1.2))
    ax.set_ylim(-0.01, upper_ylim)
    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("Next-Token Probability", fontsize=12)
    ax.set_title(
        f"Logit Lens Probability Trajectory Across Layers\nPrompt: {prompt!r}",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()

    if save_path is None:
        out_dir = Path(__file__).resolve().parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(out_dir / "logit_lens_prob_trajectory.png")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved token probability trajectory image to: {save_path}")

    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    return save_path


def main():
    """
    Main function to run the Logit Lens examples and generate visual outputs.
    """
    # Use non-interactive backend when running as a script so figures save cleanly
    matplotlib.use("Agg")

    # 1. Load pre-trained GPT-2 model
    model = load_model("gpt2")

    # 2. Example 1: Few-shot generation demo
    generated_tokens = run_model(model)
    print("Generated continuation tokens (last 12):", generated_tokens[-12:])

    # 3. Example 2: Inspect layer-by-layer Logit Lens predictions on a factual prompt
    prompt = "The Eiffel Tower is located in the city of"
    results = get_logits(model, prompt=prompt, position=-1, top_k=5, apply_ln=True)

    # Display tabular layer-by-layer top-5 predictions
    display_results(results)

    # 4. Generate Visual Output 1: 2D Logit Lens Heatmap (Layers vs. Token Positions)
    plot_logit_lens_heatmap(model, results, show=False)

    # 5. Generate Visual Output 2: Target Token Probability Trajectory Across Layers
    candidate_tokens = [" Paris", " London", " Rome", " France", " Berlin"]
    plot_token_probability_line(
        model=model,
        prompt=prompt,
        target_tokens=candidate_tokens,
        apply_ln=True,
        show=False,
    )


if __name__ == "__main__":
    main()

