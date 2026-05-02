"""
SVG Sample Generation Script
Generates unconditional and prefix-conditioned SVG samples from trained models.
"""

import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# SVG prefixes for conditioned generation
SVG_PREFIXES = {
    "empty_svg": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">',

    "partial_face": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
<circle cx="12" cy="12" r="10" fill="none" stroke="black" stroke-width="1"/>
<circle cx="9" cy="10" r="1" fill="black"/>''',

    "open_path": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
<path d="M 4 20 Q 12 4 20 20" fill="none" stroke="black" stroke-width="1.5"/>''',

    "group_one_shape": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
<g fill="none" stroke="black" stroke-width="1.5">
<rect x="3" y="3" width="18" height="18" rx="2"/>''',

    "star_start": '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
<path d="M 12 2 L 14.5 9 L 22 9''',
}


def generate_unconditional(model, tokenizer, num_samples=10, max_tokens=512,
                           temperature=1.0, top_k=None, top_p=None,
                           device='cuda'):
    """
    Generate unconditional SVG samples.

    Args:
        model: Trained GPT model
        tokenizer: BPE tokenizer
        num_samples: Number of samples to generate
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling threshold
        device: Device

    Returns:
        list of generated SVG strings
    """
    model.eval()

    # Starting with <svg prefix
    prefix = '<svg'
    prefix_ids = tokenizer.encode(prefix).ids
    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    eos_id = tokenizer.token_to_id("<eos>")
    generated = []

    for i in range(num_samples):
        # Generate
        output = model.generate(
            prefix_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Decode
        tokens = output[0].tolist()
        # Stopping at EOS if present (only check generated tokens, not prefix)
        prefix_len = len(prefix_ids)
        gen_tokens = tokens[prefix_len:]
        if eos_id in gen_tokens:
            tokens = tokens[:prefix_len + gen_tokens.index(eos_id)]

        text = tokenizer.decode(tokens)

        # Ensuring SVG is closed
        if '</svg>' not in text:
            text += '</svg>'

        generated.append(text)
        print(f"  Sample {i+1}/{num_samples}: {len(tokens)} tokens, {len(text)} chars")

    return generated


def generate_prefix_conditioned(model, tokenizer, prefixes=None, max_tokens=512,
                                 temperature=1.0, top_k=None, top_p=None,
                                 device='cuda'):
    """
    Generate prefix-conditioned SVG completions.

    Args:
        model: Trained GPT model
        tokenizer: BPE tokenizer
        prefixes: Dict of name -> SVG prefix string (None = use defaults)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling threshold
        device: Device

    Returns:
        dict of name -> {'prefix': str, 'completion': str, 'full': str}
    """
    if prefixes is None:
        prefixes = SVG_PREFIXES

    model.eval()
    eos_id = tokenizer.token_to_id("<eos>")
    results = {}

    for name, prefix in prefixes.items():
        print(f"\n  Generating from prefix: {name}")

        prefix_ids = tokenizer.encode(prefix).ids
        prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)

        output = model.generate(
            prefix_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        tokens = output[0].tolist()
        # Stop at EOS if present (only check generated tokens, not prefix)
        prefix_len = len(prefix_ids)
        gen_tokens = tokens[prefix_len:]
        if eos_id in gen_tokens:
            tokens = tokens[:prefix_len + gen_tokens.index(eos_id)]

        full_text = tokenizer.decode(tokens)
        completion = full_text[len(prefix):]  # Approximate

        # Ensure closed
        if '</svg>' not in full_text:
            full_text += '</svg>'

        results[name] = {
            'prefix': prefix,
            'completion': completion,
            'full': full_text,
            'total_tokens': len(tokens),
        }

        print(f"    {len(tokens)} tokens, completion length: {len(completion)} chars")

    return results


def generate_multi_temperature(model, tokenizer, temperatures=[0.5, 0.8, 1.0],
                                num_per_temp=5, max_tokens=512, device='cuda'):
    """
    Generate samples at multiple temperatures for comparison.

    Returns:
        dict of temperature -> list of SVG strings
    """
    results = {}
    for temp in temperatures:
        print(f"\n  Temperature: {temp}")
        samples = generate_unconditional(
            model, tokenizer,
            num_samples=num_per_temp,
            max_tokens=max_tokens,
            temperature=temp,
            device=device,
        )
        results[str(temp)] = samples
    return results


def generate_multi_strategy(model, tokenizer, num_samples=5, max_tokens=512, device='cuda'):
    """
    Generate with different sampling strategies for comparison.

    Returns:
        dict of strategy_name -> list of SVG strings
    """
    strategies = {
        'greedy': {'temperature': 0.01, 'top_k': 1},
        'temp_0.5': {'temperature': 0.5},
        'temp_0.8': {'temperature': 0.8},
        'temp_1.0': {'temperature': 1.0},
        'top_k_50': {'temperature': 1.0, 'top_k': 50},
        'top_p_0.9': {'temperature': 1.0, 'top_p': 0.9},
        'top_p_0.95': {'temperature': 0.8, 'top_p': 0.95},
    }

    results = {}
    for name, params in strategies.items():
        print(f"\n  Strategy: {name} ({params})")
        samples = generate_unconditional(
            model, tokenizer,
            num_samples=num_samples,
            max_tokens=max_tokens,
            device=device,
            **params,
        )
        results[name] = samples

    return results


def run_full_generation(model, tokenizer, output_dir, device='cuda'):
    """
    Run the complete generation pipeline as required by the project.

    Args:
        model: Best trained model
        tokenizer: BPE tokenizer
        output_dir: Directory to save all outputs
        device: Device
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("SVG Generation Pipeline")
    print(f"{'='*60}")

    # 1. Unconditional samples (10+)
    print("\n--- Unconditional Samples ---")
    unconditional = generate_unconditional(
        model, tokenizer, num_samples=12, max_tokens=512,
        temperature=0.8, top_k=50, device=device,
    )

    # 2. Prefix-conditioned samples (5+)
    print("\n--- Prefix-Conditioned Samples ---")
    prefix_results = generate_prefix_conditioned(
        model, tokenizer, max_tokens=512,
        temperature=0.8, top_k=50, device=device,
    )

    # 3. Multi-temperature comparison
    print("\n--- Multi-Temperature Comparison ---")
    temp_results = generate_multi_temperature(
        model, tokenizer, temperatures=[0.5, 0.8, 1.0],
        num_per_temp=5, max_tokens=512, device=device,
    )

    # 4. Multi-strategy comparison
    print("\n--- Multi-Strategy Comparison ---")
    strategy_results = generate_multi_strategy(
        model, tokenizer, num_samples=3, max_tokens=512, device=device,
    )

    # Saving all results
    all_results = {
        'unconditional': unconditional,
        'prefix_conditioned': prefix_results,
        'multi_temperature': temp_results,
        'multi_strategy': strategy_results,
    }

    with open(os.path.join(output_dir, 'generation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nAll generation results saved to {output_dir}")
    return all_results
