"""A tiny causal LM + byte-level BPE tokenizer built offline, for exercising HFProposer / training mechanics."""

from __future__ import annotations


def build_tokenizer(texts, vocab_size: int = 600):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<eos>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>")
    fast.pad_token = fast.eos_token
    return fast


def build_model(vocab_size: int, hidden: int = 96, layers: int = 2):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=vocab_size, hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=hidden // 4,
                      max_position_embeddings=16384, tie_word_embeddings=True)
    return Qwen3ForCausalLM(cfg)


def train(model, tok, pairs, steps: int = 300, lr: float = 3e-3, log_every: int = 50):
    """Plain loop: caption + "\\n" + path + eos, loss on the path only (BrickNet's SFT convention)."""
    import torch

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    nl = tok("\n", add_special_tokens=False).input_ids
    batches = []
    for cap, path in pairs:
        c = tok(cap, add_special_tokens=False).input_ids + nl
        p = tok(path, add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = torch.tensor([c + p])
        labels = ids.clone()
        labels[0, : len(c)] = -100
        batches.append((ids, labels))
    model.train()
    for step in range(steps):
        ids, labels = batches[step % len(batches)]
        loss = model(input_ids=ids, labels=labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and step % log_every == 0:
            print(f"step {step} loss {loss.item():.3f}", flush=True)
    model.eval()
    return model
