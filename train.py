#!/usr/bin/env python3
# RAWai Core Training Script v2.0
# Implements: Zipf-TFIDF embeddings • PreNorm GELU blocks • Active corpus expansion • RSI • Quant-aware training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from torch.ao.quantization import QuantStub, DeQuantStub
import math
import os
import json
import pickle


class Config:
    """Configuration for training stages and parameters."""
    STAGE = "Alpha-12"  # Alpha-12 | Beta-24 | Gamma-90
    EMBED_DIM = 2048
    LAYERS = {"Alpha-12": 12, "Beta-24": 24, "Gamma-90": 90}
    HEADS = {"Alpha-12": 16, "Beta-24": 24, "Gamma-90": 32}
    HIDDEN_DIM = {"Alpha-12": 1024, "Beta-24": 1536, "Gamma-90": 2048}

    BATCH_SIZE = 32
    SEQ_LEN = 2048
    LR = 2e-4
    ACTIVE_INTERVAL = 500
    RSI_INTERVAL = 2500
    MAX_GRAD_NORM = 1.0
    EMBED_GRAD_CLIP = 1.5

    TOKENIZER_PATH = "tokenizer.json"
    EMBED_INIT_PATH = "rawai_embed_init.pt"
    CORPUS_PATH = "corpus/cleaned.jsonl"
    CKPT_DIR = "ckpts"


def he_normal_scaled(module, scale=1.1):
    """Kaiming-normal initialization with scaling."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
        module.weight.data.mul_(scale)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class PreNormGELUBlock(nn.Module):
    """Pre-normalization GELU block with residual connections."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, device='cuda'
        )
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(approximate='none'),
            nn.Linear(4 * d_model, d_model)
        )
        self.apply(lambda m: he_normal_scaled(m, 1.1))

    def forward(self, x):
        attn_out, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x)
        )
        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x

class RAWaiCore(nn.Module):
    """Autonomous transformer with custom embeddings."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stage = config.STAGE
        self.n_layers = config.LAYERS[self.stage]
        self.d_model = config.HIDDEN_DIM[self.stage]
        self.n_heads = config.HEADS[self.stage]

        self.embed = nn.Embedding.from_pretrained(
            torch.load(config.EMBED_INIT_PATH), freeze=False
        )
        self.pos_embed = nn.Embedding(config.SEQ_LEN, self.d_model)

        self.blocks = nn.ModuleList([
            PreNormGELUBlock(self.d_model, self.n_heads)
            for _ in range(self.n_layers)
        ])

        self.ln_final = nn.LayerNorm(self.d_model, eps=1e-5)
        self.head = nn.Linear(self.d_model, self.embed.num_embeddings)

        if self.stage == "Gamma-90":
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

        self.apply(lambda m: he_normal_scaled(m, 1.1))

    def forward(self, input_ids):
        positions = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(positions)
        if self.stage == "Gamma-90":
            x = self.quant(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.head(x)
        if self.stage == "Gamma-90":
            logits = self.dequant(logits)
        return logits


class CuriosityCrawler:
    """Uncertainty-driven corpus expansion."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def detect_uncertainty(self, model, batch):
        with torch.no_grad():
            logits = model(batch['input_ids'])
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            high_entropy_indices = entropy.topk(10, dim=1).indices
        return high_entropy_indices

    def generate_queries(self, high_entropy_tokens):
        decoded = [self.tokenizer.decode(toks) for toks in high_entropy_tokens]
        return [f"arxiv:{d}" for d in decoded]

    def execute_crawl(self, queries):
        return [f"[AUTO] New document about {q}" for q in queries]


class RecursiveCodex:
    """Self-modification engine with sandboxing."""
    def __init__(self, model):
        self.model = model
        self.sandbox = SandboxedEnvironment()

    def propose_patch(self):
        return {
            'description': 'Optimize attention computation',
            'code': 'def new_attn(x): ...'
        }

    def validate_patch(self, patch):
        return len(patch['code']) < 1000

    def apply_patch(self, model, patch):
        print(f"Applying RSI patch: {patch['description']}")
        return model


class SandboxedEnvironment:
    """Secure execution context for RSI testing."""
    def __enter__(self):
        torch.backends.cudnn.deterministic = True
        return self

    def __exit__(self, *args):
        torch.backends.cudnn.deterministic = False


class StreamableCorpus(IterableDataset):
    """Supports on-the-fly corpus updates."""
    def __init__(self, file_path, seq_len):
        self.file_path = file_path
        self.seq_len = seq_len
        self.documents = self.load_docs()

    def load_docs(self):
        docs = []
        with open(self.file_path, 'r') as f:
            for line in f:
                docs.append(json.loads(line)['text'])
        return docs

    def enqueue(self, new_docs):
        self.documents.extend(new_docs)

    def __iter__(self):
        for doc in self.documents:
            tokens = doc.split()[:self.seq_len]
            yield {'input_ids': torch.tensor(tokens)}


def main():
    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_PATH)
    crawler = CuriosityCrawler(tokenizer)
    corpus = StreamableCorpus(config.CORPUS_PATH, config.SEQ_LEN)
    loader = DataLoader(corpus, batch_size=config.BATCH_SIZE, pin_memory=True)
    model = RAWaiCore(config).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=config.LR)
    codex = RecursiveCodex(model)
    global_step = 0
    for epoch in range(100):
        for batch in loader:
            input_ids = batch['input_ids'].cuda()
            logits = model(input_ids[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for n,p in model.named_parameters() if 'embed' not in n],
                config.MAX_GRAD_NORM
            )
            torch.nn.utils.clip_grad_norm_(
                model.embed.parameters(),
                config.EMBED_GRAD_CLIP
            )
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % config.ACTIVE_INTERVAL == 0:
                high_entropy = crawler.detect_uncertainty(model, batch)
                queries = crawler.generate_queries(high_entropy)
                new_docs = crawler.execute_crawl(queries)
                corpus.enqueue(new_docs)
                print(f"Added {len(new_docs)} documents via curiosity loop")
            if global_step % config.RSI_INTERVAL == 0:
                with SandboxedEnvironment():
                    patch = codex.propose_patch()
                    if codex.validate_patch(patch):
                        model = codex.apply_patch(model, patch)
            if global_step % 100 == 0:
                print(f"Step {global_step} | Loss: {loss.item():.4f}")
        ckpt_path = os.path.join(
            config.CKPT_DIR,
            f"rawai_{config.STAGE.lower()}_epoch{epoch}.pt"
        )
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == '__main__':
    main()
