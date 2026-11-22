"""Run and independently decode the D2-03 GPT-2 top-40 trajectories."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from imec import BLOCK_BITS, MESSAGE_BITS, STOP_ENTROPY, TOP_K, GreedyCoupling, Trajectory, uniform_belief


class GPT2Channel:
    def __init__(self, directory, device):
        self.device = torch.device(device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(directory, local_files_only=True)
        self.model = GPT2LMHeadModel.from_pretrained(
            directory, local_files_only=True, use_safetensors=True, attn_implementation="eager",
        ).to(self.device).eval()
        self.bos = self.tokenizer.eos_token_id
        self.max_positions = self.model.config.n_positions

    def start(self, contexts):
        pieces = self.tokenizer(contexts, add_special_tokens=False)["input_ids"]
        prompt_tokens = [[self.bos, *row][-1022:] for row in pieces]
        width = max(map(len, prompt_tokens))
        ids = [[self.bos] * (width - len(row)) + row for row in prompt_tokens]
        mask = [[0] * (width - len(row)) + [1] * len(row) for row in prompt_tokens]
        return GPT2Stream(self, ids, mask, list(map(len, prompt_tokens)))


class GPT2Stream:
    def __init__(self, channel, ids, mask, lengths):
        self.channel = channel
        self.ids = torch.tensor(ids, device=channel.device)
        self.mask = torch.tensor(mask, device=channel.device)
        self.lengths = lengths
        self.positions = (self.mask.cumsum(dim=1) - 1).clamp_min(0)
        self.cache = None

    @torch.inference_mode()
    def distribution(self):
        model = self.channel.model
        hidden = model.transformer(
            input_ids=self.ids, attention_mask=self.mask, position_ids=self.positions,
            past_key_values=self.cache, use_cache=True, return_dict=True,
        )
        self.cache = hidden.past_key_values
        logits = model.lm_head(hidden.last_hidden_state[:, -1, :])
        logits[:, [628, self.channel.bos]] = -torch.inf
        # Break exact logit ties by token ID.
        labels = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :TOP_K]
        highest = logits.gather(1, labels).double()
        probabilities = torch.softmax(highest, dim=1)
        return labels.cpu().numpy(), probabilities.cpu().numpy()

    def push(self, ids):
        self.ids = torch.tensor(ids, device=self.channel.device).unsqueeze(1)
        self.positions = self.mask.sum(dim=1, keepdim=True)
        self.mask = torch.cat((self.mask, torch.ones_like(self.ids)), dim=1)


def update_case(coupling, trial, labels, probabilities, token):
    try:
        trial.advance(coupling, labels, probabilities, token)
    except (ArithmeticError, ValueError) as exc:
        trial.status = "error"
        trial.failure = str(exc)


def run_phase(channel, batch, trials, coupling, pool, *, max_tokens, timeout, received=None):
    start = time.perf_counter()
    stream = channel.start([item["context"] for item in batch])
    steps = min(max_tokens, channel.max_positions - stream.mask.shape[1] + 1)
    for step in range(steps):
        active = [i for i, trial in enumerate(trials) if trial.status == "running"]
        if not active:
            break
        if time.perf_counter() - start >= timeout:
            for i in active:
                trials[i].status = "timeout"
                trials[i].failure = "phase wall-time limit"
            break

        labels, probabilities = stream.distribution()
        updates = []
        for i in active:
            token = None
            if received is not None:
                if step >= len(received[i]):
                    trials[i].status = "incomplete"
                    trials[i].failure = "received stream ended before entropy threshold"
                    continue
                token = received[i][step]
            updates.append(pool.submit(update_case, coupling, trials[i], labels[i], probabilities[i], token))
        for update in updates:
            update.result()

        if received is None:
            next_ids = [
                trial.tokens[step] if len(trial.tokens) > step else channel.bos for trial in trials
            ]
        else:
            next_ids = [
                tokens[step] if step < len(tokens) else channel.bos for tokens in received
            ]
        stream.push(next_ids)

    for trial in trials:
        if trial.status == "running":
            trial.status = "incomplete"
            trial.failure = "token limit or model context limit"
    return time.perf_counter() - start, stream.lengths


def run_batch(channel, batch, coupling, initial, pool, args):
    encoders = []
    for case in batch:
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, case["id"]]))
        message = "".join(map(str, rng.integers(0, 2, MESSAGE_BITS)))
        encoders.append(Trajectory(initial, block_bits=args.block_bits, message=message, random=rng))

    limits = {"max_tokens": args.max_tokens, "timeout": args.timeout}
    encode_time, lengths = run_phase(channel, batch, encoders, coupling, pool, **limits)
    decoders = [Trajectory(initial, block_bits=args.block_bits) for _ in batch]
    for encoder, decoder in zip(encoders, decoders):
        if encoder.status != "complete":
            decoder.status = "skipped"
    decode_time, _ = run_phase(
        channel, batch, decoders, coupling, pool,
        received=[encoder.tokens for encoder in encoders], **limits,
    )

    rows = []
    for case, encoder, decoder, prompt_length in zip(batch, encoders, decoders, lengths):
        status = encoder.status
        error = encoder.failure
        decoded = None
        wrong = None
        if encoder.status == "complete":
            status = decoder.status
            error = decoder.failure
            if decoder.status == "complete":
                decoded = decoder.recovered_message()
                wrong = sum(a != b for a, b in zip(encoder.message, decoded))
                if len(decoder.tokens) != len(encoder.tokens):
                    status, error = "error", "decoder ended at a different token"
                elif wrong:
                    status, error = "error", "bit mismatch"
                else:
                    status = "ok"

        entropy_sum = sum(encoder.entropies)
        n_tokens = len(encoder.tokens)
        rows.append({
            "id": case["id"],
            "article_index": case["article_index"],
            "prompt_tokens": prompt_length,
            "status": status,
            "encoder_status": encoder.status,
            "decoder_status": decoder.status,
            "failure": error,
            "message_bits": encoder.message,
            "recovered_bits": decoded,
            "wrong_bits": wrong,
            "tokens": encoder.tokens,
            "n_tokens": n_tokens,
            "max_encoder_entropy": encoder.max_entropy,
            "max_decoder_entropy": decoder.max_entropy if decoder.status != "skipped" else None,
            "channel_entropy_sum": entropy_sum,
            "channel_entropy_mean": entropy_sum / n_tokens if n_tokens else None,
            "bits_per_token": MESSAGE_BITS / n_tokens if encoder.status == "complete" else None,
            "efficiency": MESSAGE_BITS / entropy_sum if encoder.status == "complete" else None,
        })
    return rows, encode_time, decode_time


def summarize(rows):
    completed = [row for row in rows if row["encoder_status"] == "complete"]
    decoded = [row for row in completed if row["decoder_status"] == "complete"]
    verified = [row for row in rows if row["status"] == "ok"]
    statuses = Counter(row["status"] for row in rows)
    efficiencies = np.array([row["efficiency"] for row in completed])
    verified_eff = [row["efficiency"] for row in verified]
    total_tokens = sum(row["n_tokens"] for row in completed)
    entropy = sum(row["channel_entropy_sum"] for row in completed)
    mean = float(efficiencies.mean()) if completed else None
    margin = 1.96 * float(efficiencies.std(ddof=1)) / math.sqrt(len(completed)) if len(completed) > 1 else None
    return {
        "attempted": len(rows),
        "encoder_completed": len(completed),
        "decoder_completed": len(decoded),
        "verified": len(verified),
        "timeout": statuses["timeout"],
        "errors": statuses["error"],
        "incomplete_other": statuses["incomplete"],
        "not_completed": len(rows) - len(completed),
        "wrong_bits": sum(row["wrong_bits"] for row in decoded),
        "bits_checked": len(decoded) * MESSAGE_BITS,
        "completed_tokens": total_tokens,
        "completed_channel_entropy_sum": entropy,
        "mean_bits_per_token": float(np.mean([row["bits_per_token"] for row in completed])) if completed else None,
        "mean_channel_entropy": float(np.mean([row["channel_entropy_mean"] for row in completed])) if completed else None,
        "mean_efficiency": mean,
        "efficiency_ci95": [mean - margin, mean + margin] if margin is not None else None,
        "mean_efficiency_verified": float(np.mean(verified_eff)) if verified_eff else None,
        "pooled_efficiency": MESSAGE_BITS * len(completed) / entropy if completed else None,
        "max_final_entropy": max((row["max_encoder_entropy"] for row in completed), default=None),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=Path("coupling.so"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--block-bits", type=int, choices=(10, BLOCK_BITS), default=BLOCK_BITS)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if min(args.batch_size, args.workers, args.max_tokens, args.timeout) <= 0 or args.offset < 0:
        parser.error("Batch size, workers, max-tokens and timeout must be positive, and offset nonnegative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main():
    args = parse_args()
    prompts = [
        json.loads(line) for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if args.limit is not None and args.offset + args.limit > len(prompts):
        raise ValueError("Requested more trajectories than the prompt file contains")
    end = args.offset + args.limit if args.limit is not None else len(prompts)
    prompts = prompts[args.offset:end]
    if not prompts:
        raise ValueError("No prompts selected")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    coupling = GreedyCoupling(args.library)
    initial = uniform_belief(args.block_bits)
    channel = GPT2Channel(args.model, args.device)
    rows = []
    encode_time = decode_time = 0.0
    start = time.perf_counter()
    with (args.out / "trajectories.jsonl").open("x", encoding="utf-8", newline="\n") as log, (
        ThreadPoolExecutor(max_workers=args.workers)
    ) as pool:
        for offset in range(0, len(prompts), args.batch_size):
            batch = prompts[offset : offset + args.batch_size]
            batch_rows, enc, dec = run_batch(channel, batch, coupling, initial, pool, args)
            encode_time += enc
            decode_time += dec
            for row in batch_rows:
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            rows.extend(batch_rows)
            print(
                f"{len(rows)}/{len(prompts)} "
                f"{dict(Counter(row['status'] for row in batch_rows))} "
                f"elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )

    summary = summarize(rows)
    summary["protocol"] = {
        "top_k": TOP_K, "temperature": 1.0, "message_bits": MESSAGE_BITS,
        "block_bits": args.block_bits, "stop_entropy_bits": STOP_ENTROPY, "seed": args.seed,
        "offset": args.offset, "batch_size": args.batch_size, "workers": args.workers,
        "max_tokens": args.max_tokens, "timeout_seconds_per_batch_phase": args.timeout,
    }
    environment = {
        "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__, "device": str(channel.device),
    }
    if channel.device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(channel.device)
        environment["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated(channel.device) / (1024 ** 2)
    if platform.system() == "Linux":
        import resource

        environment["ram_peak_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    summary["environment"] = environment
    summary["encode_wall_seconds"] = encode_time
    summary["decode_wall_seconds"] = decode_time
    summary["total_wall_seconds"] = time.perf_counter() - start
    result = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    (args.out / "summary.json").write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
