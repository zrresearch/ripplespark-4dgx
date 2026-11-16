# RippleSpark: Ripple-65B on two or four DGX Sparks with TensorFold

Ricursive's [Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) (65B dense; 80 layers, four
1,024-token sliding-window layers to every full-attention layer) served across two or four DGX Sparks with tensor
parallelism, from a 5.0 bpw EXL3 pack, on TensorFold 1.2.3, with Ricursive's DFlash2 draft model.

This is [ajensenwaud/ripplespark-2dgx](https://github.com/ajensenwaud/ripplespark-2dgx) with four commits on top:

| Commit | What it adds |
|---|---|
| Four Sparks | `WORKERS` (three workers in rank order); TensorFold patch 0010 (TP=4 as well as 1 or 2); the multirank wire's kernel for the small gathers |
| `RANK_HCAS` | Per-rank RoCE device names, for a Spark that names its ConnectX-7 ports differently from the rest |
| Start fix | A Spark that had never served Ripple had no folder for its GPU lock and read as busy |
| Engine changes | TensorFold patches 0011-0015: prompt reading about 2x on four Sparks and 1.5x on two; shared decode rounds that scale to 16 streams |

Two Sparks work as in the original recipe and get the engine changes too. The write-up, with the measurements,
scripts and logs behind the numbers below, is in
[zrresearch/ripple-65b-sparks](https://github.com/zrresearch/ripple-65b-sparks).

## Running it

Docker with the NVIDIA runtime on every Spark, and SSH from the head (rank 0) to each worker. Ports: 8888 (API),
29681 (NCCL bootstrap) and, on four Sparks, 29700-29707 (the multirank wire). Settings go in `scripts/local.sh`
(gitignored; start from `scripts/local.sh.example`).

**Two Sparks**, joined by their CX7 ports:

```bash
WORKER=user@192.0.2.2       # rank 1, over the CX7 link
MASTER=192.0.2.1            # rank 0's CX7 address
```

**Four Sparks**, one CX7 link each to a QSFP switch with RoCE:

```bash
WORKERS="user@192.0.2.2 user@192.0.2.3 user@192.0.2.4"   # ranks 1-3, in order
MASTER=192.0.2.1                                         # rank 0's fabric address
NCCL_GID=5                  # GID index of RoCE v2 over IPv4 on the switch (3 on a direct cable)
RANK_HCAS="- mlx5_0,mlx5_2 - -"   # only if a Spark names its RoCE devices otherwise; "-" keeps NCCL_HCAS
```

Then, on the head:

```bash
scripts/prepare.sh        # the image, built on the head and copied to every worker
scripts/zig_prepare.sh    # TensorFold v1.2.3 + patches-zig/, compiled in Docker, checked against the pinned tree, copied out
scripts/fetch_pack.sh     # the 5.0 model and the draft model from Hugging Face (pinned), on every Spark, verified
scripts/start.sh          # OpenAI API on :8888, model id Ripple-65B-Preview
scripts/stop.sh
```

`scripts/start.sh` serves one stream of 1M tokens by default. For several at once:

| Sparks | Streams x window | Command |
|---|---|---|
| 2 | 4 x 256K | `STREAMS=4 CONTEXT=262144 scripts/start.sh` |
| 4 | 8 x 256K | `STREAMS=8 CONTEXT=262144 scripts/start.sh` |
| 4 | 16 x 128K | `STREAMS=16 CONTEXT=131072 scripts/start.sh` |

Each stream reserves its whole window (the cache is not pooled), so these are the largest that fit. Other settings
are in `scripts/config.sh`: `PACK=5.0|6.0`, `DRAFTER=1|0`, `MAX_TOKENS` (default 32,768), `ROCE`, `GRAPHS`. A cold
start takes 20 to 60 seconds once the weights are in the page cache.

To make the pack yourself: `scripts/quantise.sh 5.0` (about 6 h on one GB10).

## Speed

5.0 bpw pack, draft model on unless marked, tokens per second.

| | Two Sparks, original recipe | Two Sparks, this recipe | Four Sparks, this recipe |
|---|---|---|---|
| One stream, coding, plain | 10.2 | 10.5 | 17.6 |
| One stream, coding | 36.1 | 36.3 | 57.4 |
| Four streams, code, total | | 109.9 | 155.7 |
| Sixteen streams, code, total | | | 228.8 |
| Prompt reading, 32K tokens | 399 | 611 | 1,109 |
| Reading a 1,015,070-token prompt | 1 h 48 min | 1 h 31 min | 47 min |

One stream on four Sparks, greedy, replies of up to 512 tokens:

| Task | Plain | Draft model |
|---|---|---|
| coding | 17.6 | 57.4 |
| agent tool call | 19.1 | 51.0 |
| chat | 17.4 | 25.1 |

Sixteen streams of sampled prose (temperature 0.7) decode 189 tokens per second in total on four Sparks.

## Quality

At greedy decoding, four Sparks give byte-identical replies to two. Four of the five engine changes leave replies
untouched by construction; pfdense rounds the prompt projections differently (WikiText perplexity 6.9793 to 6.9802),
and a full rerun of our test suite before and after showed no change on any test. On the suite (temperature 0.6,
top-p 0.95, 32,768-token limit), four Sparks score:

| Test | Score | Card |
|---|---|---|
| GPQA Diamond | 79.8 | 85.2 |
| MMLU-Pro, a fixed 700 | 81.6 | 84.2 |
| AIME 2025, four samples each | 90.8 | 98.3 |
| IFEval, prompt-level strict | 93.0 | |
| LiveBench coding | 74.2 | |
| BFCL multi-turn base / live multiple | 73.0 / 78.0 | |
| RULER-style, 32K and 128K | 77.2 | |

Most of the gap to the card is replies that run past 32,768 tokens without answering: among the GPQA replies that
finished, the score is 88.8. Set a higher `MAX_TOKENS` for reasoning work.

The pack against the original BF16 weights (96 rows of 2,048 tokens), from the original recipe:

| | 5.0 bpw | 6.0 bpw | BF16 |
|---|---|---|---|
| Same top token as BF16 | 97.9 % | 98.9 % | 100 % |
| Wikitext-2 perplexity (64 x 2048) | 9.270 | 9.248 | 9.245 |

## Engine patches

`patches-zig/` applies to TensorFold v1.2.3 in order; `scripts/zig_prepare.sh` checks the result against the pinned
source tree (`ZIG_TREE`).

| Patches | What they do |
|---|---|
| 0001-0009 | Ripple on CUDA, TP=2, native serving, the DFlash2 draft model (the original recipe; TensorFold PRs #2522 and #2523) |
| 0010 | TP=4 as well as 1 or 2 |
| 0011 | Reduce-scatter: at three ranks or more, each rank receives only its share of the summed prompt rows |
| 0012 | Prompt projections by pfdense, the fused prefill GEMM already in TensorFold, in chunks of 1,024 rows |
| 0013 | Each layer's prompt rows in two halves, one half's exchange overlapping the other's compute |
| 0014 | Shared decode rounds of up to 64 rows, priced from timed costs; one draft-model pass for every stream |
| 0015 | The draft model's pass is charged only to rounds that draft |

## Credits and licence

Thanks to ajensenwaud for the two-Spark recipe and the 5.0 bpw pack this starts from; Ricursive for
Ripple-65B-Preview and its DFlash2 draft model; Ash Hart ([@ashxhart](https://x.com/ashxhart)) and the TensorFold
contributors for [TensorFold](https://github.com/ashhart/TensorFold), including the multirank RoCE path the four-Spark
gathers reuse; BobClawblaw for the Ripple config in TensorFold PR #2231 (patch 0001); auryn-macmillan for
exllamav3's Ripple support (`feat/ricursive`), which made the packs; turboderp for
[exllamav3](https://github.com/turboderp-org/exllamav3); DeepSeek for the fused prefill GEMM; Mia AI Lab for the
TensorFold CUDA base image. Evaluation data: GSM8K (MIT), MMLU-Pro (MIT), WikiText-2 (CC BY-SA 3.0).

Recipe code: BSD 3-Clause. `patches-zig/` keep TensorFold's Apache-2.0. The model is Apache-2.0.
