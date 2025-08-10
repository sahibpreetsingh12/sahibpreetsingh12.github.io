# Sahibpreet Singh — LLM Systems Notes

> Practical, measurement-driven notes on building and optimizing large language model systems.

## About Me
I’m **Sahibpreet Singh**, working at the intersection of **AI/NLP** and **systems engineering**.  
Over the last few years, my focus has been on:

- **LLM inference optimization** — latency/throughput trade-offs, quantization, batching, KV cache engineering.
- **RAG (Retrieval-Augmented Generation)** — design, evaluation, grounding, and answer quality metrics.
- **Custom kernels** — writing and optimizing Triton and CUDA kernels for inference acceleration.
- **LLM evaluation frameworks** — creating robust metrics and pipelines that go beyond accuracy.

This blog is my public lab notebook: small, focused write-ups with runnable code and real measurements.

---

## What You’ll Find Here
- **Inference Engineering** — From KV cache paging to speculative decoding, with benchmarks.
- **RAG Evaluation** — Task design, retrieval quality scoring, and groundedness metrics that actually matter.
- **Kernel Programming** — Triton/CUDA walkthroughs for attention, matmul, and fused ops.
- **Build Logs** — Real-world engineering notes from deploying and iterating on LLM systems.

---

## Start Here
If you’re new, these are good entry points:
1. **Triton for the Impatient** — Vector add → tiled matmul → row-wise softmax → attention.
2. **A Practical RAG Evaluation Loop** — Data, retrieval, grounding, and answer scoring.
3. **LLM Throughput Engineering** — Prefill vs decode, quantization trade-offs, and batching strategies.

---

## Tech Stack
- **Static site**: GitHub Pages + Jekyll (theme: Minimal Mistakes or Chirpy)
- **Code**: Python, PyTorch, Triton, CUDA
- **Comments**: giscus (GitHub Discussions)
- **Analytics**: Privacy-friendly (Plausible / GoatCounter)

---

## Writing Principles
- Show measurements, not just intuition.
- State assumptions and hardware details.
- Keep posts concise, runnable, and reproducible.

---

## Local Development
```bash
# Requires Ruby & Bundler
bundle install
bundle exec jekyll serve --livereload
# Visit http://127.0.0.1:4000

