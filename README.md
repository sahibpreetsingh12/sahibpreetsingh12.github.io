Here’s a README built around **“Sahibpreet Singh — LLM Systems Notes”** that matches your style and avoids fluff while still feeling inviting:

---

````markdown
# Sahibpreet Singh — LLM Systems Notes

> Practical, measurement-driven notes on building and optimizing large language model systems.

## 👋 About Me
I’m **Sahibpreet Singh**, working at the intersection of **AI/NLP** and **systems engineering**.  
Over the last few years, my focus has been on:

- **LLM inference optimization** — latency/throughput trade-offs, quantization, batching, KV cache engineering.
- **RAG (Retrieval-Augmented Generation)** — design, evaluation, grounding, and answer quality metrics.
- **Custom kernels** — writing and optimizing Triton and CUDA kernels for inference acceleration.
- **LLM evaluation frameworks** — creating robust metrics and pipelines that go beyond accuracy.

This blog is my public lab notebook: small, focused write-ups with runnable code and real measurements.

---

## 📚 What You’ll Find Here
- **Inference Engineering**: From KV cache paging to speculative decoding, with benchmarks.
- **RAG Evaluation**: Task design, retrieval quality scoring, and groundedness metrics that actually matter.
- **Kernel Programming**: Triton/CUDA walkthroughs for attention, matmul, and fused ops.
- **Build Logs**: Real-world engineering notes from deploying and iterating on LLM systems.

---

## 🚀 Start Here
If you’re new, these are good entry points:
1. **Triton for the Impatient** — Vector add → tiled matmul → row-wise softmax → attention.
2. **A Practical RAG Evaluation Loop** — Data, retrieval, grounding, and answer scoring.
3. **LLM Throughput Engineering** — Prefill vs decode, quantization trade-offs, and batching strategies.

---

## 🛠 Tech Stack
- **Static site**: GitHub Pages + Jekyll (theme: Minimal Mistakes / Chirpy)
- **Code**: Python, PyTorch, Triton, CUDA
- **Comments**: giscus (GitHub Discussions)
- **Analytics**: Privacy-friendly (Plausible / GoatCounter)

---

## ✍ Writing Principles
- Show measurements, not just intuition.
- State assumptions and hardware details.
- Keep posts concise, runnable, and reproducible.

---

## ⚡ Local Development
```bash
# Requires Ruby & Bundler
bundle install
bundle exec jekyll serve --livereload
# Visit http://127.0.0.1:4000
````

---

## 📬 Contact

* **Email**: [ss9334931@gmail.com](mailto:ss9334931@gmail.com)
* **GitHub**: [sahibpreetsingh12](https://github.com/sahibpreetsingh12)

---

**License**
Content © Sahibpreet Singh. Code snippets are MIT unless otherwise noted. Please link back if you reuse.

```

---

If you want, I can also make the **_config.yml** for Jekyll so your GitHub Pages blog shows “Sahibpreet Singh — LLM Systems Notes” as the site title and tagline right away. That way, you just push and it’s live.  

Do you want me to prepare that next?
```

