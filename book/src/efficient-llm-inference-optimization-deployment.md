# Efficient LLM Inference, Optimization & Deployment

**Having a good model is only half the problem.** Once a model is trained, we still need to run it efficiently enough that real users can actually use it at an acceptable speed and cost.

Training gets a lot of attention because it is expensive and technically impressive, but inference is the recurring cost. Training might happen once, while inference happens every time a user sends a message, an agent makes a call, a document is processed, or a RAG system generates an answer. At scale, even a small improvement in inference efficiency can therefore save a huge amount of money.

This is one reason companies sometimes self-host models instead of relying entirely on hosted APIs. Self-hosting can reduce cost at sufficient scale, keep sensitive data inside the company's own environment, give complete control over model versions and availability, and make it possible to fine-tune, quantize, and configure the model specifically for the workload.

A production deployment is always balancing three things:

**accuracy, performance, and cost.**

You usually cannot optimize one without affecting the others. A very large model might give excellent answers but require expensive hardware. A tiny model might be cheap and fast but not accurate enough. The goal of these techniques is not to eliminate those trade-offs completely, but to push the boundary so that we can get more performance and lower cost without sacrificing much quality.

---

## How LLM inference actually works

A simple way to think about the production stack is:

```text
Model → Inference Server → GPU
```

For example:

```text
Llama / Qwen → vLLM → NVIDIA GPU
```

For experiments or a single user, we can load a model directly with PyTorch or Hugging Face Transformers. But once many users are involved, an inference server becomes important because it handles scheduling, batching, memory management, caching, and other optimizations that make much better use of the hardware.

LLMs generate text **one token at a time**. If a response contains 500 generated tokens, the model roughly performs 500 sequential generation steps. This is very different from something like image classification, where a single forward pass might immediately produce the final answer.

Inside the model, most of the work happens in transformer blocks. Each block contains self-attention and a feed-forward network, and most of the parameters live inside large linear layers. These linear layers are essentially giant matrix multiplications, so they account for a large part of both model size and computation.

In self-attention, we usually talk about four projections:

* **Q — Query**
* **K — Key**
* **V — Value**
* **O — Output**

A useful mental model is that the **query** represents what the current token is looking for, the **key** describes what information another token contains, and the **value** contains the information itself.

The current query is compared with the keys of previous tokens. Those scores determine how strongly the model should pay attention to each previous token, and the corresponding values are combined to produce the attention output.

The important observation is that when the model generates a new token, the keys and values of all the previous tokens **do not change**. Recalculating them every generation step would be extremely wasteful. Instead, inference systems store them in GPU memory. This stored information is called the **KV cache**.

So if the model is generating token 1000, it does not recompute K and V for tokens 1 through 999. It reuses the cached values and calculates only what is new.

The approximate KV-cache memory needed per token is:

```text
2 × number_of_layers × number_of_KV_heads × head_dimension × bytes_per_value
```

The `2` is there because both Keys and Values are stored.

For a Llama 3 70B example with 80 layers, 8 KV heads, a head dimension of 128 and 2 bytes per value, this works out to roughly **320 KB of KV cache per token**.

That becomes surprisingly large with long contexts:

| Context length | Approx. KV cache |
| -------------- | ---------------: |
| 2K tokens      |          ~640 MB |
| 8K tokens      |          ~2.5 GB |
| 32K tokens     |           ~10 GB |
| 128K tokens    |           ~40 GB |

This is one of the most important things to remember: **long context is not free**.

A single 128K-token request can use roughly 40 GB of KV-cache memory in this example. Ten concurrent long-context users could require more than 400 GB just for KV cache.

GPU memory therefore has to hold more than just the model:

```text
GPU memory
=
model weights
+
KV cache
+
temporary activations
+
runtime overhead
```

For a 70B model stored at roughly 2 bytes per parameter:

```text
70 billion × 2 bytes ≈ 140 GB
```

So the model weights alone use about 140 GB.

If the deployment uses four 80 GB GPUs, there is 320 GB total memory. After loading roughly 140 GB of model weights, the remaining memory is needed for KV cache, activations and runtime buffers. How efficiently that remaining memory is managed directly affects how many users can be served at once.

Another important idea is the **GPU memory hierarchy**.

Conceptually:

```text
CPU RAM
   ↓
GPU HBM / VRAM
   ↓
small on-chip memory / SRAM
   ↓
Tensor Cores
```

The closer memory is to the compute units, the faster it is, but the less of it there is.

Model weights and KV cache mostly live in GPU HBM. During inference, pieces of this data are moved closer to the Tensor Cores so the matrix multiplications can be performed.

This leads to one central insight:

> LLM inference is not only a compute problem. It is also a data-movement problem.

A powerful GPU can spend a lot of time simply waiting for data to move through the memory hierarchy. Many optimizations therefore work by moving less data, reusing data that has already been computed, or managing memory more efficiently.

---

## Making models smaller with quantization

Model sizes have grown much faster than GPU memory capacity. Models have gone from tens of millions of parameters to billions, hundreds of billions and, in some cases, beyond that. Bigger models require more GPUs, move more data, leave less memory available for KV cache, consume more energy, and increase infrastructure cost.

Compression is one way to close this gap.

The main compression technique here is **quantization**.

A model contains enormous numbers of numerical weights. Instead of storing every value with high precision, we can represent it using fewer bits.

Conceptually, it is similar to storing:

```text
3.14159265
```

as:

```text
3.14
```

We lose some numerical precision, but need less storage.

Typical formats include FP32, BF16, FP16, FP8, INT8 and INT4.

Most modern LLMs are commonly released around BF16 precision. BF16 uses 16 bits per number. Moving from 16 bits to 8 bits roughly halves the storage needed for the affected weights. Moving from 16 bits to 4 bits reduces it by about 75%.

The approximate weight memory of a model is:

```text
number of parameters × bits per parameter / 8
```

For example, a 109B-parameter model at 16 bits requires approximately:

```text
109B × 2 bytes ≈ 218 GB
```

At 8 bits:

```text
109B × 1 byte ≈ 109 GB
```

At 4 bits:

```text
109B × 0.5 byte ≈ 55 GB
```

So the same model could potentially move from needing several GPUs to fitting on one much larger GPU, depending on the hardware and the rest of the runtime memory requirements.

Quantization does more than just reduce storage. If weights are half the size, there is approximately half as much weight data to move from HBM toward the compute units. Since memory movement is such an important part of inference, this can directly improve latency.

There are two main things we can quantize: **weights and activations**.

Weights are the learned model parameters. Activations are the temporary tensors created while the model is processing data.

This gives notation such as:

```text
W8A16
```

meaning 8-bit weights and 16-bit activations, or:

```text
W8A8
```

meaning 8-bit weights and 8-bit activations.

With weight-only quantization such as W8A16, the main benefit is lower memory usage and less data movement. The compressed weights may still be converted back to a higher precision before the matrix multiplication.

With W8A8, both the weights and activations use lower precision. On hardware that supports fast low-precision Tensor Core operations, this can reduce both memory movement and computation time.

So a simple way to remember it is:

```text
Weight quantization
→ less memory and bandwidth

Activation quantization
→ potentially faster computation

Both
→ potentially improve both sides
```

Quantization also indirectly helps the KV cache. If the model weights become smaller, more GPU memory remains available for active requests. That can mean more concurrent users, larger batches, or longer contexts.

Another technique is **sparsification**. Quantization reduces the number of bits used to represent values. Sparsification instead sets less-important weights to zero. A common structured pattern is 2:4 sparsity, where two values out of every four are zeroed. If the hardware can exploit this structure, both memory and computation may be reduced.

A useful distinction is:

```text
Quantization → smaller numbers
Sparsification → fewer useful/non-zero numbers
```

Good quantization is not simply blind rounding. If every weight is naively converted to INT4, model quality can degrade badly.

Several smarter techniques were discussed.

**Round-to-nearest** is the simplest. Each number is converted to the nearest representable value. It is fast and requires no calibration data, but tends to perform poorly when compression becomes aggressive.

**AWQ — Activation-Aware Weight Quantization** is based on the observation that not every weight matters equally. Some weights correspond to important activations, so changing them slightly can have a large effect on model output. AWQ runs representative calibration data through the model, identifies which parts are more sensitive, and protects those more carefully while compressing the rest more aggressively.

**GPTQ** takes a more mathematical approach. It tries to estimate how sensitive the model is to changes in each weight and then compensates for quantization errors in the remaining weights. This can preserve accuracy very well, although the compression process itself requires more computation and memory.

That extra cost is usually acceptable because quantization is generally an **offline operation performed once**. Spending more GPU time once can be worthwhile if the resulting model will later serve millions of requests more cheaply.

Many of these methods require **calibration data**. The idea is to send a relatively small but representative dataset through the model so the quantizer can observe which values are important.

Around **256 calibration samples** is a reasonable example. More data can help, but after a few hundred representative samples, improvements may become small while quantization time continues to increase.

The practical workflow is roughly:

```text
Choose model
→ choose quantization algorithm
→ choose precision
→ choose calibration data
→ quantize
→ save compressed model
→ evaluate quality
→ deploy
```

One practical toolchain uses **LLM Compressor**, GPTQ, and a W4A16 configuration.

One important practical detail is that usually not every component of the model is quantized. The biggest targets are the linear layers because that is where most parameters and matrix multiplications live. Components such as the LM head, embeddings or normalization layers may stay at higher precision to protect model quality.

This explains why the theoretical compression ratio and the actual model-file reduction may differ.

Moving weights from 16-bit to 4-bit sounds like a 75% reduction, but a small model can become only about **42% smaller overall** because several components remain in higher precision. For much larger models, the linear weights make up a greater share of the total size, so the result may get closer to the theoretical compression ratio.

After compression, we have to check whether the model still works.

Looking at a few generated answers is useful for intuition, but we also need numerical measurements. One metric introduced here was **perplexity**.

Perplexity measures how well the model predicts text. At each point in a sequence, the model predicts the next token. We calculate cross-entropy loss between that prediction and the token that actually appears, average those losses, and exponentiate the result.

Conceptually:

```text
Perplexity = exp(average cross-entropy loss)
```

Lower is better.

In the example:

```text
Base model perplexity:      ~32.79
Quantized model perplexity: ~35.48
```

The quantized model was roughly 8% worse on that metric.

Whether that is acceptable depends on the use case. If a relatively small quality change allows the deployment to use much less hardware and serve far more requests, it may be an excellent trade.

The important principle is:

> Never judge an optimization only by how much smaller or faster the model became. Measure what happened to quality as well.

---

## Serving many users efficiently with vLLM

After making the model smaller, the next challenge is serving it to many users without wasting GPU compute or memory.

Three techniques are especially useful:

**continuous batching, PagedAttention, and prefix caching.**

The first problem is GPU utilization.

Every generated token requires another forward pass. If we serve only one request at a time, a huge model's weights are moved through the GPU memory system to perform relatively little useful work for that one token.

It is more efficient to process several users together so the same model weights can contribute to multiple requests.

Traditional **static batching** collects a fixed group of requests, runs them together, and waits until every request finishes before starting another batch.

That works well for workloads with predictable runtime, such as image classification. Ten images usually take roughly similar amounts of work.

LLMs behave differently because response lengths are unpredictable.

One request might ask:

```text
What's 2 + 2?
```

and generate five tokens.

Another might ask for a 2,000-word essay.

If both are in a static batch, the short request may finish almost immediately, but its batch slot remains empty until the longest request finishes.

**Continuous batching** solves this by continuously changing the active batch.

As soon as one request finishes, another waiting request can immediately take its place.

Conceptually:

```text
Request finishes
→ slot becomes free
→ new request immediately enters
```

The GPU therefore stays busy instead of waiting for the slowest member of a fixed batch.

The second major issue is KV-cache memory.

Each active request has its own KV cache, and nobody knows in advance exactly how long a request will become. Older systems sometimes reserved one large contiguous chunk of memory based on the maximum possible length.

Suppose the system reserves room for 2,048 tokens, but the request actually uses only 400. Most of that reserved space sits empty.

This creates **internal fragmentation**: unused memory inside an allocation.

There can also be gaps between allocations that are physically free but too small or awkwardly placed for another large request. This is **external fragmentation**.

The early work that led to vLLM reported situations where only around **20–40% of the reserved KV-cache memory was actually storing real token data**, with much of the rest lost to fragmentation and over-reservation.

**PagedAttention** solves this by borrowing the idea of paging from operating systems.

Instead of giving every request one huge contiguous memory block, the KV cache is split into many small fixed-size blocks.

A request might use:

```text
Block 3
Block 6
Block 11
Block 20
```

These blocks do not need to be physically next to one another.

A block table tells the system where each part of the request's KV cache lives.

Memory is allocated only when the request needs it. If one block still has free token slots, those slots are filled first. Only once it is full does the request receive another block.

When the request finishes, its blocks immediately return to the shared free pool.

The result is much better memory utilization and therefore more simultaneous requests on the same hardware.

The third optimization is **prefix caching**.

Many requests have identical beginnings. For example, every request to a chatbot may contain the exact same system prompt, perhaps along with the same few-shot examples or RAG instructions.

Without prefix caching, the model recomputes the KV cache for those same tokens every time.

With prefix caching:

```text
Compute shared prefix once
→ store KV cache
→ reuse it
```

This is especially useful for shared system prompts, few-shot examples, repeated RAG context, and multi-turn conversations.

In a conversation, the second turn usually contains the entire first turn plus some new tokens. If that existing prefix is unchanged, the server can reuse its cached KV representation and process only the new part.

With very small prompts, the gain may not be noticeable. But if thousands of users share a 2,000- or 5,000-token prompt, eliminating that repeated work can save enormous amounts of compute.

The easiest way to remember the three techniques is:

| Technique           | Main waste it removes  |
| ------------------- | ---------------------- |
| Continuous batching | Idle GPU compute       |
| PagedAttention      | Wasted KV-cache memory |
| Prefix caching      | Repeated computation   |

These techniques are implemented in inference engines such as **vLLM**.

vLLM loads the model, manages its KV cache, schedules requests, performs continuous batching and exposes the model through an HTTP server.

A useful feature is that vLLM exposes an **OpenAI-compatible API**. Applications can often use almost the same client code they would use for a hosted OpenAI-style API and simply point it at the local vLLM server instead. This makes switching between hosted and self-hosted models much easier.

The maximum configured context length also matters. vLLM has to allocate and plan memory around expected context sizes. If the application only needs 4K context, there is little reason to configure an enormous maximum context just because the model technically supports one. Production configuration should reflect the actual workload.

Running the inference server ourselves also gives access to useful metrics such as:

* active requests,
* waiting requests,
* KV-cache utilization,
* prompt tokens processed,
* generated tokens.

These tell us what the server is actually struggling with.

For example, a growing queue of waiting requests suggests that the system is reaching its serving capacity. A nearly full KV cache suggests that memory may be limiting concurrency.

vLLM can also expose **log probabilities** for generated tokens. These tell us how strongly the model preferred a token over alternatives.

For example, the model might assign a high probability to `"Paris"` after:

```text
The capital of France is
```

This is useful for inspecting model behavior, but it is important not to confuse high token probability with factual certainty. A model can strongly prefer an answer and still be wrong.

---

## Benchmarking performance and checking model quality

Once the model has been optimized and deployed, we still have to answer two separate questions:

> **Is the deployment fast enough?**

and:

> **Is the model still good enough?**

The first is a performance question. The second is a model-quality question.

Before benchmarking anything, it helps to define **Service Level Objectives, or SLOs**.

A latency of 300 ms means nothing by itself. For a highly interactive shopping assistant, it might feel slow. For a RAG system that produces a carefully grounded response, it might be perfectly acceptable.

Example SLOs for a chatbot might be:

```text
TTFT < 200 ms
ITL  < 50 ms
```

while a RAG system could tolerate something closer to:

```text
TTFT < 300 ms
ITL  < 100 ms
End-to-end latency < 3 s
```

These numbers are examples rather than universal rules. The important idea is to define the application's expectations **before** looking at the benchmark.

Some important metrics are:

**Time to First Token (TTFT)** — how long the user waits before anything appears.

**Inter-Token Latency (ITL)** — how quickly later tokens arrive during streaming.

**End-to-end latency** — how long the entire request takes.

**Throughput** — how much total work the server handles, often in tokens per second or requests per second.

Latency and throughput describe different things. Latency describes the experience of one request, while throughput tells us how much work the whole system can handle.

One particularly important lesson is that **averages can hide terrible user experiences**.

Imagine:

```text
Mean latency: 300 ms
p95 latency:  1.5 s
p99 latency:  5 s
```

The mean looks excellent, but one percent of requests are taking five seconds or more. With millions of requests, one percent is still a huge number of users.

So production performance should be examined using p50, p95 and p99, not only the mean.

A large gap between the average and p95/p99 usually indicates **tail-latency problems**.

For performance benchmarking, **GuideLLM** is designed specifically for LLM inference, so unlike a generic HTTP load tester it understands streaming and metrics such as TTFT and ITL.

Different traffic profiles answer different questions.

**Synchronous traffic** sends one request at a time and waits for it to finish. This gives a clean single-request baseline with no queueing.

**Concurrent traffic** keeps several requests running simultaneously and shows how the server behaves with multiple users.

**Constant-rate traffic** sends requests at a fixed rate, such as ten requests per second.

**Poisson traffic** sends requests with random spacing around an average rate, which better resembles real human traffic.

A **sweep** runs several load levels and produces a performance curve.

The sweep is useful because servers usually have a saturation point. Below it, traffic increases may have relatively little effect on latency. Near capacity, a small increase in request rate can suddenly produce large queues and rapidly increasing latency.

Conceptually:

```text
Load increases slowly
→ latency stays manageable
→ server approaches capacity
→ queue builds
→ TTFT and total latency rise sharply
```

Finding this point is important for autoscaling and capacity planning. A production system should normally operate below saturation with some headroom for traffic spikes.

Benchmark results always reflect the **entire stack**, not just the model.

Performance depends on:

* model architecture,
* model size,
* quantization,
* inference engine,
* GPU type,
* context length,
* output length,
* batching settings,
* concurrency,
* traffic pattern,
* KV-cache configuration.

So a benchmark saying that a model achieves 500 tokens per second somewhere else does not mean your deployment will achieve the same number.

You need to benchmark the configuration you actually plan to run.

Performance alone is still not enough. A deployment that produces 1,000 tokens per second is useless if its answers are wrong.

For model-quality evaluation, use **lm_eval**, the LM Evaluation Harness.

It can run standardized benchmarks such as MMLU, HellaSwag, GSM8K, ARC, TruthfulQA and many others.

This makes it possible to compare the original model with an optimized or quantized version and see whether accuracy changed.

Public benchmarks are useful, but the most important evaluation is usually the one that reflects the actual application.

If you are building a customer-support bot, you probably care more about whether it answers your company's support questions correctly than whether it performs well on competition mathematics.

A strong evaluation strategy therefore combines:

```text
public benchmarks
+
your own domain-specific tests
```

Another important lesson is not to trust tiny evaluation samples too much.

With only about 20 HellaSwag examples, a single question changes the measured accuracy by:

```text
1 / 20 = 5 percentage points
```

So the result will naturally be noisy.

Published model-card results may use thousands of examples and different prompting setups, which explains why they can differ significantly from a tiny local test.

When comparing benchmark numbers, always check the evaluation conditions:

* number of examples,
* zero-shot or few-shot,
* prompt template,
* benchmark version,
* model precision,
* tokenizer or chat template,
* generation settings.

Two accuracy numbers are not necessarily comparable just because they have the same benchmark name.

Model cards are also a useful source of evidence. Quantized model publishers often report both the original and compressed-model benchmark results.

A simple recovery metric is:

```text
accuracy recovery
=
quantized accuracy / base accuracy × 100
```

For example:

```text
Base accuracy:      43.04
Quantized accuracy: 41.02
```

gives approximately:

```text
95.3% accuracy recovery
```

A production decision therefore usually combines three kinds of evidence:

| Evidence   | What it tells us            |
| ---------- | --------------------------- |
| GuideLLM   | How fast the deployment is  |
| lm_eval    | How well the model answers  |
| Model card | What the publisher measured |

---

## Putting everything together

The real power comes from combining the techniques rather than using them in isolation.

Start with a normal model. Quantization makes its weights smaller. Smaller weights use less GPU memory and move through the memory hierarchy faster. That leaves more memory available for KV cache.

Then serve it with something like vLLM. Continuous batching keeps the GPU busy, PagedAttention makes KV-cache memory much more efficient, and prefix caching avoids repeating work.

Then benchmark the deployment under realistic traffic and evaluate the model to make sure those optimizations did not damage the quality that matters to your application.

A real-world example involved a company using a 70B model for SQL generation. After applying W4A16 quantization, the deployment reportedly retained more than 99% of baseline accuracy while reducing the requirement from eight GPUs to two — roughly a 75% infrastructure reduction.

Another example showed why optimization has to match the workload. A company processing millions of records initially received little benefit from its chosen quantization setup. After adjusting the optimization approach and tuning things such as batch size and concurrency, GPU hours reportedly dropped by around 40%.

The lesson is that there is **no universally best optimization**.

The right configuration depends on:

```text
model
+
hardware
+
context length
+
traffic pattern
+
batch size
+
concurrency
+
latency requirements
+
accuracy requirements
```

W4A16 can be attractive when memory capacity and bandwidth are the main bottlenecks.

W8A8 or FP8 can be especially useful on newer hardware that supports fast low-precision computation because it reduces both data movement and compute cost.

A practical deployment workflow therefore looks something like this:

```text
Understand the workload
        ↓
Define accuracy, latency and cost SLOs
        ↓
Choose a model
        ↓
Measure the unoptimized baseline
        ↓
Quantize / optimize the model
        ↓
Evaluate quality again
        ↓
Serve with an optimized inference engine
        ↓
Benchmark realistic traffic
        ↓
Find the saturation point
        ↓
Tune hardware and serving configuration
        ↓
Measure again
```

This should be an iterative process.

```text
Measure
→ identify bottleneck
→ change something
→ benchmark
→ evaluate quality
→ compare with SLOs
→ keep or reject the change
→ repeat
```

Some useful production metrics to monitor continuously are:

* requests per second,
* active and waiting requests,
* TTFT,
* ITL,
* end-to-end latency,
* p50 / p95 / p99 latency,
* prompt tokens per second,
* generated tokens per second,
* GPU memory utilization,
* KV-cache utilization,
* failed requests,
* timeouts,
* out-of-memory errors.

Quality should also be checked periodically with regression tests and domain-specific evaluations, because performance improvements are not useful if model behavior silently gets worse.

The final lesson briefly introduced **disaggregated inference** as a possible next step.

LLM inference has two somewhat different phases.

During **prefill**, the input prompt is processed and the KV cache is created. This phase can process many prompt tokens in parallel.

During **decode**, output tokens are generated one at a time.

Because these two phases behave differently, advanced systems can separate them and scale or optimize them independently.

Conceptually:

```text
Prefill workers
      ↓
KV cache / handoff
      ↓
Decode workers
```

This follows the same philosophy: understand where the actual bottleneck is, then optimize that specific part instead of treating the entire inference process as one undifferentiated task.

The most useful mental summary of everything is probably this:

```text
Quantization
→ move fewer bits

Continuous batching
→ don't leave GPU compute idle

PagedAttention
→ don't waste KV-cache memory

Prefix caching
→ don't recompute identical work

GuideLLM
→ measure serving performance

lm_eval
→ measure model quality

SLOs
→ decide whether the result is actually good enough
```

The biggest lesson is that **efficient LLM deployment is a systems problem**.

The model itself is only one component. Model precision, GPU memory, memory bandwidth, KV cache, context length, batching, concurrency, caching, hardware, traffic, latency, throughput, accuracy and cost all interact.

The difference between a model that simply *runs* and a model that can be economically served to thousands or millions of users usually comes from optimizing this entire system together.

> **Don't just optimize the model. Optimize the whole inference system.**

---

## Source credit

This entry is based on ideas and material from DeepLearning.AI's [Fast and Efficient LLM Inference with vLLM](https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm/).
