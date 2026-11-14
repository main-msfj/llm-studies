# Week 1 (2026-10-29 → 2026-11-08)

## 2026-10-29 · Environment

I watched the video on environment setup and built the dev container for this project.

### Quick wins
- I understand better how a devcontainer and a Dockerfile fit together.
- The container has GPU access, a non-root user, and persistent volumes for Claude config, the Hugging Face cache and the uv cache. I also mounted my host SSH keys so I don't re-enter credentials.

### Technical debt
- I don't fully understand every Dockerfile command yet. I get the big picture (non-root user, mounted volumes), but not the details of this block:

```dockerfile
# Non-root user setup
RUN userdel -r ubuntu 2>/dev/null || true \
    && groupadd -g ${GID} ${USERNAME} \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}
```

What it does, line by line:
- `userdel -r ubuntu`: removes the default `ubuntu` user, which already takes UID 1000.
- `groupadd` / `useradd`: create the `dev` user with UID/GID 1000, so files created in the container have the same owner as on my host.
- `echo ... > /etc/sudoers.d/dev`: lets `dev` use `sudo` without a password.

Is it worth learning in depth? Probably not. I'm not a DevOps engineer, and this block is boilerplate I can reuse as is. I only need to remember the purpose: matching UID/GID with the host so bind-mounted files stay editable. I'll revisit it only if file permissions break.

## 2026-10-30 · Tokenization and embeddings

I watched [this video](https://youtu.be/341Rb8fJxY0) on tokenization, embeddings and positional embeddings. It is essential for understanding how an LLM turns text into model input.

### Quick wins
- It refreshed basic concepts I already knew, and it was nice to review them.
- My PyTorch skills were a bit rusty, but they are coming back.

### Technical debt
- None.

### Pending
- Write the code.

## 2026-10-31 · Positional encoding

I learned the difference between the positional encoding in *Attention Is All You Need* and the one in GPT-2:

- **Attention Is All You Need:** the positional values are fixed, computed with a sine/cosine formula.
- **GPT-2:** the positional vectors are learned (trainable) and added to the token embeddings.

Now I understand that adding the positional vector moves each token embedding in the vector space, but it stays close to its original position. This lets the transformer encode both the meaning of a token and its position.

### Technical debt
- None.

### Pending
- Implement positional encoding in PyTorch.
- Understand best practices for `embedding_dim`, `stride` and batch size, since they affect training and LLM performance.

### Update
- On stride: it is the step between the start of consecutive windows.
  * `stride = max_length`: no overlap, so each token appears once per epoch.
    A good default when the dataset is large.
  * `stride < max_length`: windows overlap, so we get more samples from the same
    text, but they are highly correlated (the model sees the same tokens many
    times). Useful for small datasets, with a higher risk of overfitting.
  * `stride > max_length`: skips tokens, so data is lost. Avoid it.
- On batch size: it is limited mainly by GPU memory. It also changes training:
  small batches give noisier gradients, large batches give smoother ones
  (and usually need a different learning rate).

## 2026-11-01 · Neural networks

I re-studied the perceptron and activation functions. I first built a single
perceptron by hand, then used `nn.Sequential` to build a neural network and
trained it on a toy dataset.

### Quick wins
- I connected the project to Kaggle so I can train on its GPUs later.
- The CI pipeline is already working.

### Pending
- Implement micrograd from Andrej Karpathy. I understand the concept and the
  code structure, but I still need to write my own version.

## 2026-11-02 → 2026-11-03 · Math behind neural networks

I kept refreshing the math behind neural networks and started building micrograd from scratch.

### Pending
- Finish the whole video and implement the code (in progress).

## 2026-11-04 → 2026-11-06 · Backpropagation

I continued with Andrej Karpathy's video. Now I understand backpropagation in detail, including the chain rule, and I refreshed other related concepts.

## 2026-11-07 · micrograd

I continued with micrograd and my understanding is much better now:

- I understand the basics of the forward and backward pass in PyTorch.
- I refreshed derivatives and other related concepts.

## 2026-11-08 · nanograd finished

I finished nanograd.


# Week 2 (2026-11-09 → 2026-11-15)

## 2026-11-12 · Tokenizer, embeddings and makemore

I built the embeddings, understood them, and reorganized the week 2 notebook. I also started learning makemore and how to build a tokenizer for an LLM in a more advanced way.

### Quick wins
- Built a word-level tokenizer: `SimpleTokenizerV1` (fails on unknown words) and `SimpleTokenizerV2` (replaces them with `<|unk|>`).
- Learned that special tokens such as `<|unk|>` and `<|endoftext|>` must be in the vocabulary. Otherwise V2 fails with `KeyError: '<|unk|>'`.
- Reviewed how to validate arguments: `if ... raise TypeError/ValueError` for external input, `assert` only for internal invariants (it is removed with `python -O`), and `typing.Annotated` is only metadata unless a library like pydantic reads it.
- Started makemore: counted character bigrams with `pairwise` and `Counter`, and built the count matrix `N[i, j]` in PyTorch.
- Clarified the difference between instance methods (`self`), `@classmethod` (`cls`, e.g. alternative constructors like `from_file`) and `@staticmethod`.

### Pending
- ~~Keep `stoi` / `itos` in the `Makemore` instance so they can be reused for plotting and sampling.~~ Done 2026-11-14.
- ~~Continue with makemore: turn the counts into probabilities and sample names.~~ Done 2026-11-14.

## 2026-11-14 · makemore part 1: from counting to a neural network

I finished the first makemore video. First I built the bigram model by counting, then a one-layer neural network trained with gradient descent. Both reach the same loss (~2.45-2.48), so the network rediscovers the count table by itself.

### Quick wins
- **Sampling:** generate a name letter by letter with `torch.multinomial`, starting and ending at `.` (index 0). A seeded `torch.Generator` makes results reproducible, but only on the same PyTorch version, which is why I don't get the video's `mor. axx.`.
- **Broadcasting:** align shapes to the right; a size-1 (or missing) dim gets stretched; otherwise sizes must match. A vector `(n,)` always behaves like a row `(1, n)`.
  * `P / P.sum(1, keepdim=True)` normalizes rows. Without `keepdim` it silently normalizes columns, and no error is raised because `P` is square (27×27).
  * Habit: write the aligned shapes in a comment and check a property, e.g. `assert torch.allclose(P.sum(1), torch.ones(27))`.
  * `@` broadcasts only the batch dims; the last two follow `(n, m) @ (m, k)`.
- **Smoothing (`N + 1`):** "never seen" ≠ "impossible". It removes zero probabilities so the loss can't become infinite (`log(0) = -inf`), and barely changes the result.
- **Negative log likelihood:** loss = mean of `-log(prob given to the correct letter)`. Why log: it turns a product of probabilities into a sum, keeps the same optimum, punishes confident mistakes hard, and measures "surprise".
- **Perplexity = e^loss:** 2.45 → ~11.6, i.e. the model gives the correct letter ~8.6 % on average. Uniform guessing = 27. Above 27 means *confidently wrong* (a random `W` gave ~40).
- **The network:**
  * Dataset: pairs `xs` (current letter) and `ys` (next letter), 228,146 examples.
  * `one_hot @ W` just selects a row of `W`. `W` has shape `(inputs, outputs)`, `(27, 27)` here; its shape doesn't depend on the number of examples.
  * `logits.exp()` = "counts" (equivalent to `N`), and exp + normalize = softmax. `exp` keeps the order and is never 0.
  * No bias: with one-hot input, `W[ix] + b` can be absorbed into `W`. Verified by experiment: same loss with and without bias.
  * Learning rate 50: gradients here are tiny (~0.002), so a large step is needed. Tested 0.1 / 1 / 10 / 50 / 200; 50 worked best.
  * Compare models by loss, not by a few sampled names.
- **Python:** `torch.tensor` (function, creates data) vs `torch.Tensor` (class, for type hints); `tuple[A, B]` vs `list[A]`; `k + 1 // 5` is `k + 0` (precedence), use `(k + 1) % 5 == 0`.

### Technical debt
- Index tricks like `torch.bincount(xs * 27 + ys)` are still hard for me. I keep the explicit loop for now; understanding comes first.

### Pending
- makemore part 2 (MLP): character embeddings, a hidden layer with `tanh`, and a context of 3 letters.
- Use `F.cross_entropy` instead of writing exp → normalize → log by hand.
