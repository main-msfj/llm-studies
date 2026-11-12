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
- Keep `stoi` / `itos` in the `Makemore` instance so they can be reused for plotting and sampling.
- Continue with makemore: turn the counts into probabilities and sample names.
