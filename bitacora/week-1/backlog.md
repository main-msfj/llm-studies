# This is the backlog from 29/10/2026-08/11/2026

# (10-29-2026) Environment

I watched the video on environment setup and built the dev container for this project.

## Quick wins
- I understand better how a devcontainer and a Dockerfile fit together.
- The container has GPU access, a non-root user, and persistent volumes for Claude config, the Hugging Face cache and the uv cache. I also mounted my host SSH keys so I don't re-enter credentials.

## Technical debt
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


# (10-30-2026) Tokenization and Embeddings

I watched [this video](https://youtu.be/341Rb8fJxY0) on tokenization, embeddings and positional embeddings. It is essential for building and understanding how an LLM turns text into model input.

## Quick wins
- It refreshed basic concepts I already knew, and it was nice to review them.
- My PyTorch skills were a bit rusty, but they are coming back.

## Technical debt
- None.

## Pending 
 - Code 

# (10-31-2026) Positional Enconding 

I Learned the differences in between positional encoding from Attention is all you need 
and GPT2 Implementation , I clearly Understand now the benefit of Positional Enconding 
in Attention is all you need the values are set by the formula, in constrant
in gpt2 are sum into the positional vector this make them traininable 
now i understand that each vector is move in the space but always close to the original 
vector space , this allow the transformer to move the embedding vector to define better 
meaning and position 

## Technical debt
- None 

## Pending 
- Do the Positonal Encoding in Pytorch , also understand what are the best
pracice for embedding_dimension, stride , and batches 
as this affect the training and LLm performance 

## update 
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

# (11-01-2026) Neural Network

I re-studied the perceptron and activation functions. I first built a single
perceptron by hand, then used `nn.Sequential` to build a neural network and
trained it on a toy dataset.

## Quick wins
- I connected the project to Kaggle so I can train on its GPUs later.
- The CI pipeline is already working.

## Pending
- Implement micrograd from Andrej Karpathy. I understand the concept and the
  code structure, but I still need to write my own version.

# (11-02-2026 | 11-03-2026) 
I continue in understanding | refreshing the math behind NN 
and aso build micrograd from scrath , i need to finish the entire video and implement the Code (I am doing now )

# (11-04-2026 ~  11-06-2026)
I continue with Andrej Karpathy
but now i understand the back propagation in detail chain rules 
and refresh some moew concepts 