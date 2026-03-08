---
title: GPT, Explained
---
Andrej Karpathy recently published [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — 200 lines of Python that implement a GPT from scratch with zero dependencies. No PyTorch, no NumPy, nothing to install. Those 200 lines contain everything: dataset loading, tokenization, an autograd engine, a GPT-2-style neural network, the Adam optimizer, and both a training and inference loop.

This article walks through that file line by line, explaining the math and showing how the same ideas scale to production language models.

![[Writing/images/EndToEndPipelineAnimation.gif]]

## Roadmap

The file has six parts, and we'll cover them in order:

1. **Dataset** — downloads a list of names to train on.
2. **Tokenization** — converts characters to integers.
3. **Autograd** — a `Value` class that tracks operations and computes gradients.
4. **Model** — a GPT with embeddings, attention, and an MLP, assembled from `Value` operations.
5. **Training** — a loop that predicts the next token, measures error, and updates parameters.
6. **Inference** — a loop that samples new names from the trained model.
## What the model is trying to do

A language model implements one objective: given tokens \(x_1, x_2, \ldots, x_t\), predict \(x_{t+1}\).

microGPT trains on ~32,000 names. Each name is an independent sequence, wrapped with a special BOS (beginning of sequence) token on both sides:

```
[BOS, e, m, m, a, BOS]
```

microGPT uses the same BOS token to mark both the start and end of a sequence. At each position, the model sees all previous tokens and predicts the next one.

Formally, it learns a conditional distribution:

$$P(x_{t+1} \mid x_{\leq t})$$

The model outputs a probability distribution over all 27 possible tokens (26 letters plus BOS). The training objective is to maximize the probability assigned to the true next token across the dataset.

That's the entire game. Everything else — attention, embeddings, normalization — exists to make this conditional distribution expressive enough to model language.

---

## Dataset

```python
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
```

Each line is one name, one name is one document. The list is shuffled so the model doesn't learn the order of the file.

The model doesn't know what a "name" is. It sees sequences of integer tokens and tries to predict the next one. The dataset is the only thing that's specific to names — replace it with any other corpus of short strings and nothing else in the file needs to change.

> In production, models train on trillions of tokens of internet text — web pages, books, code — deduplicated, filtered for quality, and carefully mixed across domains. The training objective is unchanged.

---

## Tokenization

Before a neural network can process text, it must be converted into numbers:

```python
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")
```

This collects every unique character in the dataset (the 26 lowercase letters a–z), sorts them, and assigns each one an integer id by its index. So "a" is 0, "b" is 1, and so on. One additional special token, BOS, gets id 26. That gives a vocabulary of 27 tokens.

The integer values themselves are meaningless. Token 0 is not "less than" token 25. What matters is that each token gets a unique id so the model can distinguish them.

Encoding a name into token ids is just a list lookup:

```python
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
```

The name "emma" becomes `[26, 4, 12, 12, 0, 26]`. Decoding reverses the process: look up each id in the `uchars` list to get back the character.

> In production, tokenizers use **Byte Pair Encoding (BPE)** rather than splitting on individual characters. BPE iteratively merges the most frequent adjacent pair into a new token — "t"+"h" → "th", then "th"+"e" → "the" — until the vocabulary reaches a target size (~100K tokens for GPT-4). Common words become single tokens; rare words split into subword pieces. This matters because the context window is measured in tokens: efficient tokenization means the model sees more content per position and trains faster. The model itself still only receives a sequence of integer ids.

---

## Automatic differentiation: the Value class

Training a neural network means answering one question for every parameter: how does the final loss change when this parameter changes? That rate of change is called a gradient. Computing gradients for all parameters efficiently is called backpropagation, and it relies on the chain rule.

The entire autograd engine is a single class called `Value`:

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad
```

A `Value` wraps a single number and stores four things:

- `data`: the actual number (computed during the forward pass)
- `grad`: the gradient of the loss with respect to this number (filled in during the backward pass, starts at 0)
- `_children`: which Values were used to compute this one
- `_local_grads`: how sensitive this Value is to each of its children

Every operation creates a new Value that records its inputs and local sensitivities, building a directed acyclic graph on the fly. This is a _dynamic_ computation graph — the graph for a 4-letter name has fewer nodes than the graph for an 8-letter name, because the forward pass runs more steps. This is the same approach PyTorch uses ("define-by-run").

![[Writing/images/AutogradGraphAnimation.gif]]

`backward()` walks this graph in reverse topological order, applying the chain rule: at each node, every child accumulates `local_grad × parent.grad`. The topological sort guarantees that by the time we process any node, _all_ nodes that depend on it have already passed their gradient contributions back — so every node is visited exactly once with its gradient fully accumulated.

The accumulation (`+=`) is critical. When a Value feeds multiple operations, its total gradient is the sum of its influence along each path — a direct consequence of the multivariable chain rule.

The `Value` class implements six primitive operations: addition, multiplication, power, log, exp, and ReLU. Every computation in the entire GPT — attention scores, softmax, loss — is built from these six building blocks. The chain rule stitches them together automatically.

> In production, PyTorch replaces `Value` with tensor-based autograd. Every operation on a `torch.Tensor` automatically builds the same computation graph, but each node operates on entire matrices instead of scalars. `loss.backward()` walks the graph the same way — topological sort, reverse pass, chain rule — but dispatches each operation to CUDA kernels running across thousands of GPU cores. Same algorithm, hardware-accelerated.

---
## Parameters

The parameters are the knowledge of the model — floating point numbers wrapped in `Value` that start random and get iteratively adjusted during training.

```python
n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
head_dim = n_embd // n_head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")
```

Each parameter starts as a small random Gaussian. They need to start random — if all weights were identical, every neuron would compute the same thing and learning would stall.

The dimensions chain together: `wte` maps a token id to a 16-dim vector, `attn_wq` maps 16-dim to 16-dim, `mlp_fc1` maps 16-dim to 64-dim, `mlp_fc2` maps 64-dim back to 16-dim, and `lm_head` maps 16-dim to 27 (vocab size). Each matrix's input dimension must match the previous matrix's output dimension.

This totals 4,192 learnable numbers. The same architecture at GPT-4 scale has hundreds of billions — but the operations on each one are identical to what happens here.

> In production, these matrices live on GPUs — massively parallel processors with thousands of cores that each perform the same arithmetic on different data simultaneously. Matrix multiplications are exactly this pattern, which is why a single GPU can handle trillions of operations per second. When the model outgrows one device, matrices are sharded across hundreds of GPUs.

---

## Embeddings: how the model represents tokens

The model can't compute with raw token ids — a neural network has no notion of "token 5 is greater than token 3." Every token needs a fixed-length vector of real numbers it can actually multiply and add:

```python
tok_emb = state_dict['wte'][token_id]
pos_emb = state_dict['wpe'][pos_id]
x = [t + p for t, p in zip(tok_emb, pos_emb)]
```

`wte` is a matrix of shape `[27, 16]` — one row per token. Looking up `token_id` fetches that row: a 16-dimensional vector that is the model's current representation of that token. `wpe` is the same structure for position: a `[16, 16]` matrix. Looking up `pos_id` returns a vector encoding where in the sequence the token sits.

![[Writing/images/TokenEmbeddingsAnimation.gif]]

The two vectors are added element-wise to produce `x`. Concatenation would keep the signals separated but double every subsequent weight matrix. Addition keeps the dimension at 16 throughout, and the model's later layers learn to separate the two signals as needed.

Both tables are required. Without `wpe`, the model cannot distinguish "ab" from "ba." Without `wte`, every position would look identical regardless of which token occupies it.

At initialization, every row is random. Looking up a row is mathematically equivalent to multiplying by a one-hot vector — gradients flow back normally, and only the row that was actually looked up receives a nonzero gradient.

The structure that emerges is entirely a consequence of next-token prediction. Tokens that appear in similar contexts drift toward similar vectors. Nobody encodes this; it falls out of the loss signal.

![[Writing/images/EmbeddingLearningAnimation.gif]]

> In production, models use Rotary Position Embeddings (RoPE). Instead of adding a learned position vector, RoPE rotates the query and key vectors by an angle proportional to their position. The result depends only on relative distance (_i_ − _j_), not absolute positions — so the model generalizes to sequence lengths longer than it was trained on.

---

## Core building blocks

Before the full model, three helper functions.

**Linear layers** multiply an input vector by a weight matrix. Each row produces one output element via a dot product with the input — the fundamental projection from one vector space into another.

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

**Softmax** converts a vector of raw scores (logits) into a probability distribution: all values become positive and sum to 1. Subtracting the max first prevents overflow without changing the result (softmax is shift-invariant).

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

Softmax appears twice: once inside attention (to weight how much each past token contributes) and once at the output (to produce a probability distribution over the next token).

**RMSNorm** rescales a vector so its root-mean-square magnitude is 1. Without normalization, values can grow or shrink exponentially through repeated linear transformations. Normalization keeps activations in a stable range.

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

The original Transformer used LayerNorm, which both centers (subtracts the mean) and scales. RMSNorm drops the centering — the scaling is what matters for stability. Simpler, faster, and now the default.

---
## The GPT forward pass

The `gpt()` function takes a single token and its position, processes it through the network, and returns 27 logits representing how likely each token is to come next.

```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) Multi-head attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits
```

The model maintains a vector `x` that flows through every layer. At each block, the code saves a copy (`x_residual`), transforms `x`, and adds the result back:

```python
x_residual = x
x = <some transformation>
x = [a + b for a, b in zip(x, x_residual)]
```

This is a residual connection. Without it, gradients have to travel back through every layer in sequence, each one multiplying the gradient by a small number. After enough layers the signal disappears — the vanishing gradient problem. The residual addition creates a direct path: the gradient of a sum is just the sum of the gradients, so the error signal reaches early layers cleanly regardless of depth.

Residuals also change what each layer is asked to do. Instead of producing a complete representation from scratch, each layer computes a small correction to the current `x` — iterative refinement from embedding to prediction.

### Attention

The attention block is where tokens communicate. It's the only place in the model where a token at position _t_ can look at tokens at positions 0, 1, ..., _t_−1. This is causal self-attention — each token attends only to previous tokens.

The mechanism works like a soft associative memory. The current token constructs a query ("what context am I looking for?"), every previous token advertises a key ("here's what I contain"), and instead of a single exact match, the lookup returns a weighted blend of all values, with weights proportional to how well each key matches the query.

![[Writing/images/AttentionWeightsAnimation.gif]]

An RNN compresses everything it has seen into a single fixed-size hidden state — a bottleneck that loses information over distance. Attention sidesteps this: every token has direct access to every previous token. The cost is quadratic (\(n^2\) dot products for sequence length \(n\)), but the model can relate any two tokens regardless of distance.

At each position, three learned projections produce the query, key, and value:

```python
q = linear(x, state_dict[f'layer{li}.attn_wq'])
k = linear(x, state_dict[f'layer{li}.attn_wk'])
v = linear(x, state_dict[f'layer{li}.attn_wv'])
```

Keys and values are appended to the KV cache so future positions can attend to them:

```python
keys[li].append(k)
values[li].append(v)
```

The attention computation for a single head:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)V$$

The dot product \(QK^T\) measures similarity between the query and each key. Scaling by \(\sqrt{d}\) is essential: without it, the variance of the dot product grows linearly with \(d\), producing large scores that push softmax into saturation — nearly all probability mass lands on one key, gradients vanish, and learning stalls. Dividing by \(\sqrt{d}\) normalizes variance back to 1, keeping softmax in its sensitive regime.

The model runs four heads in parallel, each operating on a 4-dimensional slice of the 16-dimensional q, k, and v vectors:

```python
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]
```

Because the subspaces are separate, each head can specialize on a different relationship — one might learn vowel patterns, another positional proximity. The four outputs are concatenated back into a 16-dimensional vector and mixed by a final linear layer (`attn_wo`).

The KV cache serves different purposes in training and inference. During training, cached keys and values are part of the computation graph — gradients flow through them, connecting every position into one differentiable chain. During inference, no gradients are needed; the cache becomes a store of precomputed vectors, so generating token _t_ costs one forward pass, not _t_.

> In production, the attention equation is unchanged, but FlashAttention computes it without materializing the full N×N score matrix — tiling the work into GPU SRAM blocks for O(N) memory and 2-4× speedup — and Grouped Query Attention shares K/V projections across heads to shrink the KV cache by 4-8×.

After attention lets tokens communicate, the MLP processes each position independently:

```python
x_residual = x
x = rmsnorm(x)
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
x = [xi.relu() for xi in x]
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
x = [a + b for a, b in zip(x, x_residual)]
```

The MLP projects the 16-dimensional vector up to 64 dimensions, applies ReLU, and projects back down to 16. Two linear layers with no activation between them collapse into one:

$$W_2(W_1 x) = (W_2 W_1)x$$

ReLU breaks this linearity. With it, the MLP can approximate arbitrary functions; without it, depth adds no expressive power. The expansion to 4× gives the network a wider space to work in temporarily — ReLU zeroes out roughly half the dimensions, and the model learns which to activate for different inputs.

After all layers, the final representation is projected to vocabulary size:

```python
logits = linear(x, state_dict['lm_head'])
```

This produces 27 logits — one per token — converted to probabilities by softmax during training and inference.

---

## Training

The training loop implements the objective described at the start: increase the probability the model assigns to the correct next token.

```python
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

num_steps = 1000
for step in range(num_steps):

    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")
```

At each position, the model produces probabilities over 27 tokens. The loss is the cross-entropy — the negative log of the probability assigned to the correct next token:

```python
loss_t = -probs[target_id].log()
```

The loss is averaged across all positions. A single call to `loss.backward()` propagates gradients through the entire computation graph, all the way back to every parameter.

Vanilla gradient descent applies `p.data -= lr * p.grad` — the same learning rate multiplied by the same raw gradient for every parameter. This has two problems. First, each gradient is computed from a single document, so it's noisy — one step might point left, the next right, and progress zigzags. Second, a single learning rate can't be right for all parameters at once: too large and the loss explodes; too small and training stalls. In practice, different parameters need different step sizes because their gradients vary by orders of magnitude.

Adam fixes both problems:

```python
m[i] = beta1 * m[i] + (1 - beta1) * p.grad       # momentum
v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2   # squared gradient
m_hat = m[i] / (1 - beta1 ** (step + 1))           # bias correction
v_hat = v[i] / (1 - beta2 ** (step + 1))           # bias correction
p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam) # update
```

Adam maintains two running averages per parameter:

**First moment** (momentum): \(m_t = \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t\)

An exponential moving average of the gradient \(g_t\). Each step sees only one document, so individual gradients are noisy. Momentum smooths this by blending the current gradient with accumulated history. \(\beta_1 = 0.85\) means 85% of the previous average is retained and 15% comes from the new gradient.

**Second moment** (adaptive scaling): \(v_t = \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^2\)

An exponential moving average of the squared gradient. This tracks the *scale* of recent gradients for each parameter. \(\beta_2 = 0.99\) gives a long memory — the estimate stays stable over many steps.

**Bias correction**: \(\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}\)

Both averages are initialized to zero, so early values underestimate the true mean. At step 1, raw momentum is only \(0.15 \times g_1\). Dividing by \((1 - 0.85^1) = 0.15\) corrects this. As \(t\) grows, \(\beta^t \to 0\) and the correction vanishes.

**Update**: \(\theta_t = \theta_{t-1} - \alpha_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\)

The division by \(\sqrt{\hat{v}_t}\) is what makes Adam adaptive: parameters with consistently large gradients get smaller effective steps, while parameters with small gradients get larger ones. This matters because different parts of the network operate at very different scales — embedding rows for rare tokens receive infrequent but large gradients, while weights deep in the network get steady but small ones. Adam normalizes for this automatically.

The learning rate \(\alpha_t\) decays linearly from 0.01 to 0. Large early steps make fast progress; small late steps let the model settle precisely. After updating, `p.grad = 0` resets for the next step.

---

## Inference: generating new names

Once training is done, the parameters are frozen and we generate by sampling from the model's predictions:

```python
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

Start with BOS, run the model, sample a token from the output distribution, feed it back in, repeat until the model produces BOS again. The KV cache grows with each step — previous keys and values are reused, so each new token only requires one forward pass.

Before softmax, the logits are divided by a temperature parameter:

```python
probs = softmax([l / temperature for l in logits])
```

**Low temperature** (e.g. 0.5) sharpens the distribution — the model sticks to its top predictions. At the limit of temperature → 0, it always picks the most likely token (greedy decoding). **High temperature** (e.g. 2.0) flattens the distribution — the model samples more diversely, sometimes picking unlikely tokens.

The model never plans ahead. It doesn't think "I'm going to spell 'anna'" — it picks each letter based on what came before. Coherent names emerge from many local, probabilistic decisions. The result is the output we previewed at the start — names like "kamon," "karai," and "anna" that aren't in the dataset but sound like they could be.

---

## Closing

The core algorithm of a large language model fits in 200 lines of Python. A dataset of sequences. A tokenizer to convert text to integers. An autograd engine to compute gradients. A Transformer that predicts the next token. An optimizer that adjusts parameters to make those predictions better. A sampling loop that generates new sequences.

Everything else — the GPU kernels, the distributed training, the post-training pipeline, the serving infrastructure — is about making this same algorithm work at scale.

If you understand these 200 lines, you understand the mechanism behind every conversation you've had with a language model.
