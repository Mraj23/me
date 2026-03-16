---
title: GPT, Explained
---
Andrej Karpathy recently published [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), 200 lines of Python that implement a GPT from scratch with zero dependencies. No PyTorch, no NumPy, nothing to install. Those 200 lines contain everything: dataset loading, tokenization, an autograd engine, a GPT-2-style neural network, the Adam optimizer, and both a training and inference loop. It is the simplest complete implementation of the algorithm that trains a language model to learn sequences of characters.

This article walks through that implementation line by line, explains the math, and shows how the same ideas scale to production language models.

The code has six parts. We'll cover them in the order they appear in the file, which is not always the order that's easiest to understand. So here's the full picture before we start.

The model reads names as sequences of characters, guesses the next character at each step, checks how wrong it was, and adjusts its numbers to be less wrong. Repeat a thousand times. To do this, the code needs (1) data to learn from, (2) a way to turn characters into numbers, (3) an engine that figures out which numbers to adjust and by how much (this is the autograd engine, and it's the most abstract part), (4) the actual numbers to learn (parameters), (5) a model that transforms input into predictions by multiplying through those parameters, (6) a training loop that drives the whole cycle, and (7) a sampling loop that generates new names from the trained model.

1. **Dataset**: downloads a list of names to train on.
2. **Tokenization**: converts characters to integers.
3. **Autograd**: a `Value` class that tracks operations and computes gradients.
4. **Model**: a GPT with embeddings, attention, and an MLP, assembled from `Value` operations.
5. **Training**: a loop that predicts the next token, measures error, and updates parameters.
6. **Inference**: a loop that samples new names from the trained model.

As a preview, here's what the trained model generates. None of these names are in the dataset:

```
sample  1: kamon
sample  2: ann
sample  3: karai
sample  4: jaire
sample  5: vialan
sample  6: karia
sample  7: yeran
sample  8: anna
sample  9: areli
sample 10: kaina
```

## What the model is trying to do

A language model implements one objective: given tokens x₁, x₂, ..., xₜ, predict xₜ₊₁.

microGPT trains on ~32,000 names. Each name is an independent sequence, wrapped with a special BOS (beginning of sequence) token on both sides:

```
[BOS, e, m, m, a, BOS]
```

microGPT uses the same BOS token to mark both the start and end of a sequence. At each position, the model sees all previous tokens and only predicts the next one.

Formally, it learns a conditional distribution:

$$P(x_{t+1} \mid x_{\leq t})$$

The model outputs a probability distribution over all 27 possible tokens (26 letters of the alphabet plus the BOS token). The training objective is to maximize the probability assigned to the true next token across the dataset.

![[Writing/images/EndToEndPipelineAnimation.gif]]

That's the entire game. Every component in the code exists to make this prediction more accurate.

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

The code reads a file containing 32,000 names separated by newlines. Each line is one name, one name is one document. The list is shuffled so the model doesn't learn the order of the file.

The model doesn't know what a "name" is. It sees sequences of characters converted to integer tokens and tries to predict the next one. The dataset is the only thing specific to names. Replace it with any other corpus of short strings and nothing else in the file needs to change.

> In production, models train on trillions of tokens of internet text (web pages, books, code), deduplicated, filtered for quality, and carefully mixed across domains. The training objective is unchanged.

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

> Using each letter as a token is the simplest possible tokenizer. In production, tokenizers use **Byte Pair Encoding (BPE)** rather than splitting on individual characters. BPE iteratively merges the most frequent adjacent pair into a new token: "t"+"h" becomes "th", then "th"+"e" becomes "the", and so on until the vocabulary reaches a target size (~100K tokens for GPT-4). Common words become single tokens. Rare words split into subword pieces. The model itself still only receives a sequence of integer ids.

---

We now have sequences of integers. The next section builds the engine that makes learning possible. The model will process these integers through thousands of additions and multiplications to produce a prediction. When that prediction is wrong, we need to trace backward through every operation and figure out how to adjust each number to make the prediction less wrong. The `Value` class does this automatically. It's the most abstract part of the code. If you follow one idea from this section, follow this: every operation records its inputs, so we can walk backward and compute how much each input contributed to the error.

## Automatic differentiation: the Value class

Training a neural network means answering one question for every parameter: how does the final loss change when this parameter changes? That rate of change is called a gradient. Computing gradients for all parameters efficiently is called backpropagation. Backpropagation relies on the chain rule from calculus.

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

Every operation creates a new Value that records its inputs and local sensitivities, building a graph on the fly. This is a _dynamic_ computation graph. The graph for a 4-letter name has fewer nodes than the graph for an 8-letter name, because the forward pass runs more steps. This is the same approach PyTorch uses ("define-by-run").

The `Value` class implements six primitive operations. Every computation the model will do is built from these six building blocks:

| Operation | Forward | Local gradients |
|-----------|---------|-----------------|
| `a + b` | \(a + b\) | \(\frac{\partial}{\partial a} = 1, \quad \frac{\partial}{\partial b} = 1\) |
| `a * b` | \(a \cdot b\) | \(\frac{\partial}{\partial a} = b, \quad \frac{\partial}{\partial b} = a\) |
| `a ** n` | \(a^n\) | \(\frac{\partial}{\partial a} = n \cdot a^{n-1}\) |
| `log(a)` | \(\ln(a)\) | \(\frac{\partial}{\partial a} = \frac{1}{a}\) |
| `exp(a)` | \(e^a\) | \(\frac{\partial}{\partial a} = e^a\) |
| `relu(a)` | \(\max(0, a)\) | \(\frac{\partial}{\partial a} = \mathbf{1}_{a > 0}\) |

Here's a concrete example. Note that `a` is used twice (the graph branches), so its gradient is the sum of both paths:

```python
a = Value(2.0)
b = Value(3.0)
c = a * b       # c = 6.0
L = c + a       # L = 8.0
L.backward()
print(a.grad)   # 4.0 (dL/da = b + 1 = 3 + 1, via both paths)
print(b.grad)   # 2.0 (dL/db = a = 2)
```

What do these numbers mean? `a.grad = 4.0` tells us that if we increase `a` by a tiny amount (say 0.001), `L` would increase by about 4x that (0.004). `b.grad = 2.0` means the same nudge to `b` would increase `L` by about 0.002. These gradients tell us the direction (positive or negative) and steepness (the magnitude) of each input's influence on the output.

![[Writing/images/AutogradGraphAnimation.gif]]

`backward()` walks this graph in reverse topological order to answer one question for every parameter: how much did this value contribute to the final loss?

The chain rule is what makes this possible. If A influences B, and B influences C, then A's influence on C is the product of those two local influences. You don't need to know the whole path from A to C at once. You can multiply one hop at a time. `backward()` uses this: starting from the loss (gradient = 1), it works backwards through the graph. At each node it multiplies the node's local sensitivity by the gradient that arrived from downstream. By the time it reaches a parameter, the accumulated product encodes exactly how much that parameter nudged the final loss.

The topological sort guarantees correctness. By the time `backward()` processes any node, every node downstream of it has already contributed its gradient back, so the arriving gradient is complete. The accumulation (`+=`) handles the case where one value feeds multiple operations.

When the model later processes the name "emma", it will create thousands of Value nodes, one for every addition and multiplication in the forward pass. `backward()` will walk all of them to figure out how to improve.

> In production, PyTorch replaces `Value` with tensor-based autograd. Every operation on a `torch.Tensor` automatically builds the same computation graph, but each node operates on entire matrices instead of scalars. `loss.backward()` walks the graph the same way (topological sort, reverse pass, chain rule) but dispatches each operation to CUDA kernels running across thousands of GPU cores. Same algorithm, hardware-accelerated.

---

The autograd engine can track any computation built from `Value` objects. But we haven't created any Values yet. The next section initializes the actual numbers the model will learn: thousands of them, all starting random, all wrapped in `Value` so that gradients can flow through them.

## Parameters

The parameters are the knowledge of the model. They are floating-point numbers wrapped in `Value` that start random and get adjusted during training.

```python
n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
head_dim = n_embd // n_head
```

These five lines are architecture hyperparameters. They define the shape of the network, not anything it learns. `n_embd = 16` means each token is represented by a vector of 16 numbers. `n_head = 4` means the attention mechanism runs four parallel lookups. `n_layer = 1` means there's one processing block. `block_size = 16` means the model can look at up to 16 previous tokens.

```python
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

Every parameter is a `Value` initialized to a small random Gaussian. They must start random. If all weights were identical, every neuron would compute the same thing and learning would stall. The `matrix` helper creates a 2D grid of these `Value` objects, and `state_dict` collects every weight matrix the model uses into a single dictionary.

Each name in `state_dict` corresponds to a specific part of the model. `wte` stores the token embeddings we'll see next. `wpe` stores position embeddings. `attn_wq`, `attn_wk`, `attn_wv` are the attention weight matrices. `mlp_fc1` and `mlp_fc2` are the feedforward network weights. `lm_head` is the final output projection. Each one is a matrix of learnable numbers that the model multiplies inputs through during the forward pass. This is how neural networks transform data: a series of matrix multiplications, mixed with nonlinear functions like ReLU, where the matrices *are* the parameters. Learning is the process of adjusting these matrices so that the chain of multiplications produces the right output.

For example, the embedding for token 4 ("e") is the 5th row of `wte`: 16 random numbers right now. After training, those 16 numbers will encode something meaningful about how "e" behaves in names.

This model has 4,192 parameters. GPT-4 has hundreds of billions, and every forward pass multiplies through all of them. At that scale, training requires trillions of matrix multiplications. This is why large models train on GPUs: massively parallel processors with thousands of cores that perform the same arithmetic on different data simultaneously. Matrix multiplication fits this pattern exactly. A single GPU can handle trillions of operations per second. When the model outgrows one device, its matrices are split across hundreds of GPUs.

---

We have thousands of random numbers organized into matrices. They're meaningless right now. The next section defines what to *do* with them: a function that takes a token, multiplies it through these matrices in a specific pattern, and outputs a prediction. This is the model architecture.

## Embeddings: how the model represents tokens

The model can't compute with raw token ids. A neural network has no notion of "token 5 is greater than token 3." Every token needs a fixed-length vector of real numbers it can actually multiply and add:

```python
tok_emb = state_dict['wte'][token_id]
pos_emb = state_dict['wpe'][pos_id]
x = [t + p for t, p in zip(tok_emb, pos_emb)]
```

`wte` is a matrix of shape `[27, 16]`, one row per token. Looking up `token_id` fetches that row: a 16-dimensional vector that is the model's current representation of that token. `wpe` is the same structure for position: a `[16, 16]` matrix. Looking up `pos_id` returns a vector encoding where in the sequence the token sits.

![[Writing/images/TokenEmbeddingsAnimation.gif]]

The two vectors are added element-wise to produce `x`. Addition keeps the dimension at 16 throughout, and the model's later layers learn to separate the two signals as needed.

Both tables are required. Without `wpe`, the model cannot distinguish "ab" from "ba." Without `wte`, every position would look identical regardless of which token occupies it.

![[Writing/images/PositionalEncodingAnimation.gif]]

At initialization, every row in `wte` is random. The model has no idea which tokens are similar. But each row is a `Value`, so gradients flow through it during training. When the model looks up token 4 ("e"), that row becomes part of the forward pass. If the model's prediction is wrong, `backward()` computes a gradient for every `Value` in the computation graph, including the 16 numbers in token 4's embedding row. The gradient tells that row which direction to shift so the prediction improves next time. Only the row that was actually looked up receives a gradient; the other 26 rows are untouched.

Over thousands of training steps, tokens that appear in similar contexts receive similar gradient signals and drift toward similar regions of the 16-dimensional space. After training, "a" and "e" end up close together because they appear in similar positions within names. Tokens that behave differently drift apart. Nobody encodes this structure; it emerges entirely from the loss signal of next-token prediction.

![[Writing/images/EmbeddingLearningAnimation.gif]]

> In production, models use Rotary Position Embeddings (RoPE). Instead of adding a learned position vector, RoPE rotates the query and key vectors by an angle proportional to their position. The result depends only on relative distance (_i_ − _j_), not absolute positions, so the model generalizes to sequence lengths longer than it was trained on.

---

## Core building blocks

The model transforms vectors through three operations that appear throughout the forward pass. **Linear** projects a vector into a new space. This is how the model applies its learned parameters. **Softmax** converts raw scores into probabilities. This is how the model makes decisions. **RMSNorm** stabilizes values that would otherwise grow or shrink out of control.

**Linear** multiplies an input vector by a weight matrix. Each row of the weight matrix produces one output element by taking a dot product with the input. This is a way to transform one vector into another.

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

**Softmax** converts a vector of raw scores (logits) into a probability distribution: all values become positive and sum to 1. Subtracting the max first prevents overflow without changing the result.

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

---
## The GPT forward pass

The `gpt()` function takes a single token and its position, processes it through the network, and returns 27 logits representing how likely each token is to come next.

The model maintains a vector `x` that flows through every layer. Think of it as a running summary that gets refined at each step.

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

Let's trace what happens to the first real token of "emma." Token 4 ("e") at position 1 enters the model. It looks up row 4 of `wte` and row 1 of `wpe`, adds them, normalizes the result, and passes it into the attention block. This is the vector `x` that flows through every layer.

At each block, the code saves a copy (`x_residual`), transforms `x`, and adds the result back:

```python
x_residual = x
x = <some transformation>
x = [a + b for a, b in zip(x, x_residual)]
```

This is a residual connection. Without it, gradients have to travel back through every layer in sequence, each one multiplying the gradient by a small number. After enough layers the signal disappears. This is the vanishing gradient problem. The residual addition creates a direct path: the gradient of a sum is just the sum of the gradients, so even if a layer's transformation shrinks the gradient, the residual path carries the original gradient through unchanged.

Residuals also change what each layer is asked to do. Instead of producing a complete representation from scratch, each layer computes a small correction to the current `x`. Each layer refines the representation rather than rebuilding it.

### Attention

The attention block is where tokens communicate. It's the only place in the model where a token at position _t_ can look at tokens at positions 0, 1, ..., _t_−1. This is causal self-attention: each token attends only to previous tokens.

The current token constructs a query ("what am I looking for?"). Every previous token advertises a key ("here's what I contain"). Instead of a single exact match, the lookup returns a weighted blend of all values, with weights proportional to how well each key matches the query. It's a soft lookup over the entire history.

![[Writing/images/AttentionWeightsAnimation.gif]]

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

The dot product \(QK^T\) measures similarity between the query and each key. Scaling by \(\sqrt{d}\) is important. Without it, the variance of the dot product grows linearly with \(d\), producing large scores that push softmax into saturation. Nearly all probability mass lands on one key, gradients vanish, and learning stalls. Dividing by \(\sqrt{d}\) normalizes variance back to 1, keeping softmax in its sensitive range.

The model runs four heads in parallel, each operating on a 4-dimensional slice of the 16-dimensional q, k, and v vectors:

```python
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]
```

Because the subspaces are separate, each head can specialize on a different relationship. The four outputs are concatenated back into a 16-dimensional vector and mixed by a final linear layer (`attn_wo`).

The KV cache serves different purposes in training and inference. During training, cached keys and values are part of the computation graph. Gradients flow through them, connecting every position into one differentiable chain. During inference, no gradients are needed. The cache becomes a store of precomputed vectors, so generating token _t_ costs one forward pass instead of _t_.

> In production, the attention equation is unchanged, but FlashAttention computes it without materializing the full N×N score matrix. It tiles the work into GPU SRAM blocks for O(N) memory and 2-4x speedup. Grouped Query Attention shares K/V projections across heads to shrink the KV cache by 4-8x.

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

ReLU breaks this linearity. With it, the MLP can learn complex, nonlinear transformations. Without it, depth adds no expressive power. The expansion to 4x gives the network a wider space to work in temporarily. ReLU zeroes out roughly half the dimensions, and the model learns which to activate for different inputs.

After all layers, the final representation is projected to vocabulary size:

```python
logits = linear(x, state_dict['lm_head'])
```

This produces 27 logits, one per token, converted to probabilities by softmax during training and inference.

---

The model can now produce predictions, but every parameter is still random. The training loop is where learning happens: feed in a name, get a prediction, measure the error, call `backward()` to compute gradients, and adjust every parameter to reduce the error.

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

At each position, the model produces probabilities over 27 tokens. The loss is the cross-entropy, the negative log of the probability assigned to the correct next token:

```python
loss_t = -probs[target_id].log()
```

For example, suppose the model is processing "emma" and has just seen [BOS, e, m] and needs to predict the second "m." If the model assigns probability 0.04 to "m" (barely above the random baseline of 1/27 ≈ 0.037), the loss for this position is −log(0.04) = 3.22. If it assigns probability 0.9, the loss drops to −log(0.9) = 0.11. The loss is high when the model is surprised by the correct answer, and low when it predicted correctly.

Why use −log(p) instead of something simpler like squared error (1−p)²? When the model assigns probability 0.01 to the correct token, squared error gives (1−0.01)² = 0.98. Cross-entropy gives −log(0.01) = 4.6. Cross-entropy punishes confident wrong predictions much more harshly, which drives faster learning. It also has a clean gradient: ∂(−log p)/∂p = −1/p. When the model is very wrong (p near 0), the gradient is huge, forcing a large update. Squared error's gradient doesn't have this property.

At initialization the loss starts around 3.3, which is −log(1/27): the model is guessing uniformly at random among 27 tokens. After 1,000 training steps, the loss drops to about 2.37. Lower is better, and the lowest possible is 0 (perfect predictions). A loss of 2.37 means the model has narrowed its effective choices from 27 tokens down to about \(e^{2.37} \approx 10.7\) tokens per position. It's clearly learning the statistical patterns of names.

The loss is averaged across all positions. A single call to `loss.backward()` propagates gradients through the entire computation graph, all the way back to every parameter.

Basic gradient descent applies `p.data -= lr * p.grad`, the same learning rate multiplied by the same raw gradient for every parameter. This has two problems. First, each gradient is computed from a single document, so it's noisy. One step might point left, the next right, and progress zigzags. Second, a single learning rate can't be right for all parameters at once. Too large and the loss explodes. Too small and training stalls. Different parameters need different step sizes because their gradients vary by orders of magnitude.

Adam fixes both problems:

```python
m[i] = beta1 * m[i] + (1 - beta1) * p.grad       # momentum
v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2   # squared gradient
m_hat = m[i] / (1 - beta1 ** (step + 1))           # bias correction
v_hat = v[i] / (1 - beta2 ** (step + 1))           # bias correction
p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam) # update
```

The core problem is that gradient descent uses one learning rate \(\alpha\) for all parameters. On a loss surface where different parameters have very different curvatures, a learning rate that's good for one parameter is terrible for another. Too large and steep dimensions overshoot. Too small and shallow dimensions crawl. Adam fixes this with two separate mechanisms.

**First moment (momentum):** \(m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t\)

Momentum changes the *direction* of each step. It's an exponential moving average of past gradients. Consider a parameter whose gradients over 5 steps are \(+10, -8, +9, -7, +10\). The raw gradients oscillate wildly, but the running average stays small. The positive and negative values partially cancel, dampening oscillation in steep dimensions. Now consider consistent gradients: \(+2, +2, +2, +2, +2\). The running average steadily builds up, accelerating progress in that direction. Momentum dampens oscillation where gradients flip sign, and accelerates where they don't.

**Second moment (adaptive scaling):** \(v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2\)

This changes the *step size* per parameter based on the history of gradient magnitudes. It's a different mechanism from momentum. Momentum cares about signed direction (positives and negatives cancel). Adaptive scaling cares about magnitude (squares are always positive, so large gradients accumulate regardless of sign). The effective learning rate per parameter becomes \(\alpha / \sqrt{v_t}\). A parameter with gradients of \(\pm 10\) accumulates large \(v\) and gets small steps. A parameter with gradients of \(\pm 2\) accumulates small \(v\) and gets large steps.

**Update:** \(\theta_t = \theta_{t-1} - \dfrac{\alpha}{\sqrt{\hat{v}_t}} \cdot \hat{m}_t\)

Adam combines both. \(\hat{m}_t\) provides the smoothed direction ("which way?") and \(\alpha / \sqrt{\hat{v}_t}\) sets the per-parameter step size ("how far?"). The \(\hat{\phantom{x}}\) denotes bias correction. Both running averages start at zero, so early estimates are too low. Dividing by \((1 - \beta^t)\) corrects this:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

At step 1, raw momentum is only \((1 - \beta_1) \cdot g_1\); dividing by \((1 - \beta_1)\) corrects it back to \(g_1\). As \(t\) grows, \(\beta^t \to 0\) and the correction vanishes.

![[Writing/images/AdamOptimizerAnimation.gif]]

The animation shows these ideas on a 2D loss surface with elongated elliptical contours, steep vertically and shallow horizontally. Vanilla gradient descent zigzags, oscillating in the steep direction while barely progressing in the shallow one. Momentum and adaptive rates are shown separately first, then combined into Adam and raced against vanilla GD. Adam reaches the minimum first.

The learning rate \(\alpha_t\) decays linearly from 0.01 to 0. Large early steps make fast progress; small late steps let the model settle precisely. After updating, `p.grad = 0` resets every gradient to zero. Without this, gradients from the previous step would add to the next step's gradients, corrupting the update.

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

Start with BOS, run the model, sample a token from the output distribution, feed it back in, repeat until the model produces BOS again. The KV cache grows with each step. Previous keys and values are reused, so each new token requires only one forward pass.

There's one parameter that controls the character of the output. Before softmax, the logits are divided by a temperature:

```python
probs = softmax([l / temperature for l in logits])
```

**Low temperature** (e.g. 0.5) sharpens the distribution. The model sticks to its top predictions. At the limit of temperature → 0, it always picks the most likely token (greedy decoding). **High temperature** (e.g. 2.0) flattens the distribution. The model samples more diversely, sometimes picking unlikely tokens.

The model never plans ahead. It doesn't think "I'm going to spell 'anna'." It picks each letter based on what came before. Coherent names emerge from many local, probabilistic decisions. The result is names like "kamon," "karai," and "anna" that aren't in the dataset but sound like they could be.

---

## Closing

The core algorithm of a large language model fits in 200 lines of Python. A dataset of sequences. A tokenizer to convert text to integers. An autograd engine to compute gradients. A Transformer that predicts the next token. An optimizer that adjusts parameters to make those predictions better. A sampling loop that generates new sequences.

Everything else (the GPU kernels, the distributed training, the post-training pipeline, the serving infrastructure) is about making this same algorithm work at scale.

When you talk to ChatGPT, the system prompt, your message, and its reply are all just tokens in a sequence. The model is completing the document one token at a time, same as microGPT completing a name. The scale is different. The algorithm is the same.

If you understand these 200 lines, you understand the mechanism behind every conversation you've had with a language model.
