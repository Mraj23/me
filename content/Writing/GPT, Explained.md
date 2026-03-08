---
title: GPT, Explained
---
Andrej Karpathy recently published [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — a single file of 200 lines of pure Python, with no dependencies, that trains and runs a GPT. No PyTorch, no NumPy, nothing to install. The file contains a dataset, a tokenizer, an autograd engine, a GPT-2-style neural network, the Adam optimizer, a training loop, and an inference loop.

This article walks through that file, line by line, with deeper explanations of the math, worked examples where the code alone isn't enough, and connections to how the same ideas appear in production language models like ChatGPT.

By the end you'll understand every line of the code and also know what changes between this toy model and a real one.

---

## What the model is trying to do

Before we look at any code, let's be clear about the goal.

A language model takes a sequence of tokens and predicts what comes next. That's it. Every component in this file — the embeddings, the attention mechanism, the optimizer — exists to serve this single objective.

The training data is a list of around 32,000 names: emma, olivia, ava, isabella, and so on. Each name is treated as an independent sequence. The model's job is to learn the statistical patterns in these sequences so it can generate new, plausible-sounding names.

Here's what that looks like concretely. The name "emma" gets wrapped with a special beginning-of-sequence token on both sides:

```
[BOS, e, m, m, a, BOS]
```

At each position, the model sees everything to the left and predicts the next token:

```
Given [BOS]           → predict "e"
Given [BOS, e]        → predict "m"
Given [BOS, e, m]     → predict "m"
Given [BOS, e, m, m]  → predict "a"
Given [BOS, e, m, m, a] → predict BOS (end of name)
```

The model outputs a probability distribution over all 27 possible tokens (26 letters plus BOS). A good model assigns high probability to the correct next token. A bad model spreads probability evenly or puts it on the wrong token.

That's the entire training objective: make the correct next token more probable. Everything else follows.

> **🎬 Animation: the prediction task.** Show tokens appearing one at a time, left to right. At each position, a bar chart of 27 probabilities appears. The correct next token's bar is highlighted. When the bar is tall, a small loss number appears. When the bar is short, a large loss number appears. This establishes the full training objective visually before any code.

---

## Dataset

The dataset code downloads a file of names and loads them into a list:

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

This is toyish, but the model doesn't know that. It doesn't know what a "name" is. It sees sequences of integer tokens and tries to predict the next one. Swap in a file of city names, Pokémon names, or English words and the rest of the code doesn't change at all.

In production language models, the same logic applies to trillions of tokens of internet text — web pages, books, code. The data is deduplicated, filtered for quality, and carefully mixed across domains. The training objective is unchanged.

---

## Tokenization

Before a neural network can process text, it must be converted into numbers. The tokenizer here is as simple as possible:

```python
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")
```

This collects every unique character in the dataset (the 26 lowercase letters a–z), sorts them, and assigns each one an integer id by its index. So "a" is 0, "b" is 1, and so on. One additional special token, BOS (beginning of sequence), gets id 26. That gives a vocabulary of 27 tokens.

The integer values themselves are meaningless. Token 0 is not "less than" token 25. They might as well be different-colored marbles. What matters is that each token gets a unique id so the model can distinguish them.

Encoding a name into token ids is just a list lookup:

```python
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
```

The name "emma" becomes `[26, 4, 12, 12, 0, 26]`. Decoding reverses the process: look up each id in the `uchars` list to get back the character.

**Real tokenizers** are more sophisticated. Production models use algorithms like BPE (Byte Pair Encoding), which learn to merge frequently co-occurring characters into single tokens. The algorithm starts with individual characters, finds the most common adjacent pair in the training data, merges that pair into a new token, and repeats thousands of times. This builds a vocabulary of common subwords — "the" becomes one token, "ing" becomes one token, and rare words get broken into pieces. GPT-4's tokenizer has roughly 100,000 tokens. This is much more efficient because the model sees more content per position. But the model itself still only ever sees integer ids.

---

## Automatic differentiation: the Value class

This is the mathematical heart of the file and where we'll spend the most time.

Training a neural network requires knowing, for every parameter in the model: "if I nudge this number up a tiny amount, does the loss go up or down, and by how much?" That rate of change is called a gradient. Computing gradients for all parameters efficiently is called backpropagation, and it relies on the chain rule from calculus.

In production, libraries like PyTorch handle this automatically. They track every operation you perform on tensors (multi-dimensional arrays of numbers), build a computation graph behind the scenes, and compute all gradients in one call to `.backward()`. They also run these operations on GPUs, which can process billions of numbers in parallel. microGPT reimplements the same idea from scratch, but operating on individual scalar numbers instead of tensors. The algorithm is identical — just much slower.

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

Let's break this down.

### What a Value is

A `Value` wraps a single number and stores four things:

- `data`: the actual number (computed during the forward pass)
- `grad`: the gradient of the loss with respect to this number (filled in during the backward pass, starts at 0)
- `_children`: which Values were used to compute this one
- `_local_grads`: how sensitive this Value is to each of its children

Think of each `Value` as a node in a graph. The edges point backward to the inputs that produced it.

### The forward pass builds a graph

Every time you do arithmetic with Value objects, a new Value is created that remembers where it came from. Let's trace through a concrete example:

```python
a = Value(2.0)
b = Value(3.0)
c = a * b       # c.data = 6.0
L = c + a       # L.data = 8.0
```

After these four lines, we have four Value nodes connected like this:

```
a (2.0) ──┬──→ [*] ──→ c (6.0) ──┬──→ [+] ──→ L (8.0)
b (3.0) ──┘                       │
a (2.0) ──────────────────────────┘
```

Notice that `a` is used twice — once in the multiplication and once in the addition. The graph branches. This matters for gradients.

Each operation also records its local gradients — the partial derivative of its output with respect to each input. For multiplication, `c = a * b`, the local gradient with respect to `a` is `b.data = 3.0`, and with respect to `b` is `a.data = 2.0`. For addition, `L = c + a`, the local gradient with respect to both inputs is just `1`.

> **🎬 Animation: computation graph construction.** Nodes appear as circles labeled with their values. As each line of code executes, new nodes and edges appear. First `a` and `b` are isolated. Then `a * b` creates `c` with two edges pointing back. Then `c + a` creates `L` with two edges — one back to `c`, one directly to `a`. Highlight that `a` now has two outgoing paths to `L`.

### The backward pass computes gradients

Once we have the graph, `L.backward()` computes gradients for every node. It does this in two phases.

First, it sorts the graph in topological order — an ordering where every node comes after all the nodes it depends on. Then it walks this order in reverse, starting from `L` and working back to the inputs.

At `L`, we set `grad = 1`. This is the starting point: the loss's rate of change with respect to itself is 1.

Then at each node, we apply the chain rule: for each child, we add `local_grad × parent.grad` to the child's gradient. Let's trace through our example:

**Step 1: Start at L (grad = 1)**

L was computed as `c + a`. The local gradient for both `c` and `a` is 1.

- `c.grad += 1 × 1 = 1`
- `a.grad += 1 × 1 = 1`

**Step 2: Process c (grad = 1)**

c was computed as `a * b`. The local gradient for `a` is `b.data = 3.0`, and for `b` is `a.data = 2.0`.

- `a.grad += 3.0 × 1 = 3.0` (now a.grad = 1 + 3 = 4.0)
- `b.grad += 2.0 × 1 = 2.0`

**Result:** `a.grad = 4.0`, `b.grad = 2.0`

This tells us: if we increase `a` by a tiny amount ε, `L` increases by about 4ε. If we increase `b` by ε, `L` increases by about 2ε.

Notice how `a.grad` accumulated contributions from two paths — the direct path through the addition (contributing 1.0) and the path through the multiplication (contributing 3.0). The `+=` is critical. When a value is used in multiple places, gradients from all paths must be summed. This is a consequence of the multivariable chain rule.

We can verify with PyTorch:

```python
import torch
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = a * b
L = c + a
L.backward()
print(a.grad)   # tensor(4.)
print(b.grad)   # tensor(2.)
```

Identical. microGPT's `Value` class does exactly what PyTorch does — the same algorithm, just on scalars instead of tensors.

> **🎬 Animation: backpropagation.** Same graph as before. A gradient signal (colored pulse) starts at L with value 1. It flows backward along each edge, being multiplied by the local gradient at each step. At node `a`, two pulses arrive from the two paths — one carrying 1.0, one carrying 3.0 — and they sum to 4.0. End state: every node is labeled with its gradient.

### The local gradient rules

Here are all six operations the Value class supports, with brief explanations of why each gradient is what it is:

**Addition: `a + b`** Local gradients: ∂/∂a = 1, ∂/∂b = 1. Adding doesn't scale either input — if you increase `a` by ε, the output increases by exactly ε.

**Multiplication: `a * b`** Local gradients: ∂/∂a = b, ∂/∂b = a. If you increase `a` by ε, the output increases by b × ε. The other input acts as a scaling factor.

**Power: `a ** n`** Local gradient: ∂/∂a = n · a^(n−1). This is the power rule from calculus.

**Logarithm: `log(a)`** Local gradient: ∂/∂a = 1/a. The slope of ln(x) is 1/x — steep near zero, flat for large values.

**Exponential: `exp(a)`** Local gradient: ∂/∂a = exp(a). The exponential function is its own derivative.

**ReLU: `relu(a) = max(0, a)`** Local gradient: ∂/∂a = 1 if a > 0, else 0. If the input is positive, the gradient passes through unchanged. If negative, the gradient is blocked entirely.

These six operations are all you need. Every computation in the entire GPT model — attention scores, softmax, loss — is built from combinations of these six building blocks. The chain rule stitches them together.

---

## Parameters

The parameters are the knowledge of the model. They're a collection of floating point numbers (wrapped in `Value` so gradients can flow through them) that start out random and get iteratively adjusted during training.

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

Each parameter is initialized to a small random number from a Gaussian distribution. They need to start random — if all weights were identical, every neuron would compute the same thing and learning would stall. Small random values break this symmetry.

Notice how the dimensions chain together: `wte` maps a token id to a 16-dim vector, `attn_wq` maps a 16-dim vector to another 16-dim vector, `mlp_fc1` maps 16-dim to 64-dim, `mlp_fc2` maps 64-dim back to 16-dim, and `lm_head` maps 16-dim to 27 (vocab size). Each matrix's input dimension must match the previous matrix's output dimension — otherwise the dot products in `linear()` would be between vectors of different lengths, which is undefined. This is the fundamental constraint that shapes the entire architecture.

The `state_dict` organizes parameters into named matrices. We'll see what each one does shortly. The total comes to 4,192 parameters. GPT-2 had 1.6 billion. Modern LLMs have hundreds of billions. The math is the same.

In PyTorch, these would be `torch.nn.Parameter` tensors stored inside `torch.nn.Module` objects, initialized with carefully tuned distributions, and potentially sharded across multiple GPUs. The `state_dict` name is borrowed directly from PyTorch's convention for serializing model weights.

---

## Embeddings: how the model represents tokens

This is a piece that deserves more attention than it usually gets.

### The token embedding table (wte)

The model can't work with raw token ids. It can't do math on the number 5 in any meaningful way — token 5 isn't "greater than" token 3. It needs each token to be represented as a vector (a list of numbers) that it can transform and combine with other vectors.

The token embedding table `wte` is a matrix of shape `[27, 16]` — 27 rows (one per token in the vocabulary), 16 columns (the embedding dimension). When the model sees token id 5, it grabs row 5 from this table. That row — a vector of 16 numbers — _is_ the model's representation of that token.

At initialization, every row is random. The model has no idea that "a" and "e" are both vowels, or that "q" usually appears before "u." All of that structure must be learned from the training signal.

### The position embedding table (wpe)

The position embedding table `wpe` works identically but for positions. It has shape `[16, 16]` — 16 rows (one per possible position in the sequence, since `block_size = 16`), 16 columns. When the model processes the token at position 3, it looks up row 3 from `wpe`.

Why does the model need position information? Because attention, as we'll see shortly, is position-blind by default. It compares tokens based on their content, not their location. Without position embeddings, the model literally cannot tell "ab" from "ba" — both contain the same tokens with the same content vectors. Position embeddings break this symmetry.

### Why we add them together

The token embedding and position embedding are added element-wise:

```python
x = [t + p for t, p in zip(tok_emb, pos_emb)]
```

This is a natural question: why add instead of concatenate? Concatenation would keep the two signals separate, giving the model a 32-dimensional vector with 16 dimensions for "what token" and 16 for "where."

Addition is more efficient — it keeps the dimension at 16, which cascades through every subsequent layer. But there's a deeper reason: addition creates a shared space where token identity and position interact implicitly. Each dimension of the resulting vector carries a blend of both signals. The model's later layers learn to disentangle them as needed.

### How embeddings learn

Here's the part that often gets skipped. Looking up a row from a table might not seem like a differentiable operation, but it is. Looking up row 5 from a matrix is mathematically identical to multiplying the matrix by a one-hot vector — a vector of all zeros with a single 1 at index 5. That's just a matrix multiplication, and gradients flow through matrix multiplications perfectly well.

During `loss.backward()`, the gradient reaches the embedding vector for whichever token was used at that position. Only that row gets a nonzero gradient — the other 26 rows have zero gradient for that particular training step. Over many training steps, each row accumulates gradients from every example where that token appeared, and the optimizer nudges it toward a representation that makes predictions better.

So the model discovers structure entirely from the prediction objective. It learns that giving "a" and "e" similar vectors (because they appear in similar contexts in names) leads to better next-token predictions. Nobody tells it about vowels. The structure emerges from the loss signal alone.

> **🎬 Animation: embedding learning.** Show the 27 token embedding vectors as points in 2D (projected from 16D). At step 0, they're randomly scattered. As training progresses, animate them reorganizing — vowels clustering together, common consonants grouping, rare letters drifting apart. If possible, use real data from running microgpt and extracting embeddings at checkpoints during training.

In production models, `wte` might be [100,000 × 4,096] — a hundred thousand tokens, each represented by a 4,096-dimensional vector. These embeddings encode rich semantic relationships, all learned from the same mechanism: look up a row, predict the next token, backpropagate, nudge the row.

---

## Core building blocks

Before we look at the full model, there are three small helper functions.

### Linear layers

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

A linear layer multiplies an input vector by a weight matrix. Each row of the weight matrix produces one element of the output via a dot product with the input.

This is the fundamental building block. A linear layer doesn't add meaning on its own — it reshapes representations so later layers can work with them. It projects data from one space into another.

In PyTorch, this is `torch.nn.Linear` and it runs as a single highly optimized matrix multiplication on the GPU. Here, it's a nested loop over individual scalar multiplications.

### Softmax

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

Softmax converts a vector of raw scores (logits) into a probability distribution: all values become positive and sum to 1. It works by exponentiating each value (which makes everything positive) and then dividing by the total (which makes them sum to 1).

Subtracting the max first is a numerical stability trick — it doesn't change the math but prevents the exponentials from overflowing to infinity.

Softmax doesn't make decisions — it normalizes them. It appears twice in the model: once inside attention (to weight how much each past token contributes) and once at the output (to produce a probability distribution over the next token).

### RMSNorm

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

RMSNorm rescales a vector so its root-mean-square magnitude is 1. This keeps activations in a reasonable numerical range as data flows through the network.

Without normalization, values can grow or shrink exponentially as they pass through repeated linear transformations. When values get too large, gradients explode and training diverges. When they get too small, gradients vanish and training stalls. Normalization prevents both.

This is one of many components in deep learning that exist primarily for numerical reasons, not conceptual ones.

---

## The GPT forward pass

Now we can look at the full model. The `gpt()` function takes a single token and its position, processes it through the network, and returns 27 scores (logits) representing how likely each token is to come next.

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

Let's walk through this piece by piece.

### The residual stream

Before diving into attention and MLP, notice the overall wiring pattern. The model maintains a vector `x` that flows through every layer. At each block, the code saves a copy (`x_residual`), transforms `x`, and then adds the result back:

```python
x_residual = x
x = <some transformation>
x = [a + b for a, b in zip(x, x_residual)]
```

This is a residual connection. The model doesn't replace `x` — it adds a correction to it. Think of it as a shared highway: each block reads from the highway, computes something useful, and writes its contribution back. Information flows through the highway unimpeded, and each block incrementally refines the representation.

This matters for two practical reasons. First, it makes training possible — without residual connections, gradients would have to flow through every transformation in sequence, and they'd shrink to nothing in deep networks. With residual connections, gradients have a direct path backward through the addition operations (whose local gradient is 1, so nothing shrinks). Second, it means each layer's job is easy: it only needs to compute the _difference_ between the current representation and a better one, not build the entire representation from scratch.

> **🎬 Animation: the residual stream.** Show a single vector flowing downward through layers. At each attention block, a copy branches off to the right, gets transformed, and the result is added back to the main stream. Same for MLP. The main stream is visually prominent — a highway. The side computations are incremental refinements feeding back into it.

### Attention in detail

The attention block is where tokens communicate with each other. It's the only place in the model where a token at position t can look at tokens at positions 0, 1, ..., t−1.

The idea is surprisingly close to a database lookup. In a traditional database, you have a query ("find customers named Emma"), keys (the indexed fields in each row), and values (the data you get back). Attention does the same thing, but soft and learned: the current token constructs a query ("what kind of context am I looking for?"), every previous token advertises a key ("here's what I contain"), and the lookup returns a weighted blend of values rather than a single exact match. The model learns _what to search for_ and _what to retrieve_ entirely from the training signal.

This is what made the Transformer a breakthrough over its predecessor, the RNN. RNNs processed tokens one at a time and compressed all past context into a single fixed-size hidden state — like trying to remember an entire book by passing a single note from page to page. Long-range dependencies got diluted or lost entirely. Attention sidesteps this bottleneck: every token can directly look up any previous token in the sequence, no matter how far back, and crucially, all these lookups can be computed in parallel on a GPU rather than sequentially.

At each position, the model computes three learned projections — queries, keys, and values:

```python
q = linear(x, state_dict[f'layer{li}.attn_wq'])
k = linear(x, state_dict[f'layer{li}.attn_wk'])
v = linear(x, state_dict[f'layer{li}.attn_wv'])
```

Each of these is a matrix multiplication that transforms the 16-dimensional input into a new 16-dimensional vector. But each has a different purpose:

- The **query** (q) represents what this position is looking for
- The **key** (k) represents what this position has to offer as a match
- The **value** (v) represents what information this position will contribute if selected

The key and value for each position are stored in the KV cache so that future positions can attend to them:

```python
keys[li].append(k)
values[li].append(v)
```

#### A worked example

Let's make this concrete. Say we're processing the name "emma" and we're at position 3 (the letter "a", trying to predict BOS). The model has already processed positions 0–2, so the KV cache contains keys and values for BOS, "e", "m", "m".

For one attention head operating on a 4-dimensional slice, suppose after the linear projections we have:

```
q (position 3, "a"):  [1.2, -0.5,  0.8,  0.3]

Cached keys:
  k₀ (BOS):           [0.1,  0.3, -0.2,  0.5]
  k₁ ("e"):           [0.9, -0.4,  0.7,  0.1]
  k₂ ("m"):           [0.2,  0.1,  0.3, -0.6]
  k₃ ("m"):           [0.2,  0.1,  0.3, -0.6]

Cached values:
  v₀ (BOS):           [0.5,  0.2, -0.1,  0.4]
  v₁ ("e"):           [0.8, -0.3,  0.6,  0.2]
  v₂ ("m"):           [-0.1, 0.7,  0.3,  0.5]
  v₃ ("m"):           [-0.1, 0.7,  0.3,  0.5]
```

**Step 1: compute attention scores.** For each cached key, compute the dot product with the query:

```
score₀ = (1.2×0.1) + (-0.5×0.3) + (0.8×-0.2) + (0.3×0.5) = 0.12 - 0.15 - 0.16 + 0.15 = -0.04
score₁ = (1.2×0.9) + (-0.5×-0.4) + (0.8×0.7) + (0.3×0.1) = 1.08 + 0.20 + 0.56 + 0.03 = 1.87
score₂ = (1.2×0.2) + (-0.5×0.1) + (0.8×0.3) + (0.3×-0.6) = 0.24 - 0.05 + 0.24 - 0.18 = 0.25
score₃ = same as score₂ = 0.25
```

The dot product measures similarity between the query and each key. Position 1 ("e") has the highest score — the model has learned that "a" should pay attention to "e" in this context.

**Step 2: scale.** Divide each score by √(head_dim) = √4 = 2:

```
scaled: [-0.02, 0.935, 0.125, 0.125]
```

Why scale? Without it, when the head dimension is large, the dot products become large in magnitude. Large inputs to softmax produce distributions that are nearly one-hot (all probability on one token), which means gradients become tiny and learning stalls. Dividing by √d keeps the variance of the scores roughly constant regardless of dimension.

**Step 3: softmax.** Convert scores to probabilities:

```
weights: [0.14, 0.37, 0.245, 0.245]
```

Position 1 ("e") gets the most weight, but the other positions contribute too. This is a soft selection — rather than picking one position, the model blends information from all positions in proportion to relevance.

**Step 4: weighted sum of values.** For each dimension of the output, take a weighted sum of the corresponding value dimensions:

```
out[0] = 0.14×0.5  + 0.37×0.8  + 0.245×(-0.1) + 0.245×(-0.1) = 0.317
out[1] = 0.14×0.2  + 0.37×(-0.3) + 0.245×0.7 + 0.245×0.7     = 0.230
out[2] = 0.14×(-0.1) + 0.37×0.6 + 0.245×0.3 + 0.245×0.3      = 0.355
out[3] = 0.14×0.4  + 0.37×0.2  + 0.245×0.5 + 0.245×0.5       = 0.375
```

The output is a 4-dimensional vector that blends information from all previous positions, weighted by relevance to the current query. This is what one attention head produces.

> **🎬 Animation: the attention mechanism.** Show the current position's query vector reaching backward to compare against each cached key. Visualize each dot product as the brightness or thickness of a connecting line. Apply softmax — the lines normalize so their intensities sum to 1. Then show each value vector being scaled by its attention weight and the results being summed into a single output vector.

#### Multi-head attention

The model doesn't run one attention head — it runs four in parallel:

```python
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]
```

The 16-dimensional q, k, and v vectors are split into 4 heads of 4 dimensions each. Each head independently computes attention over its own 4-dimensional slice. The outputs are concatenated back into a 16-dimensional vector and projected through one more linear layer (`attn_wo`):

```python
x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
```

Why multiple heads? Each head can specialize. One head might learn to attend to the previous vowel. Another might track consonant clusters. Another might focus on the beginning of the name. By having multiple heads operating in parallel on different subspaces, the model can track several patterns simultaneously.

### The MLP block

After attention (communication between positions), the model applies an MLP (computation within a single position):

```python
x_residual = x
x = rmsnorm(x)
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
x = [xi.relu() for xi in x]
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
x = [a + b for a, b in zip(x, x_residual)]
```

The MLP projects the 16-dimensional vector up to 64 dimensions, applies ReLU (which zeros out all negative values), and projects back down to 16 dimensions. This is another residual block — the result is added back to the input.

Why does ReLU matter? Without a nonlinearity, stacking two linear layers is equivalent to a single linear layer (matrix times matrix is still a matrix). ReLU breaks this — it lets the network compute nonlinear functions, which is essential for learning complex patterns.

The expansion to 4× the dimension and back gives the network a wider space to work in temporarily. ReLU zeroes out roughly half the 64 dimensions, and the model learns which dimensions to activate for different inputs. Think of it as a set of 64 learned features, of which only a relevant subset fires for any given input.

The Transformer architecture alternates between attention (mixing information across positions) and MLP (transforming information within a position). Attention gathers context; MLP processes it.

### Output projection

After all layers, the final representation is projected to vocabulary size:

```python
logits = linear(x, state_dict['lm_head'])
```

This produces 27 numbers — one per token in the vocabulary. Higher means the model thinks that token is more likely to come next. These raw scores are called logits, and they'll be converted to probabilities by softmax during training and inference.

---

## Training

The training loop puts everything together. Each step: pick a document, run the model forward, compute the loss, backpropagate, and update the parameters.

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

### The loss: measuring misprediction

At each position, the model produces probabilities over 27 tokens. The loss measures how wrong the prediction was:

```python
loss_t = -probs[target_id].log()
```

This is the cross-entropy loss. It takes the probability the model assigned to the correct next token and computes the negative logarithm. If the model assigned probability 0.7 to the correct token, the loss is −log(0.7) ≈ 0.36 — small, because the model was fairly confident and correct. If the model assigned probability 0.01, the loss is −log(0.01) ≈ 4.6 — large, because the model was very surprised by what actually came next.

The loss is averaged across all positions in the document. An untrained model with random weights assigns roughly equal probability to all 27 tokens, so the initial loss is around −log(1/27) ≈ 3.3. Over training, this drops to about 2.37, meaning the model has learned to concentrate probability on the correct token.

> **🎬 Animation: the loss function.** Show a bar chart of 27 predicted probabilities. Highlight the correct token's bar. Show −log of that bar's height as the loss value. Animate training: the correct token's bar grows taller over many steps, and the loss number shrinks.

### The backward pass

```python
loss.backward()
```

One call propagates gradients through the entire computation graph — from the loss, back through softmax, back through every layer of the model, back through the embeddings, all the way to every parameter. After this call, every parameter's `.grad` field contains the gradient of the loss with respect to that parameter: the direction and magnitude of change that would reduce the loss.

### The Adam optimizer

We could update parameters with simple gradient descent: `p.data -= lr * p.grad`. But Adam is smarter.

```python
m[i] = beta1 * m[i] + (1 - beta1) * p.grad       # momentum
v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2   # squared gradient
m_hat = m[i] / (1 - beta1 ** (step + 1))           # bias correction
v_hat = v[i] / (1 - beta2 ** (step + 1))           # bias correction
p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam) # update
```

Adam maintains two running averages for each parameter:

**Momentum (m):** an exponential moving average of recent gradients. Each training step sees only one document, so individual gradients are noisy — they point in roughly the right direction but jitter around. Momentum smooths this out by averaging over recent steps, giving a more stable update direction. It's like a rolling ball that resists sudden direction changes.

**Second moment (v):** an exponential moving average of recent squared gradients. This tracks the magnitude of gradients for each parameter. Parameters that consistently receive large gradients get their updates scaled down. Parameters with small gradients get relatively larger updates. This per-parameter adaptation means the optimizer automatically adjusts its step size for each of the 4,192 parameters.

**Bias correction (m_hat, v_hat):** since `m` and `v` start at zero, they're biased toward zero in the early steps. The correction factors `1 / (1 - beta^(step+1))` compensate for this, making the estimates accurate even at the beginning of training.

The learning rate decays linearly from 0.01 to 0 over the course of training. Early in training, you want large steps to quickly find a good region. Late in training, you want small steps to settle into a precise minimum.

After updating, we reset `p.grad = 0` for the next step.

> **🎬 Animation: one full training step.** Show the cycle: a document is selected → tokenized → fed through the model (the computation graph grows forward) → loss is computed → backward pass (gradient signal flows backward through the entire graph) → each parameter is nudged slightly → loss decreases. Then speed up and show 1,000 steps as a timelapse with the loss curve dropping from 3.3 to 2.37.

In production, training uses large batches (millions of tokens per step rather than one document), gradient accumulation, mixed precision arithmetic (float16 or bfloat16 instead of float64), and is distributed across thousands of GPUs running for months. PyTorch's `torch.optim.Adam` does the same math as the loop above, but on tensors containing millions of parameters at once.

---

## Inference: generating new names

Once training is done, the parameters are frozen and we generate by repeatedly sampling from the model's predictions:

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

The process is simple: start with BOS, run the model, sample a token from the output distribution, feed it back in, repeat until the model produces BOS again (signaling the end of the name). At the very start, the KV cache is empty — the model has no past context to attend to. The first token requires a full forward pass to generate its keys and values from scratch. This is why in production LLMs, the first token of a response takes noticeably longer to appear (called the "prefill" phase). After that, each subsequent token only needs to process one new position while reusing all the cached keys and values from previous positions, making generation much faster.

### Temperature

Before softmax, the logits are divided by a temperature parameter:

```python
probs = softmax([l / temperature for l in logits])
```

This controls how "creative" the model is. Suppose the logits are `[2.0, 1.0, 0.5]`.

At **temperature 1.0** (no change), softmax produces a moderate distribution — the top token is favored but others have a chance.

At **temperature 0.5**, the logits become `[4.0, 2.0, 1.0]` — the differences are amplified. Softmax produces a sharper distribution. The model is more conservative, sticking closely to its top predictions.

At **temperature 0.1**, the logits become `[20.0, 10.0, 5.0]` — softmax produces a nearly one-hot distribution. The model almost always picks the single most likely token. Approaching temperature 0 is called greedy decoding.

At **temperature 2.0**, the logits become `[1.0, 0.5, 0.25]` — the differences are compressed. Softmax produces a flatter distribution. The model samples more diversely, sometimes picking unlikely tokens. This produces more varied but potentially less coherent output.

> **🎬 Animation: temperature.** Show the same set of logits being transformed at three temperatures. Three bar charts side by side: low temperature (one bar dominates), medium (clear preference but diverse), high (nearly uniform). Below each, show a sample name generated at that temperature.

The model never plans ahead. It doesn't think "I'm going to spell 'anna' — it just picks each letter based on what came before. Coherent names emerge from many local, probabilistic decisions.

The generated names look like this:

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

These are not real names from the dataset. They're new sequences that the model "hallucinated" — statistically plausible completions that happen to sound like names. This is the same phenomenon as ChatGPT confidently stating a false fact. Both are plausible-sounding completions that happen not to be real.

---

## From microGPT to modern LLMs

microGPT contains the complete algorithmic core. But between 200 lines of Python and ChatGPT, a lot changes. None of it alters the fundamental algorithm — it's all about making it work at scale.

### Scale of parameters and data

microGPT has 4,192 parameters and trains on 32,000 short names. GPT-2 had 1.6 billion parameters and trained on 40GB of internet text. GPT-4-class models have hundreds of billions of parameters and train on trillions of tokens. The architecture is recognizably the same Transformer — just much wider (embedding dimensions of 10,000+) and much deeper (100+ layers). The relationship between model size and data isn't arbitrary — the Chinchilla paper showed that for a given compute budget, there's an optimal balance between number of parameters and number of training tokens, and that most earlier models were too large for the amount of data they were trained on.

### What PyTorch actually does

Throughout this article, I've noted that production models use PyTorch. Here's a summary of what that means concretely.

microGPT's `Value` class wraps individual numbers and tracks operations one scalar at a time. PyTorch replaces this with **tensors** — multi-dimensional arrays that can contain millions of numbers. A single `torch.matmul` call replaces the nested loops in our `linear()` function, computing thousands of dot products simultaneously on a GPU. The autograd system tracks operations on entire tensors rather than individual scalars, so `loss.backward()` computes gradients for millions of parameters in one pass. CUDA kernels on NVIDIA GPUs perform the actual arithmetic — a single GPU can do trillions of floating point operations per second.

The math is identical. The `Value` class and PyTorch's autograd implement the same chain rule algorithm. Every gradient that microGPT computes scalar-by-scalar, PyTorch computes in parallel on tensors. The result is the same — just many orders of magnitude faster.

### Positional encoding: from learned embeddings to RoPE

microGPT uses learned position embeddings — a lookup table where each position gets its own vector. This works fine up to the maximum sequence length seen during training (`block_size = 16`), but it can't generalize beyond that. Position 17 has no embedding.

Modern models use Rotary Position Embeddings (RoPE). Instead of adding a position vector, RoPE rotates the query and key vectors by an angle proportional to their position. The key insight: the dot product between a query at position i and a key at position j ends up depending only on their relative distance (i − j), not their absolute positions. This means the model can handle sequences longer than it was trained on, because the mechanism only cares about how far apart two tokens are.

### Activations: from ReLU to SwiGLU

microGPT uses ReLU in its MLP blocks — a simple function that passes positive values through and zeros out negative values. This works but is crude: it permanently kills information in the zeroed-out dimensions.

Modern models use gated activations like SwiGLU. Instead of a hard zero/pass decision, SwiGLU uses a learned gate that smoothly controls how much each dimension contributes. The gate is itself a function of the input, so the network learns which features to activate rather than relying on a fixed threshold. This is more expressive and wastes less capacity.

### Flash Attention

The attention computation in microGPT computes all dot products, stores all attention weights, and then does the weighted sum. At scale, this is a memory bottleneck — for a sequence of length 8,000, the attention weight matrix has 64 million entries per head.

Flash Attention is an algorithm that computes the exact same attention output but never materializes the full attention matrix in memory. It processes the computation in tiles that fit in the GPU's fast SRAM cache, dramatically reducing memory usage and improving speed. The mathematical result is identical — it's purely a systems-level optimization.

### Mixture of Experts

Standard Transformers activate all parameters for every token. Mixture of Experts (MoE) models have multiple MLP blocks at each layer (say, 8 or 16 "experts") but only activate a small subset (typically 2) for each token. A learned router decides which experts to use based on the input.

This allows much larger total parameter counts without proportionally increasing the compute per token. A model might have 100 billion total parameters but only use 15 billion for any given token. The trade-off: more total memory, but much faster per-token inference and training.

### Scaling laws

Researchers discovered predictable power-law relationships between compute budget, model size, dataset size, and final loss. The Chinchilla paper showed that for a fixed compute budget, there's an optimal balance — most earlier models were too large for their training data. These scaling laws let teams predict the performance of a model before committing millions of dollars to train it.

### Post-training: from document completer to chatbot

The model that comes out of training on raw text (the "pretrained" model) is a document completer. It can continue text, but it doesn't follow instructions or have a conversation.

**Supervised fine-tuning (SFT)** is the first step. You swap the training data for curated conversations in a specific format — system prompt, user message, assistant response — and keep training. Algorithmically, nothing changes. It's still next-token prediction. The model just learns to predict the next token in conversations rather than in web pages.

**Reinforcement learning from human feedback (RLHF)** adds a second signal. The model generates multiple responses to the same prompt, human raters rank them, and a reward model is trained on those rankings. The language model is then fine-tuned to maximize the reward model's scores. A simpler alternative called DPO (Direct Preference Optimization) skips the reward model and optimizes directly on the preference pairs.

The model is still predicting tokens. But the training signal has shifted from "what text naturally follows?" to "what text do humans prefer?"

### Inference at scale

Serving a model to millions of users requires its own engineering. The KV cache that microGPT builds explicitly must be managed carefully when handling thousands of concurrent requests — systems like vLLM use paged memory to avoid waste. Quantization reduces parameter precision from 16-bit floats to 8-bit or 4-bit integers, cutting memory usage with minimal quality loss. Speculative decoding uses a small, fast model to draft multiple tokens at once, then the large model verifies them in a single pass — producing the same output but faster. All of these are engineering optimizations around the same core loop: predict the next token, sample, repeat.

---

## Closing

The core algorithm of a large language model fits in 200 lines of Python. A dataset of sequences. A tokenizer to convert text to integers. An autograd engine to compute gradients. A Transformer that predicts the next token. An optimizer that adjusts parameters to make those predictions better. A sampling loop that generates new sequences.

Everything else — the GPU kernels, the distributed training, the post-training pipeline, the serving infrastructure — is about making this same algorithm work at scale, run faster, and produce more useful output.

If you understand these 200 lines, you understand the mechanism behind every conversation you've had with a language model.
