---
title: Embeddings, Explained
---
Embeddings have exploded in popularity with the rise of RAG systems that give ChatGPT and other LLMs relevant context to answer questions. But embeddings aren't new — they were the first successful application of machine learning to images and language, powering search engines and recommendation systems long before generative models became mainstream. In this article, I'll trace the journey of embeddings from early handcrafted features through Word2Vec's breakthrough "word math" to today's contextual embeddings.

### What Are Embeddings?

Computers operate on numbers. They can't directly work with words, images, or concepts. Everything must be converted into numbers first. The question is how.

The traditional approach was feature engineering — manually defining measurable characteristics that capture what matters. Imagine building a system to tell apart dogs and humans in photos. You might define features like height-to-width ratio, presence of clothing, upright posture, number of legs. For each image, you measure these features and get numbers:

```
Image_1: [height_ratio=1.8, has_clothing=1, upright=1, legs=2] → Human
Image_2: [height_ratio=0.7, has_clothing=0, upright=0, legs=4] → Dog
```

This works for dogs vs. humans because the differences are obvious.

But try distinguishing dogs from cats. Suddenly handcrafted features fail. Size varies enormously in both. Ear shape varies. Fur texture varies. You might try "whisker prominence" or "tail curviness," but you quickly realize you're not an expert in animal anatomy, there are thousands of subtle visual cues you can't articulate, and some distinguishing features might be invisible to human perception.

This is where latent features come in. Instead of hand-designing features, let the machine **learn features of** its own. Embeddings create hidden mathematical dimensions called latent features that capture patterns in data, even if we can't interpret what those dimensions represent.

Think of it as a coordinate system where similar things end up close together. A dog embedding might look like `[0.2, -1.1, 0.8, -0.3, 1.7, ...]`. We might not know what dimension #3 represents, but the machine learned that this combination of numbers distinguishes dogs from other animals.

### The Language Problem

While embeddings proved useful for images, they became transformational for language. Images have inherent structure — pixels arranged spatially, nearby pixels often relate to the same object. Language is less structured.

Consider how computers traditionally saw text: as arbitrary symbols. The word "cat" might be number 4,287 in a vocabulary, "dog" might be 8,193. These numbers are meaningless. There's no mathematical relationship between 4,287 and 8,193 that reflects cats and dogs being similar animals.

This is the breakthrough embeddings enabled. Instead of arbitrary IDs, words became points in a meaningful geometric space where "cat" and "dog" actually are close together, where "king" and "queen" have a mathematical relationship, where you can compute with meaning itself.

### Word2Vec: The First Major Milestone

[Word2Vec](https://arxiv.org/pdf/1301.3781) made one simple observation: words that appear in similar contexts probably have similar meanings.

Think about it:

- "The cat sat on the mat"
- "The dog sat on the mat"
- "The puppy sat on the mat"

Since "cat," "dog," and "puppy" all appear in similar contexts, they should have similar embeddings.

### How Word2Vec Learns

Word2Vec learns from raw text without any labels. It uses one of two training methods:

**Skip-gram**: Given a target word, predict the surrounding words.

Take "The quick brown fox jumps." For the target word "brown," create training pairs:

- "brown" → "The"
- "brown" → "quick"
- "brown" → "fox"
- "brown" → "jumps"

The neural network takes "brown" as input, and its hidden layer becomes the embedding. The network learns by trying to predict the context words. Words that consistently appear in similar contexts get similar embeddings.

**CBOW (Continuous Bag of Words)**: The reverse. Given context words, predict the target.

Input: {"The", "quick", "_", "fox", "jumps"}
Target: Predict "brown"

Both methods create a space where semantic similarity becomes geometric proximity.

![[content/Writing/images/word2vec_diagram.png]]


The famous demonstration of Word2Vec was vector arithmetic that seemed to capture meaning:

**king - man + woman ≈ queen**

How does this work? "King" has high values on dimensions related to royalty and maleness. "Man" has high maleness but neutral royalty. "Woman" has high femaleness but neutral royalty. So king - man + woman removes maleness, adds femaleness, keeps royalty. The result points to "queen."

Other examples:

- Paris - France + Italy ≈ Rome
- walking - walk + swims ≈ swimming

The geometric relationships in embedding space mirror semantic relationships in language.

Word2Vec had one crucial flaw: one vector per word, regardless of context.

Consider "bank":

- "I deposited money at the bank" (financial institution)
- "We sat by the river bank" (edge of water)

Word2Vec gives "bank" the same embedding in both sentences, averaging across all meanings. This fails for ambiguous words.

### Contextual Embeddings

The next breakthrough was making word meanings change based on context. This led to contextual embeddings — the same word gets different embeddings depending on surrounding words.

[ELMo](https://arxiv.org/abs/1802.05365) was an early solution using bidirectional LSTMs. For "bank" in "river bank," it looks left at "river" and right at words like "muddy," creating an embedding weighted toward geographical meaning. For "bank" in "deposit money," the context creates an embedding weighted toward financial meaning.

The real advance came with the attention mechanism, which lets models dynamically focus on relevant parts of input when creating embeddings. Instead of just immediate neighbors, attention considers which words are most important for understanding the current word, what long-range dependencies matter, which contextual clues are most relevant.

### BERT: Modern Embeddings

BERT represents the current state of the art. Its innovation is both architectural and in how it trains. Training BERT is elegantly simple, it uses "Masked Language Modeling (MLM)", a training method where words are randomly masked in sentences and the goal is to predict them using bidirectional context.

Input: "The cat sat on the mat"  
Masked: "The [MASK] sat on the [MASK]"  
Task: Predict [MASK] #1 = "cat" and [MASK] #2 = "mat"

This sounds similar to Word2Vec's CBOW, but there's a crucial difference. Word2Vec only looks at a small fixed window of nearby words — typically 5-10 words total. BERT sees the entire sentence. For a sentence like "The bank manager said the company's financial performance near the river was excellent," Word2Vec might only see "The ___ manager said the" when predicting "bank," missing the crucial disambiguating words "financial" and "river" that appear later. BERT's attention mechanism can focus on both "financial" and "river" even though they're far apart, giving it much richer context.

To predict masked words accurately, BERT must learn syntactic relationships, semantic relationships, long-range dependencies, and world knowledge. And it learns all this from raw text without any human annotations.

BERT uses the transformer architecture, which processes all words in parallel while letting each word "attend to" every other word.

In "The cat sat on the mat":

- "cat" attends strongly to "sat" (what the cat is doing)
- "sat" attends to both "cat" (who is sitting) and "mat" (where)
- "mat" attends to "on" and "sat" (its role as location)

What makes embedding models powerful is they don't require labeled data. They're unsupervised learning methods that discover meanings purely through statistical patterns — which words appear together, which contexts are similar. This lets us train them on massive amounts of internet text without human annotation.

That said, most useful applications — facial recognition, sentiment analysis, document classification — do require a final step of fine-tuning with labeled data. BERT's pre-training creates general-purpose embeddings that understand language. For specific tasks, you add task-specific layers and train on labeled examples. The pre-trained embeddings already capture rich linguistic knowledge, so fine-tuning needs much less data than training from scratch.

### Why This Matters

The journey from handcrafted features to Word2Vec to BERT shows how embeddings are able to represent meaning mathematically. They're the universal language that lets AI systems build vector spaces across words, images, sounds, and even molecules.

This matters because embeddings enable the AI applications we use daily. When you search Google, your query gets converted to an embedding and matched against document embeddings based on semantic similarity, not just keyword matching. When ChatGPT answers questions about your documents, it uses RAG — converting both your question and relevant passages into embeddings, finding the closest matches, then generating a response with that context. When DALL-E generates an image from text, it's matching text embeddings to image embeddings in a shared space. 
