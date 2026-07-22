---
title: Open Source Models Just Need to Be Good Enough For Long Enough
---
_Kimi K3's release, and how competitive open models could shape the hardware and software standards of AI_

Kimi K3 is not the best model available, and that may not matter. Moonshot’s own comparisons place it behind Claude Fable 5 and GPT-5.6 Sol overall, even though it beats them on several coding and agentic benchmarks and sits close to the closed frontier in independent evaluations. [Kimi K3 release](https://www.kimi.com/fr-fr/blog/kimi-k3?utm_source=chatgpt.com)

The usual question is whether an open model will eventually become the most capable. I think the more interesting question is whether it needs to.

K3 shares important computational patterns with other open models from DeepSeek, Qwen, and OpenAI. If these models remain capable enough for developers and enterprises to adopt them, and similar enough for optimizations to transfer between them, they could become a common target for the rest of the ecosystem. Inference providers would have stronger incentives to make them cheaper and easier to deploy, while hardware companies would have stronger reasons to support their recurring patterns in future products.

That creates a flywheel. Better software and hardware drive more adoption, and more adoption justifies further investment. Open models do not need to become the state of the art for this to happen. They need to remain good enough for long enough that the ecosystem begins building around them.

## A common target

K3 is a 2.8-trillion-parameter Mixture-of-Experts model. Instead of using the entire model for every token, it routes each token through a small selection of experts. It also introduces Kimi Delta Attention, an approach designed to make long-context processing more efficient. [Kimi K3 architecture](https://www.kimi.com/fr-fr/blog/kimi-k3)

The exact specifications matter less than the convergence around them. DeepSeek-V3 established a competitive design based on MoE and a more memory-efficient form of attention. Kimi K2’s technical report explicitly says it followed a similar architecture, and K3 extends that lineage rather than discarding it. Other leading open models make different choices, but many are converging on sparse expert activation and similar efficiency goals.

These models do not need to be identical. They only need to share enough computational patterns that optimizations built for one also benefit others, creating a common target for the rest of the industry.

## The optimization flywheel

Inference providers move first. Because open models expose their weights and architectures, providers can inspect where they are slow and build software specifically around them.

We can already see this with companies such as Baseten, Together, and Fireworks. They have built model-specific optimizations for DeepSeek, Kimi, gpt-oss, and other open MoE models. Although their implementations differ, the objective is the same: make these models faster and less expensive to operate on existing hardware. [Baseten](https://www.baseten.co/blog/how-we-built-the-fastest-glm-5-api/), [Together](https://www.together.ai/blog/fastest-inference-for-deepseek-r1-0528-with-nvidia-hgx-b200)

K3 is already entering this process. Its Delta Attention mechanism requires a different approach to caching, so Moonshot has contributed an implementation to vLLM, one of the main open-source inference engines. It is also working with inference partners and open-source maintainers as part of the release.

This is an advantage closed architectures cannot reproduce as broadly. A proprietary lab can work privately with hardware and cloud partners, but independent providers cannot all inspect the model and compete to improve the same implementation.

Hardware companies respond more slowly because chips take years to design. They also do not build around one named model. Instead, they invest in the recurring operations that appear across many valuable workloads. That is already happening with MoE. NVIDIA has designed its latest systems to run MoE models more efficiently and has worked extensively on accelerating DeepSeek. AMD has also added software and hardware support aimed at the same family of models, with shared implementations covering DeepSeek, Qwen, Llama, and other MoE architectures. [NVIDIA's DeepSeek-R1 work](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/) and [AMD's MoE optimization](https://rocmdocs.amd.com/en/develop/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html)

AMD and NVIDIA are not making separate bets on every model. They are responding to a computational pattern that has become common enough to justify investment.

From there, the flywheel is straightforward:

1. Competitive open models make an architecture widely available.
2. Inference providers optimize around it.
3. Hardware companies invest in the recurring patterns behind those models.
4. Better software and hardware make compatible models easier to deploy.
5. Adoption grows, making further optimization more valuable.

Nobody needs to coordinate this. Each participant is responding to the market.

This does not make the current model more intelligent by itself. It makes it faster, cheaper to operate, and easier to deploy. Over time, those economics can also improve future generations by making experimentation less expensive and attracting more researchers and developers.

## Good enough can beat better

The objection here is that hardware companies optimize general operations, not specific models such as Kimi. That is true, and it is the mechanism I am describing. Widely adopted models determine which operations become economically worthwhile to optimize.

MoE appears to have crossed that threshold. Delta Attention has not. It will first receive model-specific software support. If Delta Attention or similar approaches persist across several important models, they may eventually influence what future infrastructure is designed to run efficiently. That is why “long enough” matters. Whereas software can adapt quickly, hardware needs a stable signal before companies make a multiyear investment.

For that signal to persist, these models do not need to be the best. They only need to be good enough for enough users. Much of enterprise AI involves summarizing calls, searching internal knowledge, parsing documents, and automating workflows. Once open models can perform these tasks at the same reliability as proprietary models, additional benchmark gains and PhD+ levels of intelligence may matter less than cost, reliability, and ease of deployment. The hardest tasks may determine which model tops the benchmarks, but the most common tasks may determine which models are deployed at scale and, therefore, which architectures become standards.

An analogy to this could be USB-C. It did not become the standard because it was the best connector anyone could possibly design. It became the standard because enough manufacturers, suppliers, and consumers coordinated around it. Eventually, the value was no longer just the connector but everything built around it. A company could invent something technically better and still lose because the improvement was not large enough to justify abandoning the ecosystem.

AI architectures can benefit from the same path dependence - a closed model could remain ahead in raw capability. A proprietary architecture could even be objectively better. But once inference providers, hardware vendors, software frameworks, universities, and technical talent have accumulated around the open alternative, being slightly better may no longer be enough to displace it.
## Setting the standard is geopolitical influence

The US is generally considered ahead in the AI race because models from OpenAI, Anthropic, and Google still lead most benchmarks. That lead is real, but it may be an incomplete way of measuring the AI race.

If Kimi, DeepSeek, and Qwen become the leading open model developers that university students can study, startups can deploy, and and hardware companies optimize around, China could increasingly set the standards for the broader AI ecosystem even while American companies retain the most capable models. The US may still manufacture the chips and operate the clouds, but benefiting from a standard is not the same as setting it.
## Where I could be wrong

Two things could prevent this from happening.

First, open models may not converge enough to create a common hardware target. Their architectures could remain too different, change too quickly, or require optimizations that do not transfer between models. If each model relies on substantially different computational patterns, there is little reason for hardware companies to standardize around the open ecosystem.

Second, the largest model companies could vertically integrate. Rather than waiting for NVIDIA, AMD, or the wider market to optimize around shared patterns, they could design hardware around their own models and serving workloads. OpenAI’s Jalapeño chip is an early example of this approach. OpenAI designed it with Broadcom as part of an effort to control more of the stack, from models and serving software to inference hardware. [OpenAI’s Jalapeño announcement](https://openai.com/index/openai-broadcom-jalapeno-inference-chip/?utm_source=chatgpt.com)

In that world, the industry may support several proprietary stacks rather than converging on a common architecture. Each major lab could optimize its own models, software, and hardware together, making convergence among open models less important.

K3 makes a different outcome plausible. Open models do not need to beat the state of the art. They need to be capable enough for developers and enterprises to adopt them, and similar enough for the broader ecosystem to optimize around them. Better software and hardware would then make them cheaper and easier to deploy, driving more adoption and justifying further investment. That flywheel could turn models that begin as merely competitive into the standards that future models and infrastructure are built around.
