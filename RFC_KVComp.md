# [RFC]: KVComp Sparse Attention for Long-Context Inference

## Motivation

Long-context inference has become increasingly critical for large language models (LLMs) as applications require processing extensive contextual information. However, traditional attention mechanisms face significant challenges in this domain:

1. **Attention Computational Overhead**: Full attention computation exhibits quadratic complexity $O(L^2)$ with respect to sequence length $L$, making it computationally expensive for long contexts.

2. **KV Memory Bottleneck**: The KV cache grows linearly $O(L)$ with sequence length $L$ and model depth, leading to prohibitive memory consumption for long sequences.

3. **KV Communication Bandwidth and Latency Overhead**: In distributed KV cache architectures such as PD disaggregation and Mooncake/LMCache's remote separation, KV transfer operations consume substantial communication bandwidth and incur significant latency. Transferring the entire KV cache across network boundaries becomes a critical bottleneck, especially for long-context scenarios where the KV cache size grows linearly with sequence length. This communication overhead can dominate inference latency and limit the scalability of distributed inference systems.


Sparse attention mechanisms are essential for efficient long-context inference to address these challenges, as they only compute attention on the most relevant (top-k) tokens of the context sequence. The difficulty to apply sparse attention is how to maintan the accurancy with full attention. 

**KVComp** (Key-Value Compression) addresses these challenges by implementing a **Hash-Aware Top-k Attention** framework that efficiently compresses the KV cache while maintaining model accuracy. As detailed in the ACL-2025 paper ["HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference"](https://aclanthology.org/2025.findings-acl.1275/), KVComp leverages trainable hash functions to compute attention relevance, which is significantly faster than exact attention score computation while preserving the relative order of query-key scores needed for top-k attention.

Key benefits of KVComp include:
- **2-5x speedup** in attention computation for long sequences
- **Minimal accuracy loss** (< 1%) on downstream tasks like LongBenchv2
- **Scalable to 128K+ context lengths** with linear complexity
- **Hardware-efficient** implementations optimized for NPU architectures with customized NPU kernels like `hamming_dist_top_k` for computing top-k indices and `reshape_and_cache_BNSD` for efficiently hashk key cache layout
- **Broad model architecture support**: Works seamlessly with both **GQA models** (e.g., Qwen3) and **MLA models** (e.g., DeepSeek), demonstrating versatility across different attention mechanisms
- **Flexible deployment**: Supports both **KV cache non-offloading** (HBM) and **KV cache offloading** (DRAM/SSD) scenarios, enabling efficient memory management for various hardware configurations

Integrating KVComp into vllm-ascend will significantly enhance the project's capability to handle long-context inference efficiently on Ascend hardware, addressing a critical gap in the current feature set and aligning with the community's roadmap for 2026.

## Proposed Change

The integration of KVComp into vllm-ascend requires modifications across several key components, with the **primary focus on the attention backend**. The changes can be categorized as follows:

### 1. Attention Backend Enhancement (Primary Change)

The core modification involves updating vllm-ascend's attention backend to incorporate KVComp's sparse attention mechanisms:

- **Hash Encoding Integration**: Implement hash encoding modules that convert attention keys and queries into compact binary hash codes. This includes:
  - Hash encoder initialization and configuration
  - **Comprehensive model architecture support**: Full support for both **GQA models** (e.g., Qwen3 series) and **MLA models** (e.g., DeepSeek series), with specialized hash encoding strategies for each:
    - For GQA models: Standard hash encoding based on query/key representations
    - For MLA models: Dual hash encoding leveraging both NoPE (Normalized Position Embedding) and RoPE (Rotary Position Embedding) components
  - Hash code computation during attention forward pass

- **Sparse Attention Computation**: Integrate hash-based similarity computation and top-k block selection:
  - Hamming distance computation between query hash codes and cached key hash codes
  - Top-k block selection based on hash similarity scores
  - Dynamic block table updates for selective KV cache loading
  - Support for layer-wise sparsity ratios and adaptive sparsity patterns

- **Attention Metadata Management**: Modify attention metadata handling to support sparse attention:
  - Top-k block table management
  - Sequence length adjustments for sparse windows
  - Integration with existing attention metadata structures

The key integration points are
  - `vllm_ascend/attention/attention_v1.py`: Attention backend forr GQA 
  - `vllm_ascend/attention/mla_v1.py`: Attention backend for MLA


### 2. Hash Key Cache Allocation and Layout

To support efficient KVComp operation, modifications to hash key cache allocation and layout are necessary:

- **Hash Cache Initialization**: Implement hash cache allocation alongside standard KV cache:
  - Hash cache tensor allocation with appropriate shape and dtype (uint8 for packed hash codes)
  - Layer-wise hash cache management (supporting rollback and skip layers)
  - Integration with existing KV cache initialization logic

- **Cache Layout Optimization**: Design cache layout to support rapid hash code access:
  - Efficient hash code storage format (packed binary representation)
  - Cache layout compatible with block-based attention mechanisms
  - Memory-efficient hash cache management

- **Hash Code Caching**: Implement hash code caching during KV cache updates:
  - Hash code computation and storage in NPU
  - Possibly integration with existing `reshape_and_cache` operations
  - **KV Cache Offloading Support**: Full support for both **non-offloading** (HBM) and **offloading** scenarios (DRAM/SSD):
    - Non-offloading mode: Hash codes stored alongside KV cache in device memory for fast access
    - Offloading mode: Hash codes enable efficient selective loading of KV cache blocks from offloaded storage, significantly reducing memory transfer overhead
    - Seamless integration with vllm-ascend's offloading capabilities like [memfabric](https://gitcode.com/jamiecui/memfabric_hybrid) for optimal memory management

The key integration point is:
  - `initialize_kv_cache_tensors()` in `vllm_ascend/worker/model_runner_v1.py`: Hash key cache allocation and layout in KV cache management modules in 

### 3. Additional Changes

Other supporting changes include:

- **Configuration Management**: Add KVComp-specific configuration support:
  - Configuration file parsing and validation
  - Model-specific configuration presets
  - Runtime configuration options (top-k ratios, sparsity thresholds, etc.)

The key integration point is:
  - `AscendMetadata` for vllm-ascend's attention metadata
 

## Implementation Details

### Architecture Overview

KVComp operates through a three-stage process:

1. **Hash Encoding**: Convert attention keys and queries into compact hash codes using trainable hash functions
2. **Similarity Computation**: Use efficient hash-based similarity (Hamming distance) to identify relevant KV cache blocks
3. **Selective Loading**: Load only the top-k most relevant KV blocks for attention computation

The algorithm maintains three critical windows:
- **Initial Window**: First few blocks (always loaded)
- **Sparse Window**: Top-k selected blocks (dynamically chosen based on hash similarity)
- **Local Window**: Recent blocks (always loaded)

### Model Architecture Support

KVComp provides comprehensive support for diverse model architectures:

- **GQA Models** (e.g., Qwen3): 
  - Standard hash encoding based on query/key representations
  - Efficient handling of grouped query attention patterns
  - Optimized hash computation for GQA-specific attention mechanisms

- **MLA Models** (e.g., DeepSeek):
  - Dual-component hash encoding leveraging NoPE and RoPE
  - Specialized hash encoders for KV LoRA and QK RoPE components
  - Support for MLA-specific attention patterns and metadata structures

### KV Cache Management Modes

KVComp supports flexible KV cache management strategies:

- **Non-Offloading Mode**: 
  - Hash codes stored alongside KV cache in device memory
  - Fast hash-based similarity computation for in-memory cache blocks
  - Optimal for scenarios with sufficient device memory

- **Offloading Mode**:
  - Hash codes enable efficient selective loading from offloaded storage
  - Dramatically reduces memory transfer overhead by loading only top-k relevant blocks
  - Seamless integration with vllm-ascend's offloading infrastructure
  - Particularly beneficial for extremely long contexts where full KV cache cannot fit in device memory
  - **Distributed KV Cache Support**: In distributed architectures (e.g., PD disaggregation, mooncake/LMCache remote separation), KVComp's selective block loading significantly reduces communication bandwidth consumption and network latency by transferring only the most relevant KV cache blocks across network boundaries, rather than the entire cache


### Backward Compatibility

The implementation will:
- Make KVComp an optional feature (disabled by default and enabled with a system environment variablwe `ENABEL_KVCOMP_SPARSE_ATTENTION`)
- Maintain full backward compatibility with existing models, configurations and features
- Support gradual migration path for users

## Potential Benefits

- **Enhanced Memory Efficiency**: By compressing the KV cache through sparse attention, KVComp significantly reduces memory consumption during long-context inference, enabling support for longer sequences on the same hardware.

- **Improved Inference Performance**: The reduction in memory requirements and computational overhead leads to lower latency and higher throughput, enhancing overall inference performance for long-context scenarios.

- **Scalability**: The integration of KVComp enables vllm-ascend to handle longer contexts more effectively, improving scalability for large-scale applications requiring extensive contextual understanding.

- **Broad Model Support**: KVComp's support for both GQA models (e.g., Qwen3) and MLA models (e.g., DeepSeek) ensures wide applicability across different model architectures, making it a versatile solution for diverse use cases.

- **Flexible Deployment**: Support for both KV cache non-offloading and offloading scenarios provides deployment flexibility, allowing users to choose the optimal memory management strategy based on their hardware constraints and performance requirements.

- **Reduced Communication Overhead**: By enabling selective loading of only top-k relevant KV cache blocks, KVComp dramatically reduces communication bandwidth requirements and latency in distributed KV cache architectures (e.g., PD disaggregation, mooncake/LMCache remote separation), making distributed long-context inference more practical and scalable.


## Potential Drawbacks

- **Implementation Complexity**: Integrating KVComp requires substantial modifications to the attention backend and cache management systems, which may increase development complexity and maintenance overhead.

- **Model-Specific Configuration**: KVComp requires model-specific configuration files and tuning, which may require additional effort for each supported model architecture.

- **Accuracy-Perforamnce Trade-offs**: While KVComp maintains high accuracy (< 1% loss), there may be edge cases where accuracy degradation is more significant, requiring careful validation and testing.

- **Feature Addition Complexity**: KVComp is a sparese attention mechanism which could have accurancy loss, and thus affects other featurs like quantization and MTP. 



## Feedback Period

This RFC will be open for feedback until **January 31, 2026**, allowing the community to review and provide input on the proposed integration.

## CC List

- @wangxiyuan
- @MengqingCao
- @weijinqian0
- @Yikun
- @shen-shanshan

## Any Other Things

- We welcome community feedback and collaboration on this feature, particularly regarding:
  - Model coverage and configuration requirements
  - Performance optimization opportunities
  - Integration with other vllm-ascend features
  - Testing and validation strategies
  - Additional model architectures that could benefit from KVComp integration

- Related work and references:
  - [HATA Paper](https://aclanthology.org/2025.findings-acl.1275/): Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference
  - [RFC]: KV cache layout combining all layers per block (#4140): Related to cache layout optimizations
  - [RFC]: Refactor Attention module (#4629): Related to attention backend refactoring

