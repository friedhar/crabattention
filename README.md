# Crabattention - Multi-Flavour Attention Implementations In Rust

Attention algorithms are the core of modern generative models, including, but not limited to - transformers.

The Library provides optimized implementations of various Attention Algorithms, with the goal of supporting multiple backends.

Currently, the only supported target device is plain ol' CPU LLIR. 

## (Semi-Scientific) Benchmarks
> NOTE: The benchmarks we're done on a standard work machine, a M4 Macbok Pro. results obviously have high beta to varying hardware.
> NOTE: CrabAttention doesn't necessiraly aim to be as fast / faster than SOTA implementations like `FlashAttention`..
```
Model Name:	                    MacBook Pro
Model Identifier:	              Mac16,1
Chip:	                          Apple M4
Total Number of Cores:	        10 (4 performance and 6 efficiency)
Memory:	                        16 GB
System Firmware Version:       	11881.1.1
OS Loader Version:            	11881.1.1
```










