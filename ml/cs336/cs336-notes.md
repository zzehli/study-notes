## Lecture 2
* 6 (# data points) (# parameters) FLOPs per training step
* How long would it take to train a 70B parameter model on 15T tokens on 1024 H100s?
```
    total_flops = 6 * 70e9 * 15e12  
    h100_flop_per_sec = 1979e12 / 2
    mfu = 0.5
    flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24  
    days = total_flops / flops_per_day  
```
* What's the largest model that can you can train on 8 H100s using AdamW?
```
    h100_bytes = 80e9  
    bytes_per_parameter = 2 + 2 + (4 + 4)  # parameters (2), gradients (2), optimizer state (4 + 4) 
    num_parameters = (h100_bytes * 8) / bytes_per_parameter  
```
* in transformers tensors are rank 4:
```
    B = 32   # Batch size
    S = 16   # Sequence length
    H = 16   # Number of heads
    D = 64   # Hidden dimension per head
    x = torch.zeros(B, S, H, D)
```
* Q: transformer block: feed forward block and multi-headed attention
* elements of tensors are floating point numbers. fp32 is the default in scientific computing, but in deep learning, you can be sloppier.
* floating point:
  * fp16:  half precision, each tensor is 2 bytes; underflow if the number is too lose to 0
  * bf16: google develop this to address underflow; it has the dynamic range as fp32, only the worse resolution
  * mixed precision: use bf16 for parameters, activations and gradients and fp32 for optimizer states
  * fp8: standardized in 2022
  * fp4: introduced in 2025
* gpu
