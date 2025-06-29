# MPS (Metal Performance Shaders) Reference

## Requirements for Apple Silicon
- Apple silicon (M1/M2) device
- macOS 12.6+ (13.0+ recommended) 
- arm64 Python version
- PyTorch 2.0 (recommended) or 1.13 (minimum)

## Basic MPS Usage
```python
import torch
# Check MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Move model to MPS
model = model.to(device)
```

## HuggingFace with MPS
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model with MPS
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = model.to("mps")

# For generation
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello", return_tensors="pt").to("mps")
outputs = model.generate(**inputs)
```

## Best Practices
- Use float16 dtype for better performance: `model.half()`
- Enable attention slicing for <64GB RAM: `pipe.enable_attention_slicing()`
- Avoid batch processing (can be unreliable)
- Set fallback: `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Do warmup pass for PyTorch 1.13: `_ = model(dummy_input)`

## Limitations
- Does not support NDArray sizes > 2^32
- Batch inference can be unreliable
- Performance sensitive to memory pressure
- Some PyTorch operations fall back to CPU

## For Prompt Generation
- Use smaller models (GPT-2, small Llama variants)
- Generate prompts in batches of 1 to avoid MPS batch issues
- Consider CPU fallback for complex operations
- Memory efficient with Apple's unified memory architecture