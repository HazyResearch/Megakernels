To download and prepare:

```bash
MODEL_REPO=meta-llama/Llama-3.2-1B-Instruct

pip install torch sentencepiece tiktoken blobfile \
            safetensors huggingface_hub requests
python download.py --repo_id "$MODEL_REPO"
python convert_hf_checkpoint.py \
  --checkpoint_dir "checkpoints/$MODEL_REPO" \
  --model_name llama-3.2-1b-instruct
```

To run:

```bash
MODEL_REPO=meta-llama/Llama-3.2-1B-Instruct
python generate.py \
  --checkpoint_path checkpoints/$MODEL_REPO/model.pth \
  --prompt "Tell me a joke about cookies" \
  --max_new_tokens 100 \
  --num_samples 5 \
  --warmup 5 \
  --compile max-autotune \
  --pdl
```
