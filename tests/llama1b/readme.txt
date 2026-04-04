Llama 3.2 1B decode baseline (2026-04-01, B300, bf16, M=1)
Instruction correctness
---------------------------------------------------------
instruction              max_diff    mean_diff
rms_qkv_rope_append Q    1.0000      0.0892
rms_qkv_rope_append K    1.0000      0.0982
rms_qkv_rope_append V    0.5000      0.0590
attention_partial        0.0078      0.0006
o_proj_residual          1.0000      0.0539
down_proj_residual       2.0000      0.1068
rms_upgate_silu         64.0000      1.4320
rms_lm_head              0.5000      0.0001

End-to-end decode correctness (real HF weights, prompt="The cat sat on")
------------------------------------------------------------------------
MK vs PyTorch ref:  max_diff=0.0938  mean_diff=0.0145
MK vs HuggingFace:  max_diff=0.1250  mean_diff=0.0209
Top token agreement: MK='windows', ref='windows', HF='windows'

Instruction benchmarks (isolated, single instruction type)
----------------------------------------------------------
instruction              MK (us)    PyTorch (us)    roofline (us)    MK GB/s
rms_qkv_rope_append       67.4        146.9             1.57          187.0
attention_partial         188.5         60.9             0.07            2.8
o_proj_residual            44.3         17.4             1.05          189.6
rms_upgate_silu            71.7         92.9             8.39          936.3
down_proj_residual         42.5         20.1             4.20          790.6
rms_lm_head               119.9        116.2            65.70         4383.2

Full decode (16 layers + lm_head, seq_len=256)
----------------------------------------------
Megakernel:   7170.4 us  (345.9 GB/s)
PyTorch:      6826.1 us  (363.4 GB/s)
Roofline:      310.0 us
