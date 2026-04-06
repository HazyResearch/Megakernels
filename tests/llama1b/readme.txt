Llama 3.2 1B decode baseline (2026-04-04, B300, bf16, M=1)

instruction              max_diff    mean_diff
rms_qkv_rope_append Q    2.0000      0.056127
rms_qkv_rope_append K    1.0000      0.061733
rms_qkv_rope_append V    0.0000      0.000000
attention_partial        0.007812   0.000659
o_proj_residual          0.015625   0.000011
down_proj_residual       2.000000    0.221401
rms_upgate_silu         64.000000    1.433940
rms_lm_head              0.5000      0.000077
rms_lm_head_pipelined   0.5000      0.000077

End-to-end decode correctness
MK vs PyTorch ref:  max_diff=0.1406  mean_diff=0.0250
Top token: MK='doesn', ref='doesn'

Instruction benchmarks 
instruction              MK (us)    roofline (us)    MK GB/s
rms_qkv_rope_append      171.3         1.57           73.6
attention_partial        119.3         0.03            2.3
o_proj_residual           80.1         1.05          104.9
rms_upgate_silu           121.0         8.39          555.0
down_proj_residual         78.8         4.20          426.0
rms_lm_head               103.9        65.70         5056.3

Full decode (16 layers + lm_head, seq_len=128)
Megakernel:   992.7 us  (2494.3 GB/s)
Roofline:      309.5 us
