Llama 3.2 1B decode baseline (2026-04-04, B300, bf16, M=1)

instruction              max_diff    mean_diff
rms_qkv_rope_append Q    2.0000      0.0561
rms_qkv_rope_append K    1.0000      0.0617
rms_qkv_rope_append V    0.0000      0.0000
attention_partial        0.0078      0.0007
o_proj_residual          0.0156      0.0000
down_proj_residual       2.0000      0.2214
rms_upgate_silu         64.0000      1.4339
rms_lm_head              0.5000      0.0001
rms_lm_head_pipelined   0.5000      0.0001

End-to-end decode correctness
MK vs PyTorch ref:  max_diff=0.1562  mean_diff=0.0254
Top token: MK='doesn', ref='doesn'

Instruction benchmarks 
instruction              MK (us)    roofline (us)    MK GB/s
rms_qkv_rope_append      159.1         1.57           79.2
attention_partial         109.9         0.03            2.5
o_proj_residual            75.2         1.05          111.7
rms_upgate_silu           115.0         8.39          583.8
down_proj_residual         73.1         4.20          459.3
rms_lm_head                95.6        65.70         5499.6

Full decode (16 layers + lm_head, seq_len=128)
Megakernel:   1115.0 us  (2220.7 GB/s)
Roofline:      309.5 us
