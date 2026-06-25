# CUDA kernels for fused operations
# Auto-detected and linked when CUDA compiler is available

add_library(olmo_kernels SHARED
    olmo_ops_autograd.cpp        # Autograd key registration (must be in shared lib — see file header)
    rms_norm.cu
    rope.cu
    silu_mul.cu
    topp_radix.cu              # fast-inference [7]
    fused_lm_head_sample.cu    # fast-inference [6]
    paged_attention.cu         # fast-inference [9]
    lm_head_gemv.cu            # fast-inference [11]
    flash_decode.cu            # fast-inference [12]
    sparse_attn_decode.cu      # SubQ Phase-1: fused content-selected SDPA
    quant_dequant.cu           # FP8 / INT4 AWQ device-resident dequant
    topk_argmax.cu             # K (fused GEMV+argmax, no logits materialize)
    fused_qkv_rope.cu          # G (fused QKV projection + reshape + RoPE)
    fused_qkv_rope_wmma.cu     # G + item 2: tensor-core (WMMA) variant
    fused_ffn.cu               # I (fused SwiGLU FFN macro)
    fused_ffn_wmma.cu          # I + item 2: tensor-core (WMMA) variant
    fused_ffn_tma.cu           # I + item W: TMA async-load variant (sm_90+)
    fused_lm_head_ce.cu        # A3 — fused LM-head + softmax-CE
    rms_norm_backward.cu       # B3 — fused RMSNorm backward
    # Host-side TMA descriptor builder. Lives HERE (not in olmo_cpp) because
    # make_tma_descriptor is called only by fused_ffn_tma.cu and by nothing in
    # olmo_cpp — keeping it in the static lib left it an orphan object that the
    # olmo_kernels shared lib could not resolve. Co-locate with its caller.
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/backend/tma_loads.cpp  # W (TMA descriptor)
)
target_include_directories(olmo_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include   # public headers: olmo_cpp/backend/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}              # kernel-local .cuh (cuda_reduce, mma_sync)
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${TORCH_INCLUDE_DIRS}
)
# CUDA::cuda_driver provides cuTensorMapEncodeTiled (driver API) used by
# tma_loads.cpp. cudart alone does not export it.
target_link_libraries(olmo_kernels PRIVATE ${TORCH_LIBRARIES} CUDA::cudart CUDA::cuda_driver)

# The kernels reference each other through the public headers, whose CUDA
# declarations are gated behind OLMO_HAS_CUDA_KERNELS (e.g. fused_ffn_tma.cu
# calls fused_ffn_wmma_cuda). This target obviously HAS the kernels, so the
# macro must be defined here too — matching olmo_cpp.
target_compile_definitions(olmo_kernels PRIVATE OLMO_HAS_CUDA_KERNELS=1)

# CUDA compilation flags: fast math + architecture-aware optimization
target_compile_options(olmo_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
    >
)
set_target_properties(olmo_kernels PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    PREFIX ""
    CUDA_SEPARABLE_COMPILATION ON
    # Single-GPU build: don't inherit PyTorch's fat-binary sm_50..sm_120 list.
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
