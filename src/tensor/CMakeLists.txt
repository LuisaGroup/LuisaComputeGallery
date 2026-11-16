# GENERATED FILE - do not edit.
# Source of truth: scripts/gallery_targets.json
# Regenerate with: python scripts/gen_build.py

# subcommand tool that also runs a 36-case tile self-test; the CUDA backend needs the TIRx bridge for tile::compile
gallery_target(example_tensor
    SOURCES "main.cpp" "tensor_kernels.cpp" "kernel_*.cpp" "poly_fit.cpp" "poly_fit_kernels.cpp" "linear_regression.cpp" "linear_regression_kernels.cpp" "mlp.cpp" "mnist.cpp" "cnn_inference.cpp" "cnn_kernels.cpp" "rnn.cpp" "tensor_basics.cpp"
    DEPS lc-tile
    HEADLESS
    SKIP_BACKENDS cuda
)
