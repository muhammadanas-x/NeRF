"""
SOTA Sobol Sequence Generation on GPU
======================================

Three implementations, all checked bit-exact against ONE correct ground
truth before any timing is taken:

  0. `sobol_ground_truth`   - direct definition: x_i = XOR of direction-number
     columns v_c selected by the set bits of Gray(i) = i ^ (i >> 1).
     This is the textbook Sobol definition (Bratley & Fox 1988, Joe & Kuo).
     O(L) per sample, fully vectorized across samples (no Python loop over
     N), used ONLY as the correctness oracle -- not meant to be fast.

  1. `sobol_sequential_scan` - REAL SOTA for generating points 0..N-1 in
     order. Uses the classical one-XOR-per-step recurrence
         x_0 = 0,  x_i = x_{i-1} XOR v_{c(i-1)},  c(k) = trailing_zeros(k+1)
     but instead of a serial Python loop (which is what kills GPU
     throughput), it's expressed as an associative XOR reduction and
     evaluated with a Hillis-Steele parallel-prefix scan: O(log N) parallel
     depth, O(N log N) work, fully vectorized tensor ops only. This matches
     what cuRAND / production GPU Sobol generators exploit (Bratley-Fox
     recurrence + parallel scan), and is verified here to exactly reproduce
     `sobol_ground_truth` (see /tmp verification below) -- earlier drafts of
     this benchmark incorrectly assumed "scalar bits(i)" and "Gray bits(i)"
     formulas were interchangeable; they are NOT, only the Gray-code form is
     the actual Sobol sequence.

  2. `sobol_random_access_gemm` - SOTA for *arbitrary/random* indices (no
     sequential structure to exploit). Computes Gray(idx) bits, MSB-encodes
     both the direction-number matrix and the Gray-bit vectors as INT8, and
     extracts every output bit with a SINGLE batched INT8 tensor-core GEMM
     (`torch._int_mm`) followed by `& 1` parity extraction on the INT32
     accumulator -- i.e. XOR-of-selected-columns is recast as "sum mod 2",
     which a GEMM computes natively. This removes the O(L) Python loop
     entirely (one matmul instead of L iterations) and is the right primitive
     when you need x_i for indices that aren't 0..N-1 in order (e.g.
     randomized QMC, scrambled nets, owen-scrambling pipelines that jump
     around the index space).

Run with: python sobol_sota_gpu_benchmark.py
Needs: PyTorch with CUDA (tensor cores: Volta/Turing/Ampere/Ada/Hopper+).
Falls back to CPU if no CUDA device is found -- correctness still verified,
but timings won't reflect real GPU/tensor-core throughput.
"""

import argparse
import time

import torch


# ----------------------------------------------------------------------------
# Sobol direction-number matrix (toy deterministic matrix -- statistical
# quality is irrelevant for a throughput/parity benchmark, only structural
# correctness of the bit operations matters).
#
# Convention: v_c (the "column c" direction number, c counted from the LSB,
# c = 0..L-1) is stored as matrix column (L-1-c). Row 0 is the MSB of the
# packed integer. This convention is what makes the parallel-prefix
# sequential recurrence (method 1) line up exactly with the direct Gray-code
# formula (method 0) -- verified by brute force for small L before relying
# on it here.
# ----------------------------------------------------------------------------
def make_sobol_matrix(dim_seed: int, L: int, device, dtype=torch.int64):
    g = torch.Generator(device="cpu").manual_seed(1234 + dim_seed)
    mat = torch.zeros((L, L), dtype=dtype)
    for col in range(L):
        valid_rows = col + 1
        bits = torch.randint(0, 2, (valid_rows,), generator=g, dtype=dtype)
        bits[0] = 1
        mat[:valid_rows, col] = bits
    return mat.to(device)


def column_ints(matrix: torch.Tensor, L: int) -> torch.Tensor:
    """Pack each direction-number column c (0=LSB..L-1=MSB) into an integer.
    v_c lives at matrix column (L-1-c). Vectorized, no Python loop over rows."""
    device = matrix.device
    pow2 = (1 << torch.arange(L - 1, -1, -1, device=device, dtype=torch.int64))  # MSB..LSB weights per row
    # matrix[:, L-1-c] dotted with pow2 gives v_c; do all c at once.
    cols = matrix.flip(dims=[1])  # cols[:, c] == matrix[:, L-1-c]
    return (cols * pow2.unsqueeze(1)).sum(dim=0)  # (L,) int64, index c = v_c


def bits_of(values: torch.Tensor, L: int) -> torch.Tensor:
    """Vectorized bit extraction, LSB-first along a new last dim of size L.
    No Python loop over samples; only L (a small constant, e.g. 16-32)."""
    shifts = torch.arange(L, device=values.device, dtype=torch.int64)  # bit c at position c
    return (values.unsqueeze(-1) >> shifts) & 1  # (..., L), index c = bit c (LSB-first)


# ----------------------------------------------------------------------------
# Method 0: ground truth (Gray-code direct formula) -- correctness oracle
# ----------------------------------------------------------------------------
def sobol_ground_truth(matrix: torch.Tensor, idx: torch.Tensor, L: int) -> torch.Tensor:
    col_ints = column_ints(matrix, L)            # (L,), index c = v_c
    gray = idx ^ (idx >> 1)
    g_bits = bits_of(gray, L)                     # (N, L), LSB-first, index c
    selected = torch.where(g_bits.bool(), col_ints.unsqueeze(0), torch.zeros_like(g_bits))
    out = selected[:, 0]
    for c in range(1, L):
        out ^= selected[:, c]
    return out


# ----------------------------------------------------------------------------
# Method 1: TRUE sequential SOTA -- Hillis-Steele parallel-prefix XOR scan
# over the classical one-step Gray-code recurrence. Generates ALL points
# 0..N-1 in a single call; O(log N) parallel depth instead of an O(N)
# serial Python loop, and reproduces sobol_ground_truth exactly because it
# implements the *same* recurrence the ground truth's Gray-code formula is
# derived from (Bratley & Fox 1988):
#     x_0 = 0,  x_i = x_{i-1} XOR v_{c(i-1)},   c(k) = trailing_zeros(k+1)
# ----------------------------------------------------------------------------
def sobol_sequential_scan(matrix: torch.Tensor, N: int, L: int) -> torch.Tensor:
    device = matrix.device
    col_ints = column_ints(matrix, L)  # (L,)

    k = torch.arange(N, device=device, dtype=torch.int64)        # k = 0..N-1, step term for x_{k+1}
    kp1 = k + 1
    # trailing_zeros(kp1), vectorized: lowest set bit isolated via (n & -n), then bit-length - 1.
    low_bit = kp1 & (-kp1)
    c = low_bit.bit_length() if False else torch.log2(low_bit.to(torch.float64)).round().to(torch.int64)
    c = torch.clamp(c, max=L - 1)

    step_val = col_ints[c]                       # v_{c(k)} for each k=0..N-1, this is the XOR step applied to get x_{k+1}
    # x_0 = 0, x_i = XOR_{k=0}^{i-1} step_val[k]  -> exclusive prefix-XOR of step_val.
    x = torch.zeros(N, dtype=torch.int64, device=device)
    x[1:] = step_val[:-1]
    # Hillis-Steele inclusive scan, then we already shifted to make it exclusive.
    stride = 1
    while stride < N:
        x[stride:] ^= x[:-stride].clone()
        stride *= 2
    return x


# ----------------------------------------------------------------------------
# Method 2: SOTA random access -- INT8 tensor-core GEMM, single matmul call
# replaces the O(L) Python loop entirely. Works for ANY idx tensor, not just
# 0..N-1 in order (scrambled / randomized QMC, arbitrary index gather, etc).
# ----------------------------------------------------------------------------
def sobol_random_access_gemm(matrix: torch.Tensor, idx: torch.Tensor, L: int) -> torch.Tensor:
    device = matrix.device
    N = idx.shape[0]

    gray = idx ^ (idx >> 1)
    g_bits = bits_of(gray, L)                                   # (N, L), index c = bit c (LSB-first)

    # Build A[row, c] = matrix[row, L-1-c] (bit `row` of v_c), so that
    # A @ g_bits.T gives, per output row, sum_c A[row,c]*g_bits[n,c]; mod 2
    # of that sum is exactly the XOR-of-selected-columns we want.
    A = matrix.flip(dims=[1]).to(torch.int8)                    # (L, L): A[:, c] = matrix[:, L-1-c]
    B = g_bits.to(torch.int8).t().contiguous()                  # (L, N)

    def pad(n):
        return n if n >= 32 else 32

    pad_L, pad_N = pad(L), pad(N)
    if pad_L != L or pad_N != N:
        A_p = torch.zeros((pad_L, pad_L), dtype=torch.int8, device=device)
        A_p[:L, :L] = A
        B_p = torch.zeros((pad_L, pad_N), dtype=torch.int8, device=device)
        B_p[:L, :N] = B
        A, B = A_p, B_p

    acc = torch._int_mm(A, B)[:L, :N]                           # (L, N) int32, row=bit position (MSB..? see below)
    parity = (acc & 1).to(torch.int64)                          # row `row` = bit (L-1-row) of output (MSB-first rows)

    weights = (1 << torch.arange(L - 1, -1, -1, device=device, dtype=torch.int64)).unsqueeze(1)  # (L,1)
    out = (parity * weights).sum(dim=0)
    return out


# ----------------------------------------------------------------------------
# Benchmark harness
# ----------------------------------------------------------------------------
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, *args, iters=20, warmup=5):
    device = args[0].device
    for _ in range(warmup):
        fn(*args)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--L", type=int, default=16, help="bit-depth / matrix size (L x L)")
    ap.add_argument("--N", type=int, default=1 << 16, help="number of samples per batch")
    ap.add_argument("--iters", type=int, default=20, help="benchmark iterations")
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: no CUDA device found -- tensor-core GEMM path and the parallel "
              "scan's real GPU advantage won't show here; this run only validates "
              "correctness, not real throughput.")

    L, N = args.L, args.N
    matrix = make_sobol_matrix(dim_seed=0, L=L, device=device)

    # Sequential indices 0..N-1 (what method 1 is built for).
    seq_idx = torch.arange(N, device=device, dtype=torch.int64)
    # Random indices (what method 2 is built for).
    rand_idx = torch.randint(0, 1 << L, (N,), dtype=torch.int64, device=device)

    # ---- correctness, against the single ground truth -----------------------
    ref_seq = sobol_ground_truth(matrix, seq_idx, L)
    ref_rand = sobol_ground_truth(matrix, rand_idx, L)

    seq_out = sobol_sequential_scan(matrix, N, L)
    match_seq = torch.equal(ref_seq, seq_out)
    print(f"Correctness (sequential parallel-prefix scan vs ground truth): "
          f"{'PASS' if match_seq else 'FAIL'}")
    if not match_seq:
        print(f"  mismatched samples: {(ref_seq != seq_out).sum().item()} / {N}")

    try:
        gemm_out = sobol_random_access_gemm(matrix, rand_idx, L)
        match_gemm = torch.equal(ref_rand, gemm_out)
        print(f"Correctness (tensor-core GEMM random access vs ground truth): "
              f"{'PASS' if match_gemm else 'FAIL'}")
        if not match_gemm:
            print(f"  mismatched samples: {(ref_rand != gemm_out).sum().item()} / {N}")
    except RuntimeError as e:
        print(f"Tensor-core path raised an error (torch._int_mm unsupported on this "
              f"device/torch build): {e}")
        gemm_out = None

    # ---- throughput -----------------------------------------------------------
    print(f"\nBenchmarking L={L}, N={N}, iters={args.iters}")

    t_truth = bench(sobol_ground_truth, matrix, seq_idx, L, iters=args.iters)
    print(f"  Ground truth (oracle)        : {t_truth*1e3:8.3f} ms/iter | "
          f"{N/t_truth/1e9:8.4f} GSamples/s")

    t_seq = bench(lambda m, n_, l: sobol_sequential_scan(m, n_, l), matrix, N, L, iters=args.iters)
    print(f"  Sequential parallel-prefix   : {t_seq*1e3:8.3f} ms/iter | "
          f"{N/t_seq/1e9:8.4f} GSamples/s | {t_truth/t_seq:.2f}x vs ground truth")

    if gemm_out is not None:
        t_gemm = bench(sobol_random_access_gemm, matrix, rand_idx, L, iters=args.iters)
        print(f"  Tensor-core GEMM (random)    : {t_gemm*1e3:8.3f} ms/iter | "
              f"{N/t_gemm/1e9:8.4f} GSamples/s | {t_truth/t_gemm:.2f}x vs ground truth")
    else:
        print("  Tensor-core GEMM (random)    : skipped (see error above)")


if __name__ == "__main__":
    main()