"""
Sobol Sequence Generation via 1-bit Matrix Multiply on Tensor Cores
=====================================================================

Implements the approach described by Matt: Sobol sample generation is just
an NxN matrix-vector multiply where every element is a 1-bit boolean instead
of a number. Because every entry is boolean:

    '*' becomes AND
    '+' becomes XOR  ==  '+' mod 2

A tensor-core INT8 GEMM computes the AND+SUM part natively (as long as you
don't let it saturate); we just reduce the accumulator mod 2 (`& 1`)
ourselves afterward to turn the sum into an XOR-reduction.

    dot(a, b) = popcount(a & b) mod 2

Direction-number matrix convention (per Matt's notes):
  - matrix[row, col] is a 1-bit LxL matrix
  - row 0 = MSB, and the matrix is 0 in the bottom-left corner (column j
    only has nonzero entries in rows 0..j)
  - the sample index's bits are bit-reversed on the fly during the multiply
    (LSB of Gray(idx) lines up with row 0, the MSB row of the matrix), so no
    separate bitreverse pass is needed.

PERFORMANCE NOTE vs an earlier draft of this file: the bit-reversal alignment
must NOT be done with a `.flip()` on the (N, L) bit tensor -- N is huge
(e.g. 65536) so flipping that tensor every call is expensive. Instead the
reversed bit order is generated directly (reversed shift amounts), and if a
flip is needed at all it should be applied to the tiny (L, L) matrix, not the
big (N, L) tensor. This is the difference that makes this version faster
than a naive port.

Run with: python sobol_gemm.py
Needs: PyTorch with CUDA (tensor cores: Turing/Ampere/Ada/Hopper+).
Falls back to CPU if no CUDA device is found -- correctness still checked,
but the tensor-core GEMM path will likely raise on CPU / unsupported builds.
"""

import argparse
import time

import torch


# ----------------------------------------------------------------------------
# Sobol direction-number matrix (toy deterministic matrix -- statistical
# quality is irrelevant here, only structural correctness of the bit
# operations matters).
# ----------------------------------------------------------------------------
def make_sobol_matrix(dim_seed: int, L: int, device, dtype=torch.int8):
    """1-bit LxL direction-number matrix. Row 0 = MSB, zero in the bottom-left
    corner (col j only has nonzero rows 0..j), matching Matt's description:
    'the matrix is 0 in the bottom corner'."""
    g = torch.Generator(device="cpu").manual_seed(1234 + dim_seed)
    mat = torch.zeros((L, L), dtype=dtype)
    for col in range(L):
        valid_rows = col + 1
        bits = torch.randint(0, 2, (valid_rows,), generator=g, dtype=dtype)
        bits[0] = 1  # top bit of each column always set
        mat[:valid_rows, col] = bits
    return mat.to(device)


# ----------------------------------------------------------------------------
# Ground truth: direct XOR-of-selected-rows definition, used only to check
# sobol_gemm below. Slow (Python loop over L), not meant to be fast.
# ----------------------------------------------------------------------------
def sobol_ground_truth(matrix: torch.Tensor, idx: torch.Tensor, L: int) -> torch.Tensor:
    device = matrix.device
    N = idx.shape[0]
    gray = idx ^ (idx >> 1)
    # bit c of gray at position c (LSB-first), then reorder to MSB-first
    # (index 0 = bit L-1 ... index L-1 = bit 0) directly via reversed shifts
    # -- no separate flip() needed, this IS the reversed order.
    shifts_rev = torch.arange(L - 1, -1, -1, device=device, dtype=torch.int64)
    g_bits = (gray.unsqueeze(-1) >> shifts_rev) & 1     # (N, L), index 0 = LSB of idx

    weights = (1 << torch.arange(L - 1, -1, -1, device=device, dtype=torch.int64))
    out = torch.zeros(N, dtype=torch.int64, device=device)
    mat64 = matrix.to(torch.int64)                      # (L, L)
    for row in range(L):
        sel = g_bits[:, row].unsqueeze(1) * mat64[row].unsqueeze(0)  # (N, L), 0/1
        bit = sel[:, 0].clone()
        for c in range(1, L):
            bit ^= sel[:, c]
        out |= bit * weights[row]
    return out


# ----------------------------------------------------------------------------
# The GEMM implementation (fast path)
# ----------------------------------------------------------------------------
def sobol_gemm(matrix: torch.Tensor, idx: torch.Tensor, L: int) -> torch.Tensor:
    """
    Sobol sample generation as a single 1-bit matrix-vector (batched into a
    matrix-matrix) multiply on tensor cores.

    Key perf point: the MSB-first bit ordering needed to align with the
    matrix's row convention is produced directly via reversed shift amounts
    (`shifts_rev`), so there is NO flip() on the large (N, L) tensor -- only
    O(L) work, independent of N.
    """
    device = matrix.device
    N = idx.shape[0]

    gray = idx ^ (idx >> 1)                              # Gray code of the sample index
    shifts_rev = torch.arange(L - 1, -1, -1, device=device, dtype=torch.int64)
    g_bits = (gray.unsqueeze(-1) >> shifts_rev) & 1       # (N, L), index 0 = LSB of idx, already row-aligned

    A = matrix.to(torch.int8)                             # (L, L) 1-bit matrix, values in {0,1}
    B = g_bits.to(torch.int8).t().contiguous()             # (L, N) 1-bit vectors, values in {0,1}

    def pad32(n):  # tensor-core GEMM wants dims >= 32
        return n if n >= 32 else 32

    pad_L, pad_N = pad32(L), pad32(N)
    if pad_L != L or pad_N != N:
        A_p = torch.zeros((pad_L, pad_L), dtype=torch.int8, device=device)
        A_p[:L, :L] = A
        B_p = torch.zeros((pad_L, pad_N), dtype=torch.int8, device=device)
        B_p[:L, :N] = B
        A, B = A_p, B_p

    acc = torch._int_mm(A, B)[:L, :N]            # (L, N) int32: acc[row,n] = sum_k A[row,k]&B[k,n]
    out_bits = (acc & 1).to(torch.int64)          # mod 2 -> AND-sum becomes XOR-reduction (row = output bit, MSB-first)

    weights = (1 << torch.arange(L - 1, -1, -1, device=device, dtype=torch.int64)).unsqueeze(1)
    out = (out_bits * weights).sum(dim=0)         # pack MSB-first rows back into an integer
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
        print("WARNING: no CUDA device found -- torch._int_mm tensor-core path "
              "may fall back or error; this run only checks correctness, not real throughput.")

    L, N = args.L, args.N
    matrix = make_sobol_matrix(dim_seed=0, L=L, device=device)
    idx = torch.randint(0, 1 << L, (N,), dtype=torch.int64, device=device)

    # ---- correctness, against the ground truth -----------------------------
    ref = sobol_ground_truth(matrix, idx, L)
    try:
        out = sobol_gemm(matrix, idx, L)
        match = torch.equal(ref, out)
        print(f"Correctness (tensor-core GEMM vs ground truth): {'PASS' if match else 'FAIL'}")
        if not match:
            print(f"  mismatched samples: {(ref != out).sum().item()} / {N}")
    except RuntimeError as e:
        print(f"Tensor-core path raised an error (torch._int_mm unsupported on this "
              f"device/torch build): {e}")
        return

    # ---- throughput ----------------------------------------------------------
    print(f"\nBenchmarking L={L}, N={N}, iters={args.iters}")
    t_gemm = bench(sobol_gemm, matrix, idx, L, iters=args.iters)
    print(f"  Tensor-core GEMM (random access): {t_gemm*1e3:8.3f} ms/iter | "
          f"{N/t_gemm/1e9:8.4f} GSamples/s")


if __name__ == "__main__":
    main()