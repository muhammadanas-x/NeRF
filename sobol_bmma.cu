// sobol_bmma.cu
//
// Random-access Sobol generation via native single-bit (b1) AND-POPC
// Tensor Core MMA, instead of INT8/INT4 GEMM + software parity.
//
// STATUS: illustrative reference implementation, not a tuned kernel.
// Targets Turing / Ampere / Ada (sm_75 - sm_89), where mma.and.popc.b1
// executes on real Tensor Core hardware. On Hopper (sm_90+) the fast
// XOR-POPC binary path is deprecated performance-wise and requires an
// AND-based two-instruction emulation to recover throughput; INT4 mma
// on Hopper falls back to CUDA-core IMAD instead of Tensor Core execution.
// For a from-scratch Hopper target, profile against a plain INT8 mma.sync
// path before assuming the binary path wins there.
//
// Correctness: for boolean a_i, b_i in {0,1},
//     XOR-reduce(a_i AND b_i) == popcount(A & B) mod 2
// which is exactly the Sobol "select columns of the direction matrix by
// the set bits of Gray(idx), XOR them together" operation recast as a
// GF(2) matrix-vector product. The `& 1` parity extraction is unchanged
// from a software implementation -- only the reduction hardware changes.
//
// Build (Ampere example):
//   nvcc -arch=sm_80 -O3 sobol_bmma.cu -o sobol_bmma
//
// This uses the *documented, portable* experimental WMMA bmma_sync API
// rather than hand-derived inline PTX fragment layouts, since NVIDIA has
// not published the exact per-thread bit mapping for b1 fragments; it has
// only been reverse-engineered via microbenchmarking in prior work. Use
// CUTLASS's b1 GEMM (marked experimental) if you need a fully tuned,
// production-grade kernel with proper tiling/pipelining.

#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace nvcuda;
using namespace nvcuda::wmma::experimental;

// Native BMMA tile shape on Turing/Ampere/Ada: m8 x n8 x k128, b1 operands,
// s32 accumulator. K=128 bits per instruction is why Sobol's L<=32 makes
// this a structurally "thin-K" GEMM: at L=32 only 25% of one instruction's
// contraction width is real data, the rest is zero-padding. There is no
// mathematically valid way to fill the remainder with unrelated Sobol
// dimensions -- doing so would XOR-mix across dimensions and corrupt the
// result. The only free batching axis is N (samples).
constexpr int BMMA_M = 8;
constexpr int BMMA_N = 8;
constexpr int BMMA_K = 128;

// ---------------------------------------------------------------------
// Host-side: pack one dimension's direction-number matrix into bit-packed
// row-major words. Done once per dimension, offline -- this matrix is
// reused across the entire point set (potentially billions of samples),
// so it belongs in constant memory / registers, never reloaded.
// L <= 32 assumed; extend the word count for larger L.
// ---------------------------------------------------------------------
std::vector<uint32_t> pack_direction_matrix(const uint8_t* mat /* L x L, 0/1, row-major */, int L) {
    // BMMA fragment layout expects K=128 columns per row; we zero-pad the
    // unused 128-L columns. Real production code should confirm this
    // against the actual fragment-load helper rather than assuming a
    // trivial row-major mapping -- the per-thread mapping is not fully
    // documented by NVIDIA and has historically been reverse-engineered.
    std::vector<uint32_t> packed(L * (BMMA_K / 32), 0u);
    for (int r = 0; r < L; ++r) {
        for (int c = 0; c < L; ++c) {
            if (mat[r * L + c]) {
                packed[r * (BMMA_K / 32) + (c / 32)] |= (1u << (c % 32));
            }
        }
    }
    return packed;
}

// ---------------------------------------------------------------------
// Device: pack the Gray-code bit vector for a batch of sample indices
// directly into b1 fragment words, without ever materializing a
// per-sample (L,) byte array (that intermediate is the main removable
// inefficiency in the INT8 reference implementation).
// ---------------------------------------------------------------------
__device__ __forceinline__ uint32_t gray_bits_packed(uint64_t idx, int L) {
    uint64_t gray = idx ^ (idx >> 1);
    // bits 0..L-1 of `gray` map directly onto bits 0..L-1 of the packed
    // word; nothing else to do -- Gray-code computation is already O(1)
    // and bit-parallel, matching the original implementation.
    return static_cast<uint32_t>(gray & ((L < 32) ? ((1u << L) - 1u) : 0xFFFFFFFFu));
}

// ---------------------------------------------------------------------
// Kernel: one warp per (dimension, batch-of-8-samples) tile. Direction
// matrix fragment is loaded once into registers and reused across all
// N/8 tiles the warp processes -- it never touches global memory beyond
// the initial constant-memory read.
// ---------------------------------------------------------------------
__global__ void sobol_bmma_kernel(
    const uint32_t* __restrict__ packed_matrix, // (L, BMMA_K/32) words, this dimension
    const uint64_t* __restrict__ indices,        // (N,) arbitrary/random Sobol indices
    uint32_t* __restrict__ out,                  // (N,) output Sobol integers (L<=32 bits)
    int L,
    int N)
{
    const int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane       = threadIdx.x % 32;
    const int sample_base = warp_id * BMMA_N;
    if (sample_base >= N) return;

    wmma::fragment<wmma::matrix_a, BMMA_M, BMMA_N, BMMA_K, precision::b1, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, BMMA_M, BMMA_N, BMMA_K, precision::b1, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, BMMA_M, BMMA_N, BMMA_K, int32_t> frag_c;

    // A: direction matrix, resident for the whole kernel invocation.
    wmma::load_matrix_sync(frag_a, packed_matrix, BMMA_K);

    // B: pack up to BMMA_N Gray-bit vectors for this tile of samples.
    // In a fully tuned kernel this would be shared-memory staged and
    // built cooperatively across the warp; kept simple here for clarity.
    __shared__ uint32_t b_words[BMMA_N * (BMMA_K / 32)];
    if (lane < BMMA_N) {
        int s = sample_base + lane;
        uint32_t word = (s < N) ? gray_bits_packed(indices[s], L) : 0u;
        for (int w = 0; w < BMMA_K / 32; ++w) {
            b_words[lane * (BMMA_K / 32) + w] = (w == 0) ? word : 0u;
        }
    }
    __syncwarp();

    wmma::load_matrix_sync(frag_b, b_words, BMMA_K);
    wmma::fill_fragment(frag_c, 0);

    // The operation that actually replaces the O(L) XOR loop / INT8 GEMM:
    // native popcount(A & B) in one Tensor Core instruction.
    bmma_sync(frag_c, frag_a, frag_b, frag_c, bmmaBitOpAND, bmmaAccumulatorOpPOPC);

    // frag_c[row] now holds popcount(A_row & B_col) for each (row, col)
    // this warp is responsible for. Parity extraction is unchanged from
    // the reference implementation: take & 1, then reassemble the L
    // output bits into the final Sobol integer. Fragment-to-thread
    // mapping for the accumulator is implementation-defined; a full
    // kernel should use wmma::store_matrix_sync into shared memory and
    // do the bit reassembly there rather than reasoning about individual
    // per-thread accumulator elements directly.
    __shared__ int32_t c_out[BMMA_M * BMMA_N];
    wmma::store_matrix_sync(c_out, frag_c, BMMA_N, wmma::mem_row_major);
    __syncwarp();

    if (lane < BMMA_N) {
        int s = sample_base + lane;
        if (s < N) {
            uint32_t result = 0;
            for (int bit = 0; bit < L; ++bit) {
                int parity = c_out[bit * BMMA_N + lane] & 1;
                result |= (parity << bit);
            }
            out[s] = result;
        }
    }
}

// ---------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------
void sobol_random_access_bmma(
    const uint8_t* h_matrix, int L,
    const uint64_t* d_indices, uint32_t* d_out, int N)
{
    auto packed = pack_direction_matrix(h_matrix, L);
    uint32_t* d_packed;
    cudaMalloc(&d_packed, packed.size() * sizeof(uint32_t));
    cudaMemcpy(d_packed, packed.data(), packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int warps_needed = (N + BMMA_N - 1) / BMMA_N;
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks = (warps_needed + warps_per_block - 1) / warps_per_block;

    sobol_bmma_kernel<<<blocks, threads_per_block>>>(d_packed, d_indices, d_out, L, N);
    cudaFree(d_packed);
}

// Note: for a paper-quality kernel, next steps beyond this skeleton are:
//  1. Batch multiple *dimensions* via a strided/batched-GEMM outer loop
//     (see design doc, Q8-9) rather than a single wider instruction.
//  2. Profile against a plain INT8 mma.sync baseline on the actual target
//     architecture -- the thin-K structure (K=128 native vs L<=32 needed)
//     means this may end up bandwidth- or launch-bound rather than
//     Tensor-Core-bound; the win is not guaranteed without measurement.
//  3. On Hopper, benchmark the AND-based two-instruction binary emulation
//     against a straight INT8 path before choosing an architecture target.
