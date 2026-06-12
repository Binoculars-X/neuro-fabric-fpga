// FP32×BF16 Multiply-Accumulate (MAC) Unit — bf16w training path
//
// Matches AdamBF16WeightsAttentionCore forward pass:
//   activations: always FP32 (never quantized)
//   weights:     stored BF16, decoded on-the-fly to FP32 before multiply
//
// Operation (LATENCY=3 pipeline stages):
//   Stage 1: latch a_fp32, decode b_bf16 → fp32; latch c_fp32
//   Stage 2: multiply a_fp32 * b_fp32 (FP32×FP32 after decode)
//   Stage 3: accumulate product + c_fp32
//
// Ports:
//   a_fp32   — 32-bit FP32 activation input
//   b_bf16   — 16-bit BF16 weight input (decoded to FP32 in stage 1)
//   c_fp32   — 32-bit FP32 accumulator input (chain or zero)
//   valid_in — qualify input
//   result_fp32 — 32-bit FP32 accumulated result (LATENCY cycles later)
//   valid_out   — qualifies output

`timescale 1ns/1ps

module bf16w_mac #(
    parameter int LATENCY = 3
)(
    input  logic        clk,
    input  logic        rst,
    input  logic        en,

    input  logic [31:0] a_fp32,    // FP32 activation
    input  logic [15:0] b_bf16,    // BF16 weight
    input  logic [31:0] c_fp32,    // FP32 accumulator input
    input  logic        valid_in,

    output logic [31:0] result_fp32,
    output logic        valid_out
);

    // -----------------------------------------------------------------------
    // Decode: BF16 weight → FP32 = zero-extend lower 16 bits
    // -----------------------------------------------------------------------
    logic [31:0] b_fp32;
    always_comb b_fp32 = {b_bf16, 16'h0000};

    // -----------------------------------------------------------------------
    // Pipeline registers
    // -----------------------------------------------------------------------
    logic [31:0] s1_a, s1_b, s1_c;
    logic        s1_v;

    logic [31:0] s2_prod, s2_c;
    logic        s2_v;

    logic [31:0] s3_result;
    logic        s3_v;

    shortreal sr_a, sr_b, sr_prod, sr_c, sr_sum;

    always_ff @(posedge clk) begin
        if (rst) begin
            s1_v <= 0; s2_v <= 0; s3_v <= 0;
        end else if (en) begin
            // Stage 1: latch; b decoded combinatorially above
            s1_a <= a_fp32;
            s1_b <= b_fp32;
            s1_c <= c_fp32;
            s1_v <= valid_in;

            // Stage 2: FP32 × FP32 multiply
            sr_a    = $bitstoshortreal(s1_a);
            sr_b    = $bitstoshortreal(s1_b);
            sr_prod = sr_a * sr_b;
            s2_prod <= $shortrealtobits(sr_prod);
            s2_c    <= s1_c;
            s2_v    <= s1_v;

            // Stage 3: accumulate
            sr_prod = $bitstoshortreal(s2_prod);
            sr_c    = $bitstoshortreal(s2_c);
            sr_sum  = sr_prod + sr_c;
            s3_result <= $shortrealtobits(sr_sum);
            s3_v      <= s2_v;
        end
    end

    always_comb begin
        result_fp32 = s3_result;
        valid_out   = s3_v;
    end

endmodule
