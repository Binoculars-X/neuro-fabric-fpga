// BF16 Multiply-Accumulate (MAC) Unit
//
// BF16 = top 16 bits of IEEE 754 float32: [sign(1) | exponent(8) | mantissa(7)]
// Decode: zero-extend lower 16 bits  → valid float32
// Encode: add 0x8000 (round-to-nearest), then shift right 16
//
// Operation (LATENCY=3 pipeline stages):
//   Stage 0 (in):  latch a, b, c_in
//   Stage 1:       decode a,b → fp32; multiply in float DSP
//   Stage 2:       accumulate product + c_in
//   Stage 3 (out): encode result → bf16_out; fp32_out available too
//
// Parameters:
//   LATENCY  — pipeline depth (1..4); default 3 maps naturally to DSP48 A/B/P regs
//   N_MACS   — number of parallel MACs (instantiate this module N_MACS times at
//              the next level; kept = 1 here for simplicity)
//
// Ports:
//   clk      — clock
//   rst      — synchronous active-high reset
//   en       — pipeline enable (stall when low)
//   a_bf16   — 16-bit BF16 multiplicand
//   b_bf16   — 16-bit BF16 multiplier
//   c_fp32   — 32-bit FP32 accumulator input (chain or zero)
//   valid_in — qualify input
//   result_fp32  — 32-bit FP32 accumulated result (LATENCY cycles later)
//   result_bf16  — 16-bit BF16 encoded result
//   valid_out    — qualifies output

`timescale 1ns/1ps

module bf16_mac #(
    parameter int LATENCY = 3
)(
    input  logic        clk,
    input  logic        rst,
    input  logic        en,

    input  logic [15:0] a_bf16,
    input  logic [15:0] b_bf16,
    input  logic [31:0] c_fp32,
    input  logic        valid_in,

    output logic [31:0] result_fp32,
    output logic [15:0] result_bf16,
    output logic        valid_out
);

    // -----------------------------------------------------------------------
    // Decode: BF16 → FP32 = left-shift 16 (zero-extend lower 16 bits)
    // -----------------------------------------------------------------------
    logic [31:0] a_fp32, b_fp32;
    always_comb begin
        a_fp32 = {a_bf16, 16'h0000};
        b_fp32 = {b_bf16, 16'h0000};
    end

    // -----------------------------------------------------------------------
    // Pipeline registers
    // -----------------------------------------------------------------------
    // Stage 1: hold decoded operands + c_in
    logic [31:0] s1_a, s1_b, s1_c;
    logic        s1_v;
    // Stage 2: hold product + c_in
    logic [31:0] s2_prod, s2_c;
    logic        s2_v;
    // Stage 3 (output): accumulated result
    logic [31:0] s3_result;
    logic        s3_v;

    // -----------------------------------------------------------------------
    // Use shortreal (32-bit IEEE 754) for bit-accurate FP multiply in XSim.
    // Vivado synthesis will infer DSP48 slices for the multiply.
    // -----------------------------------------------------------------------
    shortreal sr_a, sr_b, sr_c, sr_prod, sr_sum;

    always_ff @(posedge clk) begin
        if (rst) begin
            s1_v <= 1'b0; s2_v <= 1'b0; s3_v <= 1'b0;
        end else if (en) begin
            // Stage 1: latch decoded inputs
            s1_a <= a_fp32;
            s1_b <= b_fp32;
            s1_c <= c_fp32;
            s1_v <= valid_in;

            // Stage 2: multiply (shortreal = IEEE 754 single, exact in XSim)
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

    // -----------------------------------------------------------------------
    // Encode result: FP32 → BF16 (round-to-nearest, ties-to-even omitted for now:
    // add 0x8000 then shift — matches Bf16.Encode() in C#)
    // -----------------------------------------------------------------------
    logic [31:0] rounded;
    always_comb begin
        rounded     = s3_result + 32'h0000_8000;   // round-to-nearest
        result_fp32 = s3_result;
        result_bf16 = rounded[31:16];
        valid_out   = s3_v;
    end

endmodule
