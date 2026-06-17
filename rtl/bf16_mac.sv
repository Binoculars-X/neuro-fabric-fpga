// bf16_mac.sv — Synthesizable BF16 Multiply-Accumulate Unit
//
// BF16 = top 16 bits of IEEE 754 float32: [sign(1) | exponent(8) | mantissa(7)]
// Decode: zero-extend lower 16 bits → valid float32 (combinatorial)
// Encode: add 0x8000 (round-to-nearest) then take upper 16 bits
//
// Architecture:
//   Decode a_bf16, b_bf16 → a_fp32, b_fp32 (comb)
//   fp32_mul: a_fp32 * b_fp32 → product  (3-cycle latency)
//   Delay pipe: c_fp32 aligned to product output (3 cycles)
//   fp32_add: product + c_fp32_delayed → result  (4-cycle latency)
//
// LATENCY = 7 cycles (3 mul + 4 add).
//
// Ports:
//   clk         — clock
//   rst         — synchronous active-high reset
//   en          — retained for interface compatibility (not used internally;
//                 fp32_mul/fp32_add run free-running)
//   a_bf16      — 16-bit BF16 multiplicand
//   b_bf16      — 16-bit BF16 multiplier
//   c_fp32      — 32-bit FP32 accumulator input (chain or zero)
//   valid_in    — qualify input
//   result_fp32 — 32-bit FP32 result (7 cycles after valid_in)
//   result_bf16 — 16-bit BF16 encoded result
//   valid_out   — qualifies output
//
// Vivado-synthesizable: logic [N:0] only, no float types or simulation tasks.
// Target: Xilinx ZCU102 (UltraScale+ XCZU9EG), Vivado 2023.x.

`timescale 1ns/1ps

module bf16_mac #(
    parameter int LATENCY = 7   // fixed by submodule structure; parameter kept for interface compat
)(
    input  logic        clk,
    input  logic        rst,
    input  logic        en,         // retained for interface compat; unused internally

    input  logic [15:0] a_bf16,
    input  logic [15:0] b_bf16,
    input  logic [31:0] c_fp32,
    input  logic        valid_in,

    output logic [31:0] result_fp32,
    output logic [15:0] result_bf16,
    output logic        valid_out
);

    // -----------------------------------------------------------------------
    // Decode: BF16 → FP32 = zero-extend lower 16 bits (combinatorial)
    // -----------------------------------------------------------------------
    logic [31:0] a_fp32, b_fp32;
    always_comb begin
        a_fp32 = {a_bf16, 16'h0000};
        b_fp32 = {b_bf16, 16'h0000};
    end

    // -----------------------------------------------------------------------
    // Stage 1: fp32_mul — a_fp32 * b_fp32, latency = 3 cycles
    // -----------------------------------------------------------------------
    logic [31:0] mul_result;
    logic        mul_valid;

    fp32_mul u_mul (
        .clk      (clk),
        .rst      (rst),
        .valid_in (valid_in),
        .a        (a_fp32),
        .b        (b_fp32),
        .result   (mul_result),
        .valid_out(mul_valid)
    );

    // -----------------------------------------------------------------------
    // Delay c_fp32 by 3 cycles to align with mul_result
    // -----------------------------------------------------------------------
    logic [31:0] c_dly [0:2];

    always_ff @(posedge clk) begin
        if (rst) begin
            c_dly[0] <= 32'h0;
            c_dly[1] <= 32'h0;
            c_dly[2] <= 32'h0;
        end else begin
            c_dly[0] <= c_fp32;
            c_dly[1] <= c_dly[0];
            c_dly[2] <= c_dly[1];
        end
    end

    // -----------------------------------------------------------------------
    // Stage 2: fp32_add — product + c_fp32_delayed, latency = 4 cycles
    // -----------------------------------------------------------------------
    logic [31:0] add_result;

    fp32_add u_add (
        .clk      (clk),
        .rst      (rst),
        .valid_in (mul_valid),
        .a        (mul_result),
        .b        (c_dly[2]),
        .result   (add_result),
        .valid_out(valid_out)
    );

    // -----------------------------------------------------------------------
    // Encode result: FP32 → BF16 (round-to-nearest: add 0x8000, take upper 16)
    // -----------------------------------------------------------------------
    logic [31:0] rounded;
    always_comb begin
        rounded     = add_result + 32'h0000_8000;
        result_fp32 = add_result;
        result_bf16 = rounded[31:16];
    end

endmodule
