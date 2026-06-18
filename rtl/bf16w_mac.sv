// bf16w_mac.sv — Synthesizable FP32×BF16 Multiply-Accumulate Unit (bf16w training path)
//
// Matches AdamBF16WeightsAttentionCore forward pass:
//   activations: FP32 (a_fp32, never quantized)
//   weights:     stored BF16, decoded on-the-fly to FP32 before multiply
//
// Architecture:
//   Decode b_bf16 → b_fp32 (combinatorial, zero-extend lower 16 bits)
//   fp32_mul: a_fp32 * b_fp32 → product  (3-cycle latency)
//   Delay pipe: c_fp32 aligned to product output (3 cycles)
//   fp32_add: product + c_fp32_delayed → result  (4-cycle latency)
//
// LATENCY = 7 cycles (3 mul + 4 add).
//
// Ports:
//   clk         — clock
//   rst         — synchronous active-high reset
//   en          — retained for interface compatibility (unused internally)
//   a_fp32      — 32-bit FP32 activation input
//   b_bf16      — 16-bit BF16 weight input
//   c_fp32      — 32-bit FP32 accumulator input (chain or zero)
//   valid_in    — qualify input
//   result_fp32 — 32-bit FP32 result (7 cycles after valid_in)
//   valid_out   — qualifies output
//
// Vivado-synthesizable: logic [N:0] only, no float types or simulation tasks.
// Target: Xilinx ZCU102 (UltraScale+ XCZU9EG), Vivado 2023.x.

`timescale 1ns/1ps

module bf16w_mac #(
    parameter int LATENCY = 7   // fixed by submodule structure; parameter kept for interface compat
)(
    input  logic        clk,
    input  logic        rst,
    input  logic        en,         // retained for interface compat; unused internally

    input  logic [31:0] a_fp32,    // FP32 activation
    input  logic [15:0] b_bf16,    // BF16 weight
    input  logic [31:0] c_fp32,    // FP32 accumulator input
    input  logic        valid_in,

    output logic [31:0] result_fp32,
    output logic        valid_out
);

    // -----------------------------------------------------------------------
    // Decode: BF16 weight → FP32 = zero-extend lower 16 bits (combinatorial)
    // -----------------------------------------------------------------------
    logic [31:0] b_fp32;
    always_comb b_fp32 = {b_bf16, 16'h0000};

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
    fp32_add u_add (
        .clk      (clk),
        .rst      (rst),
        .valid_in (mul_valid),
        .a        (mul_result),
        .b        (c_dly[2]),
        .result   (result_fp32),
        .valid_out(valid_out)
    );

endmodule

