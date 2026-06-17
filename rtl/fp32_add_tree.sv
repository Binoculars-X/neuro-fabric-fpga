// fp32_add_tree.sv — Synthesizable IEEE 754 FP32 4-input adder tree (T=4)
//
// Architecture: 2-stage pipelined tree using fp32_add submodules.
//   Stage 1 (cycles 1-4): two fp32_add in parallel: s01 = in[0]+in[1], s23 = in[2]+in[3]
//   Stage 2 (cycles 5-8): one fp32_add: sum = s01 + s23
//
// LATENCY = 8 cycles (valid_out asserts 8 cycles after valid_in).
//
// C# reference: s01 = v[0]+v[1], s23 = v[2]+v[3], sum = s01+s23 (tree order, float32).
// Target: Xilinx ZCU102 (UltraScale+ XCZU9EG), Vivado 2023.x.

`timescale 1ns/1ps

module fp32_add_tree #(
    parameter int T = 4   // must be 4
)(
    input  logic            clk,
    input  logic            rst,
    input  logic            valid_in,
    input  logic [T*32-1:0] in_vec,
    output logic [31:0]     sum_out,
    output logic            valid_out
);

    // ── Stage 1: two parallel adds ───────────────────────────────────────────

    logic [31:0] s1_s01, s1_s23;
    logic        s1_valid;

    fp32_add u_add01 (
        .clk      (clk),
        .rst      (rst),
        .valid_in (valid_in),
        .a        (in_vec[  0 +: 32]),
        .b        (in_vec[ 32 +: 32]),
        .result   (s1_s01),
        .valid_out(s1_valid)           // both adds have same latency; use one valid
    );

    fp32_add u_add23 (
        .clk      (clk),
        .rst      (rst),
        .valid_in (valid_in),
        .a        (in_vec[ 64 +: 32]),
        .b        (in_vec[ 96 +: 32]),
        .result   (s1_s23),
        .valid_out()                   // tied to u_add01's valid_out — same latency
    );

    // ── Stage 2: sum the two stage-1 results ─────────────────────────────────

    fp32_add u_add_final (
        .clk      (clk),
        .rst      (rst),
        .valid_in (s1_valid),
        .a        (s1_s01),
        .b        (s1_s23),
        .result   (sum_out),
        .valid_out(valid_out)
    );

endmodule
