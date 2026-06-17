// synth_rom_test_top.sv -- Minimal wrapper to drive fp32_sqrt through Vivado synthesis.
// Purpose: verify that initial $readmemh("recipsqrt_rom.mem") infers RAMB18E2/RAMB36E2
// and produces no Synth 8-2898 ($readmemh path) warnings.
// Not intended for simulation or implementation -- synthesis checkpoint only.

`timescale 1ns/1ps
module synth_rom_test_top (
    input  logic        clk,
    input  logic        rst,
    input  logic        valid_in,
    input  logic [31:0] x_fp32,
    output logic [31:0] result_fp32,
    output logic        valid_out
);
    fp32_sqrt u_fp32_sqrt (
        .clk        (clk),
        .rst        (rst),
        .valid_in   (valid_in),
        .x_fp32     (x_fp32),
        .result_fp32(result_fp32),
        .valid_out  (valid_out)
    );
endmodule
