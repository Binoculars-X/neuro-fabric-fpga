// FP32 scalar divide: result = a / b  (combinatorial)
//
// Uses shortreal division — XSim promotes to double internally.
// C# reference: (float)((double)a / (double)b)

`timescale 1ns/1ps

module fp32_div (
    input  logic [31:0] a_fp32,
    input  logic [31:0] b_fp32,
    output logic [31:0] result_fp32
);

    shortreal a, b;

    always_comb begin
        a           = $bitstoshortreal(a_fp32);
        b           = $bitstoshortreal(b_fp32);
        result_fp32 = $shortrealtobits(a / b);
    end

endmodule
