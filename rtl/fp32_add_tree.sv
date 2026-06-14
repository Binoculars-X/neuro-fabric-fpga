// FP32 4-input reduction add tree: sum_out = in[0] + in[1] + in[2] + in[3]
//
// Architecture: combinatorial 2-stage shortreal tree
//   Stage 1: s01 = in[0]+in[1],  s23 = in[2]+in[3]
//   Stage 2: sum = s01 + s23
//
// XSim note: shortreal arithmetic uses true IEEE 754 single precision.
// C# reference: s01=v0+v1; s23=v2+v3; sum=s01+s23  (native float32)
//
// Parameter: T must be 4 for this implementation.

`timescale 1ns/1ps

module fp32_add_tree #(
    parameter int T = 4
)(
    input  logic [T*32-1:0] in_vec,
    output logic [31:0]     sum_out
);

    shortreal v0, v1, v2, v3;
    logic [31:0] s01_bits, s23_bits;

    always_comb begin
        v0       = $bitstoshortreal(in_vec[  0 +: 32]);
        v1       = $bitstoshortreal(in_vec[ 32 +: 32]);
        v2       = $bitstoshortreal(in_vec[ 64 +: 32]);
        v3       = $bitstoshortreal(in_vec[ 96 +: 32]);
        s01_bits = $shortrealtobits(v0 + v1);   // flush to 32-bit, forces IEEE 754 round
        s23_bits = $shortrealtobits(v2 + v3);   // flush to 32-bit
        sum_out  = $shortrealtobits($bitstoshortreal(s01_bits) + $bitstoshortreal(s23_bits));
    end

endmodule
