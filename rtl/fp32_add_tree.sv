// FP32 4-input reduction add tree: sum_out = in[0] + in[1] + in[2] + in[3]
//
// Architecture: combinatorial 2-stage shortreal tree
//   Stage 1: s01 = in[0]+in[1],  s23 = in[2]+in[3]
//   Stage 2: sum = s01 + s23
//
// XSim note: shortreal arithmetic promotes to double per operation, matching
// ReferenceExactHardwareMode: (float)((double)x op (double)y).
// C# reference: s01=(float)((double)e0+(double)e1); s23=(float)((double)e2+(double)e3);
//               sum=(float)((double)s01+(double)s23)
//
// Parameter: T must be 4 for this implementation.

`timescale 1ns/1ps

module fp32_add_tree #(
    parameter int T = 4
)(
    input  logic [T*32-1:0] in_vec,
    output logic [31:0]     sum_out
);

    shortreal v0, v1, v2, v3, s01, s23;

    always_comb begin
        v0      = $bitstoshortreal(in_vec[  0 +: 32]);
        v1      = $bitstoshortreal(in_vec[ 32 +: 32]);
        v2      = $bitstoshortreal(in_vec[ 64 +: 32]);
        v3      = $bitstoshortreal(in_vec[ 96 +: 32]);
        s01     = v0 + v1;
        s23     = v2 + v3;
        sum_out = $shortrealtobits(s01 + s23);
    end

endmodule
