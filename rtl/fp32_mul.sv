// fp32_mul.sv — Synthesizable IEEE 754 FP32 multiplier
//
// Pipeline: 3 stages, LATENCY = 3 cycles.
// Vivado-synthesizable — logic [31:0] only, no float types or simulation system tasks.
// The 24×24-bit unsigned multiply maps to DSP48E2 slices on UltraScale+.
// All signals at module scope — no automatic/inline variable declarations.
//
// Special cases (IEEE 754 compliant):
//   NaN  input          → quiet NaN  (0x7FC00000)
//   Inf × non-zero      → ±Inf
//   Inf × 0             → quiet NaN  (0x7FC00000)
//   zero × anything     → ±0
//   Overflow (exp≥255)  → ±Inf
//   Underflow (exp≤0)   → ±0  (flush-to-zero; no subnormal output)
//   Denormal input      → treated as ±0  (flush-to-zero input)
//
// Rounding: round-to-nearest, ties-to-even.
//
// C# reference: (float)((double)a * (double)b)  matches to <=1 ULP.
// XSim tolerance: VsSoftwareRelTol = 0.01% (XSimCollection.cs).
// Target: Xilinx ZCU102 (UltraScale+ XCZU9EG).

`timescale 1ns/1ps

module fp32_mul (
    input  logic        clk,
    input  logic        rst,
    input  logic        valid_in,
    input  logic [31:0] a,          // IEEE 754 FP32
    input  logic [31:0] b,          // IEEE 754 FP32
    output logic [31:0] result,     // IEEE 754 FP32
    output logic        valid_out
);

    // =========================================================================
    // Stage 0 — combinatorial unpack (all signals at module scope)
    // =========================================================================

    // Intermediate unpack wires
    logic        c0_sa, c0_sb;
    logic [7:0]  c0_ea, c0_eb;
    logic [22:0] c0_ma, c0_mb;
    logic        c0_a_nan, c0_b_nan, c0_a_inf, c0_b_inf, c0_a_zero, c0_b_zero;

    // Stage 0 outputs (registered into Stage 1)
    logic        s0_sign;
    logic signed [9:0] s0_exp;   // 10-bit signed: range -127..381
    logic [23:0] s0_frac_a;
    logic [23:0] s0_frac_b;
    logic        s0_nan, s0_inf, s0_zero;

    always_comb begin
        c0_sa = a[31];      c0_sb = b[31];
        c0_ea = a[30:23];   c0_eb = b[30:23];
        c0_ma = a[22:0];    c0_mb = b[22:0];

        c0_a_nan  = (c0_ea == 8'hFF) & (c0_ma != 23'h0);
        c0_b_nan  = (c0_eb == 8'hFF) & (c0_mb != 23'h0);
        c0_a_inf  = (c0_ea == 8'hFF) & (c0_ma == 23'h0);
        c0_b_inf  = (c0_eb == 8'hFF) & (c0_mb == 23'h0);
        c0_a_zero = (c0_ea == 8'h00);
        c0_b_zero = (c0_eb == 8'h00);

        s0_sign   = c0_sa ^ c0_sb;
        s0_exp    = 10'(signed'({1'b0, c0_ea})) + 10'(signed'({1'b0, c0_eb})) - 10'sd127;
        s0_frac_a = c0_a_zero ? 24'h000000 : {1'b1, c0_ma};
        s0_frac_b = c0_b_zero ? 24'h000000 : {1'b1, c0_mb};
        s0_nan    = c0_a_nan | c0_b_nan | (c0_a_inf & c0_b_zero) | (c0_b_inf & c0_a_zero);
        s0_inf    = (c0_a_inf | c0_b_inf) & ~c0_a_nan & ~c0_b_nan;
        s0_zero   = c0_a_zero | c0_b_zero;
    end

    // =========================================================================
    // Stage 1 — register unpacked operands
    // =========================================================================

    logic        s1_sign, s1_nan, s1_inf, s1_zero, s1_valid;
    logic signed [9:0] s1_exp;
    logic [23:0] s1_frac_a, s1_frac_b;

    always_ff @(posedge clk) begin
        if (rst) begin
            s1_valid <= 1'b0;
        end else begin
            s1_sign  <= s0_sign;
            s1_exp   <= s0_exp;
            s1_frac_a <= s0_frac_a;
            s1_frac_b <= s0_frac_b;
            s1_nan   <= s0_nan;
            s1_inf   <= s0_inf;
            s1_zero  <= s0_zero;
            s1_valid <= valid_in;
        end
    end

    // =========================================================================
    // Stage 2 — 24×24 multiply (maps to DSP48E2 on UltraScale+)
    // =========================================================================

    logic [47:0] s2_prod;
    logic        s2_sign, s2_nan, s2_inf, s2_zero, s2_valid;
    logic signed [9:0] s2_exp;

    always_ff @(posedge clk) begin
        if (rst) begin
            s2_valid <= 1'b0;
        end else begin
            s2_prod  <= s1_frac_a * s1_frac_b;   // 24×24 → 48-bit unsigned
            s2_sign  <= s1_sign;
            s2_exp   <= s1_exp;
            s2_nan   <= s1_nan;
            s2_inf   <= s1_inf;
            s2_zero  <= s1_zero;
            s2_valid <= s1_valid;
        end
    end

    // =========================================================================
    // Stage 3 — normalize, round, pack (all signals at module scope)
    // =========================================================================
    //
    // Product is 48 bits: 1.mant × 1.mant = 1x.mant  (result in [1, 4))
    //   bit47 = 1 → result ≥ 2: shift right 1, exp+1
    //   bit47 = 0 → result in [1,2): no shift

    logic               c3_norm;
    logic [22:0]        c3_mant_raw;
    logic               c3_round_bit, c3_sticky;
    logic signed [9:0]  c3_exp_adj;
    logic [23:0]        c3_mant_rounded;
    logic signed [9:0]  c3_exp_final;
    logic [22:0]        c3_mant_final;
    logic               c3_round_up;
    logic               c3_underflow, c3_overflow;
    logic [7:0]         c3_exp8;
    logic [31:0]        c3_result;

    always_comb begin
        c3_norm = s2_prod[47];

        if (c3_norm) begin
            c3_mant_raw  = s2_prod[46:24];
            c3_round_bit = s2_prod[23];
            c3_sticky    = |s2_prod[22:0];
            c3_exp_adj   = s2_exp + 10'sd1;
        end else begin
            c3_mant_raw  = s2_prod[45:23];
            c3_round_bit = s2_prod[22];
            c3_sticky    = |s2_prod[21:0];
            c3_exp_adj   = s2_exp;
        end

        c3_round_up     = c3_round_bit & (c3_sticky | c3_mant_raw[0]);
        c3_mant_rounded = {1'b0, c3_mant_raw} + {23'h0, c3_round_up};

        if (c3_mant_rounded[23]) begin
            c3_mant_final = c3_mant_rounded[22:0];
            c3_exp_final  = c3_exp_adj + 10'sd1;
        end else begin
            c3_mant_final = c3_mant_rounded[22:0];
            c3_exp_final  = c3_exp_adj;
        end

        c3_underflow = (c3_exp_final <= 10'sd0);
        c3_overflow  = (c3_exp_final >= 10'sd255);
        c3_exp8      = c3_exp_final[7:0];

        if (s2_nan)
            c3_result = 32'h7FC0_0000;
        else if (s2_inf | c3_overflow)
            c3_result = {s2_sign, 8'hFF, 23'h000000};
        else if (s2_zero | c3_underflow)
            c3_result = {s2_sign, 31'h0};
        else
            c3_result = {s2_sign, c3_exp8, c3_mant_final};
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            valid_out <= 1'b0;
            result    <= 32'h0;
        end else begin
            result    <= c3_result;
            valid_out <= s2_valid;
        end
    end

endmodule
