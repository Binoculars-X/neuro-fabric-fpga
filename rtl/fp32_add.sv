// fp32_add.sv — Synthesizable IEEE 754 FP32 adder (also handles subtract via sign bit)
//
// Pipeline: 4 stages, LATENCY = 4 cycles.
// Vivado-synthesizable — logic [31:0] only, no float types or simulation system tasks.
// All signals at module scope — no automatic/inline variable declarations.
//
// Algorithm:
//   Stage 1: Unpack, detect specials, compute exponent difference, swap so |a|>=|b|
//   Stage 2: Align smaller mantissa (right-shift by exp diff), add/subtract
//   Stage 3: Normalize (leading-zero count + shift), round-to-nearest-even
//   Stage 4: Pack and register result
//
// Special cases (IEEE 754):
//   NaN input             → quiet NaN (0x7FC00000)
//   Inf + Inf (same sign) → ±Inf
//   Inf + (-Inf)          → quiet NaN
//   Overflow              → ±Inf
//   Result = 0 / -0       → +0 (flush)
//   Denormal input        → flush-to-zero
//   Denormal result       → flush-to-zero
//
// Rounding: round-to-nearest, ties-to-even.
//
// C# reference: (float)((double)a + (double)b)  matches to <=1 ULP.
// Target: Xilinx ZCU102 (UltraScale+ XCZU9EG-2FFVB1156), Vivado 2023.x.

`timescale 1ns/1ps

module fp32_add (
    input  logic        clk,
    input  logic        rst,
    input  logic        valid_in,
    input  logic [31:0] a,          // IEEE 754 FP32
    input  logic [31:0] b,          // IEEE 754 FP32
    output logic [31:0] result,     // IEEE 754 FP32
    output logic        valid_out
);

    // =========================================================================
    // Stage 1 combinatorial — unpack, special-case detect, swap so |a| >= |b|
    // All wires at module scope; always_ff only drives registers.
    // =========================================================================

    logic        c1_sa, c1_sb;
    logic [7:0]  c1_ea, c1_eb;
    logic [22:0] c1_ma, c1_mb;
    logic        c1_a_nan, c1_b_nan, c1_a_inf, c1_b_inf, c1_a_zero, c1_b_zero;
    logic        c1_swap;
    logic [7:0]  c1_diff8;
    logic [31:0] c1_larger, c1_smaller;
    logic        c1_nan, c1_inf, c1_inf_sign;
    logic        c1_eff_sub;
    logic        c1_sign_large;
    logic [7:0]  c1_exp_large;
    logic [4:0]  c1_exp_diff;
    logic [24:0] c1_mant_large, c1_mant_small;

    always_comb begin
        c1_sa = a[31];     c1_sb = b[31];
        c1_ea = a[30:23];  c1_eb = b[30:23];
        c1_ma = a[22:0];   c1_mb = b[22:0];

        c1_a_nan  = (c1_ea == 8'hFF) & (c1_ma != 23'h0);
        c1_b_nan  = (c1_eb == 8'hFF) & (c1_mb != 23'h0);
        c1_a_inf  = (c1_ea == 8'hFF) & (c1_ma == 23'h0);
        c1_b_inf  = (c1_eb == 8'hFF) & (c1_mb == 23'h0);
        c1_a_zero = (c1_ea == 8'h00);
        c1_b_zero = (c1_eb == 8'h00);

        c1_nan      = c1_a_nan | c1_b_nan | (c1_a_inf & c1_b_inf & (c1_sa ^ c1_sb));
        c1_inf      = (c1_a_inf | c1_b_inf) & ~c1_a_nan & ~c1_b_nan
                      & ~(c1_a_inf & c1_b_inf & (c1_sa ^ c1_sb));
        c1_inf_sign = c1_a_inf ? c1_sa : c1_sb;

        // Swap so larger magnitude is 'large'
        c1_swap    = (c1_eb > c1_ea) | ((c1_ea == c1_eb) & (c1_mb > c1_ma));
        c1_larger  = c1_swap ? b : a;
        c1_smaller = c1_swap ? a : b;

        c1_sign_large = c1_larger[31];
        c1_exp_large  = c1_larger[30:23];
        c1_eff_sub    = c1_larger[31] ^ c1_smaller[31];

        c1_diff8    = c1_larger[30:23] - c1_smaller[30:23];
        c1_exp_diff = (c1_diff8 > 8'd25) ? 5'd25 : c1_diff8[4:0];

        // 25-bit mantissa: {implicit_1, mant[22:0], guard=0}; denormals → 0
        c1_mant_large = (c1_larger[30:23]  == 8'h00) ? 25'h0
                                                       : {1'b1, c1_larger[22:0],  1'b0};
        c1_mant_small = (c1_smaller[30:23] == 8'h00) ? 25'h0
                                                       : {1'b1, c1_smaller[22:0], 1'b0};
    end

    // Stage 1 registers
    logic        s1_valid;
    logic        s1_nan, s1_inf, s1_inf_sign, s1_eff_sub, s1_sign_large;
    logic [7:0]  s1_exp_large;
    logic [4:0]  s1_exp_diff;
    logic [24:0] s1_mant_large, s1_mant_small;

    always_ff @(posedge clk) begin
        if (rst) begin
            s1_valid <= 1'b0;
        end else begin
            s1_valid      <= valid_in;
            s1_nan        <= c1_nan;
            s1_inf        <= c1_inf;
            s1_inf_sign   <= c1_inf_sign;
            s1_eff_sub    <= c1_eff_sub;
            s1_sign_large <= c1_sign_large;
            s1_exp_large  <= c1_exp_large;
            s1_exp_diff   <= c1_exp_diff;
            s1_mant_large <= c1_mant_large;
            s1_mant_small <= c1_mant_small;
        end
    end

    // =========================================================================
    // Stage 2 combinatorial — align smaller, add/subtract
    // sum[25]=carry, sum[24]=leading-1, sum[23:2]=mant, sum[1]=round, sum[0]=sticky
    // =========================================================================

    logic [25:0] c2_small_pre;
    logic        c2_sticky;
    logic [24:0] c2_mant_small_aligned;
    logic [25:0] c2_sum;

    always_comb begin
        c2_small_pre           = {s1_mant_small, 1'b0};
        c2_sticky              = |(c2_small_pre & ((26'h1 << s1_exp_diff) - 26'h1));
        c2_mant_small_aligned  = c2_small_pre[25:1] >> s1_exp_diff;
        c2_mant_small_aligned[0] = c2_mant_small_aligned[0] | c2_sticky;

        if (s1_eff_sub)
            c2_sum = {1'b0, s1_mant_large} - {1'b0, c2_mant_small_aligned};
        else
            c2_sum = {1'b0, s1_mant_large} + {1'b0, c2_mant_small_aligned};
    end

    // Stage 2 registers
    logic        s2_valid;
    logic        s2_nan, s2_inf, s2_inf_sign, s2_sign_large;
    logic [7:0]  s2_exp_large;
    logic [25:0] s2_sum;
    logic        s2_sticky;  // sticky bit: OR of all bits shifted out during alignment

    always_ff @(posedge clk) begin
        if (rst) begin
            s2_valid <= 1'b0;
        end else begin
            s2_valid      <= s1_valid;
            s2_nan        <= s1_nan;
            s2_inf        <= s1_inf;
            s2_inf_sign   <= s1_inf_sign;
            s2_sign_large <= s1_sign_large;
            s2_exp_large  <= s1_exp_large;
            s2_sum        <= c2_sum;
            s2_sticky     <= c2_sticky;
        end
    end

    // =========================================================================
    // Stage 3 combinatorial — normalize + round-to-nearest-even
    // =========================================================================

    logic [25:0] c3_norm_sum;
    logic [4:0]  c3_lz;
    logic [8:0]  c3_exp_adj;
    logic [22:0] c3_mant_raw;
    logic        c3_round_bit, c3_sticky_bit, c3_round_up;
    logic [23:0] c3_mant_rounded;
    logic [22:0] c3_mant_out;
    logic [8:0]  c3_exp_out;
    logic        c3_overflow, c3_zero;

    always_comb begin
        c3_norm_sum = s2_sum;
        c3_lz       = 5'd24;

        if (c3_norm_sum[25]) begin
            // Carry-out: exp+1, shift right
            c3_mant_raw  = c3_norm_sum[24:2];
            c3_round_bit = c3_norm_sum[1];
            c3_sticky_bit = c3_norm_sum[0];
            c3_exp_adj   = {1'b0, s2_exp_large} + 9'd1;
        end else if (c3_norm_sum[24]) begin
            // Normalised
            c3_mant_raw   = c3_norm_sum[23:1];
            c3_round_bit  = c3_norm_sum[0];
            c3_sticky_bit = s2_sticky;          // carry sticky from Stage 2
            c3_exp_adj    = {1'b0, s2_exp_large};
        end else begin
            // Subtraction cancellation — priority encode leading 1 (low-to-high so highest wins)
            for (int k = 0; k <= 23; k++) begin
                if (c3_norm_sum[k]) c3_lz = 5'(24 - k);
            end
            c3_norm_sum  = c3_norm_sum << c3_lz;
            c3_mant_raw  = c3_norm_sum[23:1];
            c3_round_bit = c3_norm_sum[0];
            c3_sticky_bit = 1'b0;               // bits shifted in are zeros (left-shift)
            c3_exp_adj   = {1'b0, s2_exp_large} - {4'b0, c3_lz};
        end

        c3_round_up     = c3_round_bit & (c3_sticky_bit | c3_mant_raw[0]);
        c3_mant_rounded = {1'b0, c3_mant_raw} + {23'h0, c3_round_up};

        if (c3_mant_rounded[23]) begin
            // Mantissa overflowed on rounding — it's now 0x800000, which is 1.0 in normalized form
            // After implicit right-shift, this becomes {1.000...000}, mantissa bits are all 0, exp+1
            c3_mant_out = 23'h0;  // Mantissa is all zeros after overflow
            c3_exp_out  = c3_exp_adj + 9'd1;
        end else begin
            // No overflow — mantissa fits in 23 bits
            c3_mant_out = c3_mant_rounded[22:0];
            c3_exp_out  = c3_exp_adj;
        end

        c3_overflow = ~c3_exp_out[8] & (c3_exp_out[7:0] >= 8'hFF);
        c3_zero     =  c3_exp_out[8] | (s2_sum == 26'h0);
    end

    // Stage 3 registers
    logic        s3_valid;
    logic        s3_nan, s3_inf, s3_inf_sign, s3_sign;
    logic        s3_overflow, s3_zero;
    logic [8:0]  s3_exp;
    logic [22:0] s3_mant;

    always_ff @(posedge clk) begin
        if (rst) begin
            s3_valid <= 1'b0;
        end else begin
            s3_valid    <= s2_valid;
            s3_nan      <= s2_nan;
            s3_inf      <= s2_inf;
            s3_inf_sign <= s2_inf_sign;
            s3_sign     <= s2_sign_large;
            s3_overflow <= c3_overflow;
            s3_zero     <= c3_zero;
            s3_exp      <= c3_exp_out;
            s3_mant     <= c3_mant_out;
        end
    end

    // =========================================================================
    // Stage 4 combinatorial — pack result
    // =========================================================================

    logic [31:0] c4_result;

    always_comb begin
        if (s3_nan)
            c4_result = 32'h7FC0_0000;
        else if (s3_inf | s3_overflow)
            c4_result = {s3_inf ? s3_inf_sign : s3_sign, 8'hFF, 23'h0};
        else if (s3_zero)
            c4_result = 32'h0000_0000;
        else
            c4_result = {s3_sign, s3_exp[7:0], s3_mant};
    end

    // Stage 4 register
    always_ff @(posedge clk) begin
        if (rst) begin
            valid_out <= 1'b0;
            result    <= 32'h0;
        end else begin
            result    <= c4_result;
            valid_out <= s3_valid;
        end
    end

endmodule
