// Element-wise GeLU activation — tanh approximation, pipelined
//
// Algorithm (matches AttentionLayer.Gelu exactly):
//   inner = C * (x + 0.044715 * x^3)      where C = sqrt(2/pi) = 0.7978845608
//   y     = 0.5 * x * (1 + tanh(inner))
//
// tanh implementation:
//   tanh(u) = (exp(2u) - 1) / (exp(2u) + 1)
//   Uses exp_lut to compute exp(2u) — same module used by softmax.
//   For |inner| >= TANH_CLAMP (4.0), tanh(inner) = sign(inner) * 1.0 exactly.
//   For inner = 0.0, tanh = 0.0.
//
// Pipeline: 6 stages
//   Stage 1: compute x^2, x^3; compute inner = C*(x + 0.044715*x^3)
//   Stage 2: compute 2*inner; issue to exp_lut; also buffer x
//   Stages 3-6: exp_lut pipeline (EXP_LAT=4 cycles)
//   Stage EXP_LAT+2: compute e2u = exp_result; tanh = (e2u-1)/(e2u+1); y = 0.5*x*(1+tanh)
//
// Total latency from valid_in: EXP_LAT + 2 cycles  (default: 6 cycles)
//
// Interface:
//   x_fp32     — 32-bit FP32 input
//   valid_in   — qualify input
//   result_fp32 — 32-bit FP32 GeLU(x) output
//   valid_out  — qualifies output
//
// Parameters:
//   EXP_LAT  — exp_lut pipeline depth (must match exp_lut instantiation, default 4)
//   LUT_SIZE — exp LUT entries (256)
//   LUT_FILE — BRAM init hex file (same exp_lut_init.hex used by softmax)

`timescale 1ns/1ps

module gelu #(
    parameter int    EXP_LAT  = 4,
    parameter int    LUT_SIZE = 256,
    parameter string LUT_FILE = "exp_lut_init.hex"
)(
    input  logic        clk,
    input  logic        rst,
    input  logic        en,

    input  logic [31:0] x_fp32,
    input  logic        valid_in,

    output logic [31:0] result_fp32,
    output logic        valid_out
);

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------
    // C = sqrt(2/pi) = 0.7978845608 as FP32 bits
    localparam logic [31:0] C_BITS        = 32'h3F4C422A;
    // 0.044715 as FP32 bits
    localparam logic [31:0] K_BITS        = 32'h3D372713;
    // 0.5 as FP32 bits
    localparam logic [31:0] HALF_BITS     = 32'h3F000000;
    // 1.0 as FP32 bits
    localparam logic [31:0] ONE_BITS      = 32'h3F800000;
    // 2.0 as FP32 bits
    localparam logic [31:0] TWO_BITS      = 32'h40000000;
    // Clamp threshold for tanh: |inner| >= 4.0 → tanh = ±1.0
    localparam logic [31:0] TANH_CLAMP_BITS = 32'h40800000; // 4.0

    shortreal sr_C, sr_K, sr_half, sr_one, sr_two, sr_clamp;
    always_comb begin
        sr_C     = $bitstoshortreal(C_BITS);
        sr_K     = $bitstoshortreal(K_BITS);
        sr_half  = $bitstoshortreal(HALF_BITS);
        sr_one   = $bitstoshortreal(ONE_BITS);
        sr_two   = $bitstoshortreal(TWO_BITS);
        sr_clamp = $bitstoshortreal(TANH_CLAMP_BITS);
    end

    // -----------------------------------------------------------------------
    // exp_lut instance — computes exp(2*inner)
    // -----------------------------------------------------------------------
    logic [31:0] exp_x_reg;
    logic        exp_valid_in_reg;
    logic [31:0] exp_result;
    logic        exp_valid_out;

    exp_lut #(
        .LUT_SIZE(LUT_SIZE),
        .LUT_FILE(LUT_FILE)
    ) u_exp (
        .clk          (clk),
        .rst          (rst),
        .en           (en),
        .x_fp32       (exp_x_reg),
        .valid_in     (exp_valid_in_reg),
        .result_fp32  (exp_result),
        .valid_out    (exp_valid_out)
    );

    // -----------------------------------------------------------------------
    // Stage 1: polynomial — compute inner = C*(x + K*x^3)
    // Also detect saturation: |inner| >= 4.0 at this stage is approximated;
    // final clamp is applied in the output stage.
    // -----------------------------------------------------------------------
    shortreal s1_inner;
    shortreal s1_x;        // buffered x for final multiply
    logic     s1_v;

    always_ff @(posedge clk) begin
        if (rst) begin
            s1_v <= 1'b0;
        end else if (en) begin
            shortreal sx, sx3;
            sx      = $bitstoshortreal(x_fp32);
            sx3     = sx * sx * sx;
            s1_inner <= sr_C * (sx + sr_K * sx3);
            s1_x    <= sx;
            s1_v    <= valid_in;
        end
    end

    // -----------------------------------------------------------------------
    // Stage 2: issue 2*inner to exp_lut; buffer x; detect clamp
    // -----------------------------------------------------------------------
    shortreal    s2_x;
    shortreal    s2_inner;
    logic        s2_v;
    // Clamp flag: |inner| >= 4 means tanh saturates to ±1
    logic        s2_clamped;
    logic        s2_sign;    // sign of inner (1 = negative)

    always_ff @(posedge clk) begin
        if (rst) begin
            s2_v              <= 1'b0;
            exp_valid_in_reg  <= 1'b0;
        end else if (en) begin
            shortreal abs_inner;
            abs_inner        = (s1_inner < 0.0) ? -s1_inner : s1_inner;
            s2_clamped       <= (abs_inner >= sr_clamp);
            s2_sign          <= (s1_inner < 0.0);
            exp_x_reg        <= $shortrealtobits(sr_two * s1_inner);
            exp_valid_in_reg <= s1_v;
            s2_x             <= s1_x;
            s2_inner         <= s1_inner;
            s2_v             <= s1_v;
        end
    end

    // -----------------------------------------------------------------------
    // Delay line: keep x, clamped, sign aligned with exp_lut output
    // exp_lut takes EXP_LAT cycles from valid_in, so we need EXP_LAT-1 more
    // delay stages after stage 2 (stage 2 already absorbed 1 cycle of delay).
    // -----------------------------------------------------------------------
    shortreal    dly_x      [0:EXP_LAT-1];
    logic        dly_clamped[0:EXP_LAT-1];
    logic        dly_sign   [0:EXP_LAT-1];

    always_ff @(posedge clk) begin
        if (rst) begin
            // nothing to reset for data path
        end else if (en) begin
            dly_x[0]       <= s2_x;
            dly_clamped[0] <= s2_clamped;
            dly_sign[0]    <= s2_sign;
            for (int i = 1; i < EXP_LAT; i++) begin
                dly_x[i]       <= dly_x[i-1];
                dly_clamped[i] <= dly_clamped[i-1];
                dly_sign[i]    <= dly_sign[i-1];
            end
        end
    end

    // -----------------------------------------------------------------------
    // Output stage: fires when exp_lut asserts valid_out
    // Compute tanh = (e2u-1)/(e2u+1), then GeLU = 0.5*x*(1+tanh)
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            valid_out <= 1'b0;
        end else if (en) begin
            valid_out <= exp_valid_out;
            if (exp_valid_out) begin
                shortreal e2u, tanh_val, gelu_val;
                shortreal final_x;
                logic     final_clamped;
                logic     final_sign;

                final_x       = dly_x[EXP_LAT-1];
                final_clamped = dly_clamped[EXP_LAT-1];
                final_sign    = dly_sign[EXP_LAT-1];

                e2u      = $bitstoshortreal(exp_result);

                if (final_clamped)
                    // tanh saturates: sign determines +1 or -1
                    tanh_val = final_sign ? -sr_one : sr_one;
                else
                    tanh_val = (e2u - sr_one) / (e2u + sr_one);

                gelu_val    = sr_half * final_x * (sr_one + tanh_val);
                result_fp32 <= $shortrealtobits(gelu_val);
            end
        end
    end

endmodule
