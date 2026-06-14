// Adam scalar update cell — BF16-weights variant
//
// Matches AdamBF16WeightsAttentionCore.ApplyUpdate() exactly (one parameter).
//
// C# order (mHat/vHat use the NEWLY computed mf/vf — same cycle as gradient):
//   mf    = Beta1*m + (1-Beta1)*g
//   vf    = Beta2*v + (1-Beta2)*g*g
//   mHat  = mf / bc1                        -- fp32_div
//   vHat  = vf / bc2                        -- fp32_div
//   sqrtV = vHat * fp32_sqrt(vHat)          -- reciprocal sqrt, multiply back
//   denom = sqrtV + Epsilon                 -- shortreal add
//   wf    = Decode(w_bf16) - lr*mHat/denom  -- fp32_div
//   w_bf16_out = wf[31:16]
//
// All of the above is a COMBINATORIAL pipeline from inputs.
// Outputs are registered on the rising edge when en=1.
// Caller (adam_core) holds the moment stores and feeds m_fp32/v_fp32 each cycle.
//
// bc1 = 1 - Beta1^t  (per-step FP32 input; caller computes -- avoids $pow())
// bc2 = 1 - Beta2^t
//
// No $pow(), no $sqrt() -- fully synthesizable.
// C# reference: AdamBF16WeightsAttentionCore.ApplyUpdate()
// XSim tolerance: VsSoftwareRelTol = 0.01%

`timescale 1ns/1ps

module adam_cell (
    input  logic        clk,
    input  logic        rst,
    input  logic        en,

    input  logic [31:0] g_fp32,        // gradient
    input  logic [15:0] w_bf16,        // current BF16 weight
    input  logic [31:0] m_fp32,        // first moment (previous step, from caller)
    input  logic [31:0] v_fp32,        // second moment (previous step, from caller)
    input  logic [31:0] lr_fp32,       // learning rate
    input  logic [31:0] bc1_fp32,      // bias correction 1 = 1 - Beta1^t
    input  logic [31:0] bc2_fp32,      // bias correction 2 = 1 - Beta2^t

    output logic [15:0] w_bf16_out,    // updated BF16 weight
    output logic [31:0] m_fp32_out,    // updated first moment
    output logic [31:0] v_fp32_out     // updated second moment
);

    // ---- Constants ----------------------------------------------------------
    localparam shortreal BETA1   = 0.9;
    localparam shortreal BETA2   = 0.999;
    localparam shortreal EPSILON = 1e-8;

    // ---- Step 1: mf = Beta1*m + (1-Beta1)*g  combinatorial -----------------
    //             vf = Beta2*v + (1-Beta2)*g*g
    logic [31:0] mf_bits, vf_bits;
    always_comb begin
        shortreal g, m_sr, v_sr, mf, vf;
        g    = $bitstoshortreal(g_fp32);
        m_sr = $bitstoshortreal(m_fp32);
        v_sr = $bitstoshortreal(v_fp32);
        mf   = BETA1 * m_sr + (1.0 - BETA1) * g;
        vf   = BETA2 * v_sr + (1.0 - BETA2) * g * g;
        mf_bits = $shortrealtobits(mf);
        vf_bits = $shortrealtobits(vf);
    end

    // ---- Step 2: mHat = mf / bc1 --------------------------------------------
    logic [31:0] mhat_bits;
    fp32_div u_div_mhat (
        .a_fp32     (mf_bits),
        .b_fp32     (bc1_fp32),
        .result_fp32(mhat_bits)
    );

    // ---- Step 3: vHat = vf / bc2 --------------------------------------------
    logic [31:0] vhat_bits;
    fp32_div u_div_vhat (
        .a_fp32     (vf_bits),
        .b_fp32     (bc2_fp32),
        .result_fp32(vhat_bits)
    );

    // ---- Step 4: 1/sqrt(vHat) -----------------------------------------------
    logic [31:0] recip_sqrt_bits;
    fp32_sqrt u_sqrt (
        .x_fp32     (vhat_bits),
        .result_fp32(recip_sqrt_bits)
    );

    // ---- Step 5: sqrtV = vHat * recip_sqrt;  denom = sqrtV + epsilon --------
    logic [31:0] denom_bits, lr_mhat_bits;
    always_comb begin
        shortreal vhat_sr, recip_sr, sqrtv_sr, mhat_sr, lr_sr;
        vhat_sr      = $bitstoshortreal(vhat_bits);
        recip_sr     = $bitstoshortreal(recip_sqrt_bits);
        sqrtv_sr     = vhat_sr * recip_sr;
        denom_bits   = $shortrealtobits(sqrtv_sr + EPSILON);
        mhat_sr      = $bitstoshortreal(mhat_bits);
        lr_sr        = $bitstoshortreal(lr_fp32);
        lr_mhat_bits = $shortrealtobits(lr_sr * mhat_sr);
    end

    // ---- Step 6: delta = lr*mHat / denom ------------------------------------
    logic [31:0] delta_bits;
    fp32_div u_div_step (
        .a_fp32     (lr_mhat_bits),
        .b_fp32     (denom_bits),
        .result_fp32(delta_bits)
    );

    // ---- Register outputs on rising edge ------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            w_bf16_out <= 16'h0;
            m_fp32_out <= 32'h0;
            v_fp32_out <= 32'h0;
        end else if (en) begin
            shortreal wf, delta_sr;
            logic [31:0] wf_bits;
            delta_sr   = $bitstoshortreal(delta_bits);
            wf         = $bitstoshortreal({w_bf16, 16'h0000}) - delta_sr;
            wf_bits    = $shortrealtobits(wf) + 32'h8000;  // round-to-nearest (matches Bf16.Encode)
            w_bf16_out <= wf_bits[31:16];
            m_fp32_out <= mf_bits;
            v_fp32_out <= vf_bits;
        end
    end

endmodule
