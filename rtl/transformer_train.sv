// transformer_train.sv
// Full transformer train step (Phase 1: step 7b).
//
// Extends transformer.sv with backward + Adam update.
// Uses SEPARATE sub-module instances per layer so each retains its
// own forward-pass cache (mean/var latches, activations) for backward.
//
// High-level sequence:
//   1. start   → forward pass → done (logits streamed via out_row/out_valid)
//   2. Testbench computes cross-entropy, dX_init, dWout (emb grad)
//      and writes them via the extended write bus.
//   3. train_start → backward (L=2 layers, reverse) → Adam → adam_done
//   4. Testbench reads Adam state from DUT internal ports after adam_done.
//
// New write bus addresses (beyond 0x12F used by forward):
//   0x130..0x13F  dX_init[T×D]  FP32 (gradient entering layer backward from above)
//   0x140..0x17F  dWout[V×D]    FP32 (embedding tied-weight gradient)
//   0x180         lr_fp32
//   0x181         bc1_fp32  (= 1 - Beta1^step, computed by testbench)
//   0x182         bc2_fp32  (= 1 - Beta2^step, computed by testbench)
//
// Adam cores (accessible by testbench via hierarchical reference after adam_done):
//   u_adam_emb   adam_core #(.R(V), .C(D))   — embedding
//   u_adam_wq0   adam_core #(.R(D), .C(DH))  — Wq, layer 0
//   u_adam_wk0, u_adam_wv0                   — Wk, Wv, layer 0
//   u_adam_wo0   adam_core #(.R(DH), .C(D))  — Wo, layer 0
//   u_adam_wff10, u_adam_wff20               — Wff1, Wff2, layer 0
//   u_adam_wq1..u_adam_wff21                 — same for layer 1
//
// Adam m/v moments are initialised to 0 (reset). The testbench verifies step=1.
//
// Write bus and forward output interface are identical to transformer.sv so the
// same input.hex (304 entries) works for both forward-only and train-step tests.
//
// Model configuration (hardcoded): T=4, D=4, DH=4, FF=4, L=2, V=16

`timescale 1ns/1ps

module transformer_train #(
    parameter int    T        = 4,
    parameter int    D        = 4,
    parameter int    DH       = 4,
    parameter int    FF       = 4,
    parameter int    L        = 2,
    parameter int    V        = 16,
    parameter int    MAC_LAT  = 3,
    parameter int    MUL_LAT  = 3,
    parameter int    EXP_LAT  = 4,
    parameter int    LUT_SIZE = 256,
    parameter string LUT_FILE = "exp_lut_init.hex"
)(
    input  logic clk,
    input  logic rst,
    input  logic en,

    // Shared write bus (forward weights + train data)
    input  logic        wr_en,
    input  logic [8:0]  wr_addr,
    input  logic [31:0] wr_data,

    // Forward control
    input  logic start,
    output logic [V*32-1:0] out_row,
    output logic             out_valid,
    output logic [1:0]       out_row_idx,
    output logic done,

    // Backward / Adam control
    input  logic train_start,
    output logic adam_done
);

    // -----------------------------------------------------------------------
    // Registers
    // -----------------------------------------------------------------------
    logic [31:0] emb_reg     [0:V-1][0:D-1];
    logic [31:0] x_init_reg  [0:T-1][0:D-1];
    logic [31:0] x_reg       [0:T-1][0:D-1];
    logic [31:0] xnorm1_0_buf[0:T-1][0:D-1];
    logic [31:0] xnorm2_0_buf[0:T-1][0:D-1];
    logic [31:0] xnorm1_1_buf[0:T-1][0:D-1];
    logic [31:0] xnorm2_1_buf[0:T-1][0:D-1];
    logic [31:0] attn_buf    [0:T-1][0:D-1];
    logic [31:0] mlp_buf     [0:T-1][0:D-1];

    logic [31:0] layer_ln1_gamma[0:L-1][0:D-1];
    logic [31:0] layer_ln1_beta [0:L-1][0:D-1];
    logic [31:0] layer_ln2_gamma[0:L-1][0:D-1];
    logic [31:0] layer_ln2_beta [0:L-1][0:D-1];
    logic [15:0] layer_Wq   [0:L-1][0:D-1][0:DH-1];
    logic [15:0] layer_Wk   [0:L-1][0:D-1][0:DH-1];
    logic [15:0] layer_Wv   [0:L-1][0:D-1][0:DH-1];
    logic [15:0] layer_Wo   [0:L-1][0:DH-1][0:D-1];
    logic [15:0] layer_Wff1 [0:L-1][0:D-1][0:FF-1];
    logic [15:0] layer_Wff2 [0:L-1][0:FF-1][0:D-1];

    // Backward / Adam registers
    logic [31:0] dX_init_reg [0:T-1][0:D-1];  // gradient entering layer backward from testbench
    logic [31:0] dWout_reg   [0:V-1][0:D-1];  // embedding gradient from testbench
    logic [31:0] lr_reg, bc1_reg, bc2_reg;

    // Cross-entropy computation registers
    logic [31:0]  logits_reg  [0:T-1][0:V-1];   // buffered forward pass logits
    logic [3:0]   targets_reg [0:T-1];            // target token indices (0..V-1)
    logic [31:0]  dLogits_reg [0:T-1][0:V-1];    // computed dLogits (FP32 bits)
    shortreal     max_logit_sr;                   // per-row softmax stability max
    shortreal     exp_sum_sr;                     // exp accumulator
    shortreal     exp_buf_sr   [0:V-1];           // per-row exp values
    shortreal     grad_norm_sq_sr;                // ||dLogits||^2
    shortreal     clip_scale_sr;                  // 1.0 or 1/||dLogits||

    // Intermediate gradient buffers (reused across backward phases)
    logic [31:0] dx_buf   [0:T-1][0:D-1];   // current gradient propagating backward
    logic [31:0] dX_tmp   [0:T-1][0:D-1];   // scratch: captured dx from a sub-module
    logic [31:0] dAttn_tmp[0:T-1][0:D-1];   // scratch: dAttnOut = dX_ln2 + dx_buf

    // -----------------------------------------------------------------------
    // Write bus decoder
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (wr_en) begin
            // emb_reg [V*D]: addr 0x000..0x03F
            if (wr_addr <= 9'h03F)
                emb_reg[wr_addr[5:2]][wr_addr[1:0]] <= wr_data;

            // x_init [T*D]: addr 0x040..0x04F
            else if (wr_addr >= 9'h040 && wr_addr <= 9'h04F)
                x_init_reg[wr_addr[3:2]][wr_addr[1:0]] <= wr_data;

            // Layer 0: addr 0x050..0x0BF
            else if (wr_addr >= 9'h050 && wr_addr <= 9'h0BF) begin
                automatic logic [7:0] loff = wr_addr[7:0] - 8'h50;
                case (loff[6:4])
                    3'd0: case (loff[3:2])
                            2'd0: layer_ln1_gamma[0][loff[1:0]] <= wr_data;
                            2'd1: layer_ln1_beta [0][loff[1:0]] <= wr_data;
                            2'd2: layer_ln2_gamma[0][loff[1:0]] <= wr_data;
                            2'd3: layer_ln2_beta [0][loff[1:0]] <= wr_data;
                          endcase
                    3'd1: layer_Wq  [0][loff[3:2]][loff[1:0]] <= wr_data[15:0];
                    3'd2: layer_Wk  [0][loff[3:2]][loff[1:0]] <= wr_data[15:0];
                    3'd3: layer_Wv  [0][loff[3:2]][loff[1:0]] <= wr_data[15:0];
                    3'd4: layer_Wo  [0][loff[3:2]][loff[1:0]] <= wr_data[15:0];
                    3'd5: layer_Wff1[0][loff[3:2]][loff[1:0]] <= wr_data[15:0];
                    3'd6: layer_Wff2[0][loff[3:2]][loff[1:0]] <= wr_data[15:0];
                    default: ;
                endcase
            end

            // Layer 1: addr 0x0C0..0x12F
            else if (wr_addr >= 9'h0C0 && wr_addr <= 9'h12F) begin
                automatic logic [7:0] loff = wr_addr[7:0] - 8'hC0;
                case (loff[6:4])
                    3'd0: case (loff[3:2])
                            2'd0: layer_ln1_gamma[1][loff[1:0]] <= wr_data;
                            2'd1: layer_ln1_beta [1][loff[1:0]] <= wr_data;
                            2'd2: layer_ln2_gamma[1][loff[1:0]] <= wr_data;
                            2'd3: layer_ln2_beta [1][loff[1:0]] <= wr_data;
                          endcase
                    3'd1: layer_Wq  [1][loff[3:2]][loff[1:0]] <= wr_data[15:0];
                    3'd2: layer_Wk  [1][loff[3:2]][loff[1:0]] <= wr_data[15:0];
                    3'd3: layer_Wv  [1][loff[3:2]][loff[1:0]] <= wr_data[15:0];
                    3'd4: layer_Wo  [1][loff[3:2]][loff[1:0]] <= wr_data[15:0];
                    3'd5: layer_Wff1[1][loff[3:2]][loff[1:0]] <= wr_data[15:0];
                    3'd6: layer_Wff2[1][loff[3:2]][loff[1:0]] <= wr_data[15:0];
                    default: ;
                endcase
            end

            // targets [T]: addr 0x130..0x133 (4 entries, int token index)
            else if (wr_addr >= 9'h130 && wr_addr <= 9'h133)
                targets_reg[wr_addr[1:0]] <= wr_data[3:0];

            // Scalar train params
            else if (wr_addr == 9'h180) lr_reg  <= wr_data;
            else if (wr_addr == 9'h181) bc1_reg <= wr_data;
            else if (wr_addr == 9'h182) bc2_reg <= wr_data;
        end
    end

    // -----------------------------------------------------------------------
    // Sub-module signals — 4 LN instances, 2 ATT, 2 MLP, 1 PROJ
    // -----------------------------------------------------------------------

    // LN instances: 0=LN1_0, 1=LN2_0, 2=LN1_1, 3=LN2_1
    logic [D*32-1:0] lnX[4], lnG[4], lnB[4], lnY[4];
    logic            lnStart[4], lnValid[4];
    logic            lnDyEn[4]; logic [7:0] lnDyAddr[4]; logic [31:0] lnDyData[4];
    logic            lnBwdStart[4];
    logic [D*32-1:0] lnDx[4]; logic lnDxValid[4]; logic [$clog2(T)-1:0] lnDxIdx[4];
    logic [D*32-1:0] lnDGamma[4], lnDBeta[4];

    generate
        for (genvar li = 0; li < 4; li++) begin : gen_ln
            layernorm #(.D(D), .T(T)) u_ln_inst (
                .clk(clk), .rst(rst), .en(en),
                .x_in(lnX[li]), .gamma(lnG[li]), .beta(lnB[li]),
                .start(lnStart[li]), .y_out(lnY[li]), .out_valid(lnValid[li]),
                .dy_wr_en(lnDyEn[li]), .dy_wr_addr(lnDyAddr[li]), .dy_wr_data(lnDyData[li]),
                .bwd_start(lnBwdStart[li]),
                .dx_row(lnDx[li]), .dx_valid(lnDxValid[li]), .dx_row_idx(lnDxIdx[li]),
                .dGamma_flat(lnDGamma[li]), .dBeta_flat(lnDBeta[li])
            );
        end
    endgenerate

    // ATT instances: 0=layer0, 1=layer1
    logic        attXEn[2]; logic [7:0] attXAddr[2]; logic [31:0] attXData[2];
    logic        attWqEn[2]; logic [7:0] attWqAddr[2]; logic [15:0] attWqData[2];
    logic        attWkEn[2]; logic [7:0] attWkAddr[2]; logic [15:0] attWkData[2];
    logic        attWvEn[2]; logic [7:0] attWvAddr[2]; logic [15:0] attWvData[2];
    logic        attWoEn[2]; logic [7:0] attWoAddr[2]; logic [15:0] attWoData[2];
    logic        attStart[2];
    logic [D*32-1:0] attOutRow[2]; logic attOutValid[2]; logic [1:0] attOutIdx[2];
    logic        attDyEn[2]; logic [7:0] attDyAddr[2]; logic [31:0] attDyData[2];
    logic        attBwdStart[2];
    logic [D*32-1:0] attDxRow[2]; logic attDxValid[2]; logic [1:0] attDxIdx[2];
    logic [D*DH*32-1:0] attDWq[2], attDWk[2], attDWv[2];
    logic [DH*D*32-1:0] attDWo[2];

    generate
        for (genvar ai = 0; ai < 2; ai++) begin : gen_att
            attention_core #(
                .T(T), .D(D), .DH(DH), .MAC_LAT(MAC_LAT), .MUL_LAT(MUL_LAT),
                .EXP_LAT(EXP_LAT), .LUT_SIZE(LUT_SIZE), .LUT_FILE(LUT_FILE)
            ) u_att_inst (
                .clk(clk), .rst(rst), .en(en),
                .x_wr_en(attXEn[ai]), .x_wr_addr(attXAddr[ai]), .x_wr_data(attXData[ai]),
                .wq_wr_en(attWqEn[ai]), .wq_wr_addr(attWqAddr[ai]), .wq_wr_data(attWqData[ai]),
                .wk_wr_en(attWkEn[ai]), .wk_wr_addr(attWkAddr[ai]), .wk_wr_data(attWkData[ai]),
                .wv_wr_en(attWvEn[ai]), .wv_wr_addr(attWvAddr[ai]), .wv_wr_data(attWvData[ai]),
                .wo_wr_en(attWoEn[ai]), .wo_wr_addr(attWoAddr[ai]), .wo_wr_data(attWoData[ai]),
                .start(attStart[ai]),
                .out_row(attOutRow[ai]), .out_valid(attOutValid[ai]), .out_row_idx(attOutIdx[ai]),
                .dy_wr_en(attDyEn[ai]), .dy_wr_addr(attDyAddr[ai]), .dy_wr_data(attDyData[ai]),
                .bwd_start(attBwdStart[ai]),
                .dx_row(attDxRow[ai]), .dx_valid(attDxValid[ai]), .dx_row_idx(attDxIdx[ai]),
                .dWq_flat(attDWq[ai]), .dWk_flat(attDWk[ai]),
                .dWv_flat(attDWv[ai]), .dWo_flat(attDWo[ai])
            );
        end
    endgenerate

    // MLP instances: 0=layer0, 1=layer1
    logic        mlpXEn[2]; logic [7:0] mlpXAddr[2]; logic [31:0] mlpXData[2];
    logic        mlpW1En[2]; logic [7:0] mlpW1Addr[2]; logic [15:0] mlpW1Data[2];
    logic        mlpW2En[2]; logic [7:0] mlpW2Addr[2]; logic [15:0] mlpW2Data[2];
    logic        mlpStart[2];
    logic [D*32-1:0] mlpOutRow[2]; logic mlpOutValid[2]; logic [1:0] mlpOutIdx[2];
    logic        mlpDyEn[2]; logic [7:0] mlpDyAddr[2]; logic [31:0] mlpDyData[2];
    logic        mlpBwdStart[2];
    logic [D*32-1:0] mlpDxRow[2]; logic mlpDxValid[2]; logic [1:0] mlpDxIdx[2];
    logic [D*FF*32-1:0] mlpDWff1[2];
    logic [FF*D*32-1:0] mlpDWff2[2];

    generate
        for (genvar mi = 0; mi < 2; mi++) begin : gen_mlp
            mlp_core #(
                .T(T), .D(D), .FF(FF), .MAC_LAT(MAC_LAT), .EXP_LAT(EXP_LAT),
                .LUT_SIZE(LUT_SIZE), .LUT_FILE(LUT_FILE)
            ) u_mlp_inst (
                .clk(clk), .rst(rst), .en(en),
                .x_wr_en(mlpXEn[mi]), .x_wr_addr(mlpXAddr[mi]), .x_wr_data(mlpXData[mi]),
                .wff1_wr_en(mlpW1En[mi]), .wff1_wr_addr(mlpW1Addr[mi]), .wff1_wr_data(mlpW1Data[mi]),
                .wff2_wr_en(mlpW2En[mi]), .wff2_wr_addr(mlpW2Addr[mi]), .wff2_wr_data(mlpW2Data[mi]),
                .start(mlpStart[mi]),
                .out_row(mlpOutRow[mi]), .out_valid(mlpOutValid[mi]), .out_row_idx(mlpOutIdx[mi]),
                .dy_wr_en(mlpDyEn[mi]), .dy_wr_addr(mlpDyAddr[mi]), .dy_wr_data(mlpDyData[mi]),
                .bwd_start(mlpBwdStart[mi]),
                .dx_row(mlpDxRow[mi]), .dx_valid(mlpDxValid[mi]), .dx_row_idx(mlpDxIdx[mi]),
                .dWff1_flat(mlpDWff1[mi]), .dWff2_flat(mlpDWff2[mi])
            );
        end
    endgenerate

    // Output projection: fp32_matmul [T×D] · [D×V]^T → [T×V]
    logic        projAEn, projBEn;
    logic [7:0]  projAAddr, projBAddr;
    logic [31:0] projAData, projBData;
    logic        projStart;
    logic [V*32-1:0] projCRow; logic projCValid; logic [1:0] projCIdx;

    fp32_matmul #(.M(T), .K(D), .N(V), .MUL_LATENCY(MUL_LAT)) u_proj (
        .clk(clk), .rst(rst), .en(en),
        .a_wr_en(projAEn), .a_wr_addr(projAAddr), .a_wr_data(projAData),
        .b_wr_en(projBEn), .b_wr_addr(projBAddr), .b_wr_data(projBData),
        .start(projStart),
        .c_row(projCRow), .c_valid(projCValid), .c_row_idx(projCIdx)
    );

    // -----------------------------------------------------------------------
    // Adam cores
    // All grad inputs wired directly from backward module output ports.
    // m_in = v_in = 0 (initial moments for step 1).
    // w_bf16_in packed from layer register arrays (top 16 bits = BF16).
    // -----------------------------------------------------------------------

    // Packed flat gradient buses (combinatorial)
    logic [V*D*32-1:0]  adam_emb_grad;
    logic [D*DH*32-1:0] adam_wq0_grad, adam_wk0_grad, adam_wv0_grad;
    logic [DH*D*32-1:0] adam_wo0_grad;
    logic [D*FF*32-1:0] adam_wff10_grad;
    logic [FF*D*32-1:0] adam_wff20_grad;
    logic [D*DH*32-1:0] adam_wq1_grad, adam_wk1_grad, adam_wv1_grad;
    logic [DH*D*32-1:0] adam_wo1_grad;
    logic [D*FF*32-1:0] adam_wff11_grad;
    logic [FF*D*32-1:0] adam_wff21_grad;

    // Packed flat w_bf16 inputs
    logic [V*D*16-1:0]  adam_emb_w;
    logic [D*DH*16-1:0] adam_wq0_w, adam_wk0_w, adam_wv0_w, adam_wq1_w, adam_wk1_w, adam_wv1_w;
    logic [DH*D*16-1:0] adam_wo0_w, adam_wo1_w;
    logic [D*FF*16-1:0] adam_wff10_w, adam_wff11_w;
    logic [FF*D*16-1:0] adam_wff20_w, adam_wff21_w;

    // Pack gradient and weight buses combinatorially
    always_comb begin
        // Embedding gradient (from testbench-written dWout_reg)
        for (int v = 0; v < V; v++)
            for (int d = 0; d < D; d++)
                adam_emb_grad[(v*D+d)*32 +: 32] = dWout_reg[v][d];

        // Embedding w_bf16 (top 16 bits of FP32 = BF16 encoding)
        for (int v = 0; v < V; v++)
            for (int d = 0; d < D; d++)
                adam_emb_w[(v*D+d)*16 +: 16] = emb_reg[v][d][31:16];

        // Weight matrix gradients from backward module outputs (valid after backward done)
        adam_wq0_grad  = attDWq[0];
        adam_wk0_grad  = attDWk[0];
        adam_wv0_grad  = attDWv[0];
        adam_wo0_grad  = attDWo[0];
        adam_wff10_grad = mlpDWff1[0];
        adam_wff20_grad = mlpDWff2[0];
        adam_wq1_grad  = attDWq[1];
        adam_wk1_grad  = attDWk[1];
        adam_wv1_grad  = attDWv[1];
        adam_wo1_grad  = attDWo[1];
        adam_wff11_grad = mlpDWff1[1];
        adam_wff21_grad = mlpDWff2[1];

        // Weight BF16 inputs from layer register arrays
        for (int i = 0; i < D; i++) for (int j = 0; j < DH; j++) begin
            adam_wq0_w[(i*DH+j)*16 +: 16] = layer_Wq[0][i][j];
            adam_wk0_w[(i*DH+j)*16 +: 16] = layer_Wk[0][i][j];
            adam_wv0_w[(i*DH+j)*16 +: 16] = layer_Wv[0][i][j];
            adam_wq1_w[(i*DH+j)*16 +: 16] = layer_Wq[1][i][j];
            adam_wk1_w[(i*DH+j)*16 +: 16] = layer_Wk[1][i][j];
            adam_wv1_w[(i*DH+j)*16 +: 16] = layer_Wv[1][i][j];
        end
        for (int i = 0; i < DH; i++) for (int j = 0; j < D; j++) begin
            adam_wo0_w[(i*D+j)*16 +: 16] = layer_Wo[0][i][j];
            adam_wo1_w[(i*D+j)*16 +: 16] = layer_Wo[1][i][j];
        end
        for (int i = 0; i < D; i++) for (int j = 0; j < FF; j++) begin
            adam_wff10_w[(i*FF+j)*16 +: 16] = layer_Wff1[0][i][j];
            adam_wff11_w[(i*FF+j)*16 +: 16] = layer_Wff1[1][i][j];
        end
        for (int i = 0; i < FF; i++) for (int j = 0; j < D; j++) begin
            adam_wff20_w[(i*D+j)*16 +: 16] = layer_Wff2[0][i][j];
            adam_wff21_w[(i*D+j)*16 +: 16] = layer_Wff2[1][i][j];
        end
    end

    // Adam start + done signals
    logic adam_all_start;
    logic adam_emb_done, adam_wq0_done, adam_wk0_done, adam_wv0_done, adam_wo0_done;
    logic adam_wff10_done, adam_wff20_done;
    logic adam_wq1_done, adam_wk1_done, adam_wv1_done, adam_wo1_done;
    logic adam_wff11_done, adam_wff21_done;

    adam_core #(.R(V),  .C(D))  u_adam_emb   (.clk,.rst,.grad(adam_emb_grad),  .w_bf16_in(adam_emb_w),  .m_in('0),.v_in('0),.lr_fp32(lr_reg),.bc1_fp32(bc1_reg),.bc2_fp32(bc2_reg),.start(adam_all_start),.w_bf16_out(),.m_out(),.v_out(),.done(adam_emb_done));
    adam_core #(.R(D),  .C(DH)) u_adam_wq0   (.clk,.rst,.grad(adam_wq0_grad),  .w_bf16_in(adam_wq0_w), .m_in('0),.v_in('0),.lr_fp32(lr_reg),.bc1_fp32(bc1_reg),.bc2_fp32(bc2_reg),.start(adam_all_start),.w_bf16_out(),.m_out(),.v_out(),.done(adam_wq0_done));
    adam_core #(.R(D),  .C(DH)) u_adam_wk0   (.clk,.rst,.grad(adam_wk0_grad),  .w_bf16_in(adam_wk0_w), .m_in('0),.v_in('0),.lr_fp32(lr_reg),.bc1_fp32(bc1_reg),.bc2_fp32(bc2_reg),.start(adam_all_start),.w_bf16_out(),.m_out(),.v_out(),.done(adam_wk0_done));
    adam_core #(.R(D),  .C(DH)) u_adam_wv0   (.clk,.rst,.grad(adam_wv0_grad),  .w_bf16_in(adam_wv0_w), .m_in('0),.v_in('0),.lr_fp32(lr_reg),.bc1_fp32(bc1_reg),.bc2_fp32(bc2_reg),.start(adam_all_start),.w_bf16_out(),.m_out(),.v_out(),.done(adam_wv0_done));
    adam_core #(.R(DH), .C(D))  u_adam_wo0   (.clk,.rst,.grad(adam_wo0_grad),  .w_bf16_in(adam_wo0_w), .m_in('0),.v_in('0),.lr_fp32(lr_reg),.bc1_fp32(bc1_reg),.bc2_fp32(bc2_reg),.start(adam_all_start),.w_bf16_out(),.m_out(),.v_out(),.done(adam_wo0_done));
    adam_core #(.R(D),  .C(FF)) u_adam_wff10 (.clk,.rst,.grad(adam_wff10_grad),.w_bf16_in(adam_wff10_w),.m_in('0),.v_in('0),.lr_fp32(lr_reg),.bc1_fp32(bc1_reg),.bc2_fp32(bc2_reg),.start(adam_all_start),.w_bf16_out(),.m_out(),.v_out(),.done(adam_wff10_done));
    adam_core #(.R(FF), .C(D))  u_adam_wff20 (.clk,.rst,.grad(adam_wff20_grad),.w_bf16_in(adam_wff20_w),.m_in('0),.v_in('0),.lr_fp32(lr_reg),.bc1_fp32(bc1_reg),.bc2_fp32(bc2_reg),.start(adam_all_start),.w_bf16_out(),.m_out(),.v_out(),.done(adam_wff20_done));
    adam_core #(.R(D),  .C(DH)) u_adam_wq1   (.clk,.rst,.grad(adam_wq1_grad),  .w_bf16_in(adam_wq1_w), .m_in('0),.v_in('0),.lr_fp32(lr_reg),.bc1_fp32(bc1_reg),.bc2_fp32(bc2_reg),.start(adam_all_start),.w_bf16_out(),.m_out(),.v_out(),.done(adam_wq1_done));
    adam_core #(.R(D),  .C(DH)) u_adam_wk1   (.clk,.rst,.grad(adam_wk1_grad),  .w_bf16_in(adam_wk1_w), .m_in('0),.v_in('0),.lr_fp32(lr_reg),.bc1_fp32(bc1_reg),.bc2_fp32(bc2_reg),.start(adam_all_start),.w_bf16_out(),.m_out(),.v_out(),.done(adam_wk1_done));
    adam_core #(.R(D),  .C(DH)) u_adam_wv1   (.clk,.rst,.grad(adam_wv1_grad),  .w_bf16_in(adam_wv1_w), .m_in('0),.v_in('0),.lr_fp32(lr_reg),.bc1_fp32(bc1_reg),.bc2_fp32(bc2_reg),.start(adam_all_start),.w_bf16_out(),.m_out(),.v_out(),.done(adam_wv1_done));
    adam_core #(.R(DH), .C(D))  u_adam_wo1   (.clk,.rst,.grad(adam_wo1_grad),  .w_bf16_in(adam_wo1_w), .m_in('0),.v_in('0),.lr_fp32(lr_reg),.bc1_fp32(bc1_reg),.bc2_fp32(bc2_reg),.start(adam_all_start),.w_bf16_out(),.m_out(),.v_out(),.done(adam_wo1_done));
    adam_core #(.R(D),  .C(FF)) u_adam_wff11 (.clk,.rst,.grad(adam_wff11_grad),.w_bf16_in(adam_wff11_w),.m_in('0),.v_in('0),.lr_fp32(lr_reg),.bc1_fp32(bc1_reg),.bc2_fp32(bc2_reg),.start(adam_all_start),.w_bf16_out(),.m_out(),.v_out(),.done(adam_wff11_done));
    adam_core #(.R(FF), .C(D))  u_adam_wff21 (.clk,.rst,.grad(adam_wff21_grad),.w_bf16_in(adam_wff21_w),.m_in('0),.v_in('0),.lr_fp32(lr_reg),.bc1_fp32(bc1_reg),.bc2_fp32(bc2_reg),.start(adam_all_start),.w_bf16_out(),.m_out(),.v_out(),.done(adam_wff21_done));

    // Latch each adam_core done — done fires for only 1 cycle; the cores have
    // different sizes so they finish at different times. We AND the latches
    // (not the raw done pulses) to detect when ALL cores have finished.
    logic adam_emb_lat, adam_wq0_lat, adam_wk0_lat, adam_wv0_lat, adam_wo0_lat;
    logic adam_wff10_lat, adam_wff20_lat;
    logic adam_wq1_lat, adam_wk1_lat, adam_wv1_lat, adam_wo1_lat;
    logic adam_wff11_lat, adam_wff21_lat;

    always_ff @(posedge clk) begin
        if (rst || adam_all_start) begin
            adam_emb_lat  <= 1'b0; adam_wq0_lat <= 1'b0; adam_wk0_lat  <= 1'b0;
            adam_wv0_lat  <= 1'b0; adam_wo0_lat <= 1'b0; adam_wff10_lat <= 1'b0;
            adam_wff20_lat <= 1'b0;
            adam_wq1_lat  <= 1'b0; adam_wk1_lat <= 1'b0; adam_wv1_lat  <= 1'b0;
            adam_wo1_lat  <= 1'b0; adam_wff11_lat <= 1'b0; adam_wff21_lat <= 1'b0;
        end else begin
            if (adam_emb_done)   adam_emb_lat  <= 1'b1;
            if (adam_wq0_done)   adam_wq0_lat  <= 1'b1;
            if (adam_wk0_done)   adam_wk0_lat  <= 1'b1;
            if (adam_wv0_done)   adam_wv0_lat  <= 1'b1;
            if (adam_wo0_done)   adam_wo0_lat  <= 1'b1;
            if (adam_wff10_done) adam_wff10_lat <= 1'b1;
            if (adam_wff20_done) adam_wff20_lat <= 1'b1;
            if (adam_wq1_done)   adam_wq1_lat  <= 1'b1;
            if (adam_wk1_done)   adam_wk1_lat  <= 1'b1;
            if (adam_wv1_done)   adam_wv1_lat  <= 1'b1;
            if (adam_wo1_done)   adam_wo1_lat  <= 1'b1;
            if (adam_wff11_done) adam_wff11_lat <= 1'b1;
            if (adam_wff21_done) adam_wff21_lat <= 1'b1;
        end
    end

    logic adam_all_done;
    assign adam_all_done = adam_emb_lat  & adam_wq0_lat  & adam_wk0_lat  & adam_wv0_lat  &
                           adam_wo0_lat  & adam_wff10_lat & adam_wff20_lat &
                           adam_wq1_lat  & adam_wk1_lat  & adam_wv1_lat  &
                           adam_wo1_lat  & adam_wff11_lat & adam_wff21_lat;

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    localparam int TD1  = T*D - 1;   // 15
    localparam int Tm1  = T - 1;     // 3
    localparam int DV1  = D*V - 1;   // 63
    localparam int W31  = 31;        // D*DH*4-1 = 31

    typedef enum logic [6:0] {
        IDLE,
        // Layer 0 forward
        WL_ATT_0,
        FWD_LN10_FEED, FWD_LN10_WAIT,
        FWD_ATT0_XLOAD, FWD_ATT0_START, FWD_ATT0_WAIT, FWD_RESID1_0,
        WL_MLP_0,
        FWD_LN20_FEED, FWD_LN20_WAIT,
        FWD_MLP0_XLOAD, FWD_MLP0_START, FWD_MLP0_WAIT, FWD_RESID2_0,
        // Layer 1 forward
        WL_ATT_1,
        FWD_LN11_FEED, FWD_LN11_WAIT,
        FWD_ATT1_XLOAD, FWD_ATT1_START, FWD_ATT1_WAIT, FWD_RESID1_1,
        WL_MLP_1,
        FWD_LN21_FEED, FWD_LN21_WAIT,
        FWD_MLP1_XLOAD, FWD_MLP1_START, FWD_MLP1_WAIT, FWD_RESID2_1,
        // Output projection
        PROJ_LOAD_A, PROJ_LOAD_B, PROJ_START, PROJ_WAIT,
        DONE_ST,
        // Backward layer 1
        BWD_L1_MLP_DY, BWD_L1_MLP_START, BWD_L1_MLP_WAIT,
        BWD_L1_LN2_DY, BWD_L1_LN2_START, BWD_L1_LN2_WAIT, BWD_L1_LN2_RESID,
        BWD_L1_ATT_DY, BWD_L1_ATT_START, BWD_L1_ATT_WAIT,
        BWD_L1_LN1_DY, BWD_L1_LN1_START, BWD_L1_LN1_WAIT, BWD_L1_LN1_RESID,
        // Backward layer 0
        BWD_L0_MLP_DY, BWD_L0_MLP_START, BWD_L0_MLP_WAIT,
        BWD_L0_LN2_DY, BWD_L0_LN2_START, BWD_L0_LN2_WAIT, BWD_L0_LN2_RESID,
        BWD_L0_ATT_DY, BWD_L0_ATT_START, BWD_L0_ATT_WAIT,
        BWD_L0_LN1_DY, BWD_L0_LN1_START, BWD_L0_LN1_WAIT, BWD_L0_LN1_RESID,
        // Cross-entropy backward (compute dLogits, dX_init, dWout inside RTL)
        CE_ROW_MAX,         // find max logit per token row (V cycles)
        CE_ROW_EXP,         // compute exp(logit-max), accumulate sum (V cycles)
        CE_ROW_NORM,        // normalise → dLogits, subtract 1 at target (V cycles)
        CE_DIV_T,           // divide all dLogits by T (T*V cycles)
        CE_GRADNORM,        // sum dLogits^2 (T*V cycles)
        CE_CLIP,            // compute clip_scale (1 cycle)
        CE_CLIP_APPLY,      // multiply dLogits by clip_scale (T*V cycles)
        CE_DX_INIT,         // dX_init[t][d] = sum_v dLogits[t][v]*emb[v][d] (V cycles)
        CE_DWOUT,           // dWout[v][d]  = sum_t dLogits[t][v]*x_reg[t][d] (T cycles)
        // Adam
        ADAM_START, ADAM_WAIT, ADAM_DONE_ST
    } state_t;

    state_t      state;
    logic [6:0]  cnt;
    logic [1:0]  ln_row;  // row counter for LN feed

    // -----------------------------------------------------------------------
    // Combinatorial drive — default all to 0, override per state
    // -----------------------------------------------------------------------
    always_comb begin
        // LN defaults
        for (int k = 0; k < 4; k++) begin
            lnX[k] = '0; lnG[k] = '0; lnB[k] = '0; lnStart[k] = 1'b0;
            lnDyEn[k] = 1'b0; lnDyAddr[k] = '0; lnDyData[k] = '0;
            lnBwdStart[k] = 1'b0;
        end
        // ATT defaults
        for (int k = 0; k < 2; k++) begin
            attXEn[k]=0; attXAddr[k]=0; attXData[k]=0;
            attWqEn[k]=0; attWqAddr[k]=0; attWqData[k]=0;
            attWkEn[k]=0; attWkAddr[k]=0; attWkData[k]=0;
            attWvEn[k]=0; attWvAddr[k]=0; attWvData[k]=0;
            attWoEn[k]=0; attWoAddr[k]=0; attWoData[k]=0;
            attStart[k]=0;
            attDyEn[k]=0; attDyAddr[k]=0; attDyData[k]=0;
            attBwdStart[k]=0;
        end
        // MLP defaults
        for (int k = 0; k < 2; k++) begin
            mlpXEn[k]=0; mlpXAddr[k]=0; mlpXData[k]=0;
            mlpW1En[k]=0; mlpW1Addr[k]=0; mlpW1Data[k]=0;
            mlpW2En[k]=0; mlpW2Addr[k]=0; mlpW2Data[k]=0;
            mlpStart[k]=0;
            mlpDyEn[k]=0; mlpDyAddr[k]=0; mlpDyData[k]=0;
            mlpBwdStart[k]=0;
        end
        // PROJ defaults
        projAEn=0; projAAddr=0; projAData=0;
        projBEn=0; projBAddr=0; projBData=0;
        projStart=0;
        adam_all_start=0;

        case (state)
            // -- Layer 0: load Wq/Wk/Wv/Wo into u_att_inst[0] --
            WL_ATT_0: begin
                if      (cnt[5:4]==2'b00) begin attWqEn[0]=1; attWqAddr[0]={4'b0,cnt[3:0]}; attWqData[0]=layer_Wq[0][cnt[3:2]][cnt[1:0]]; end
                else if (cnt[5:4]==2'b01) begin attWkEn[0]=1; attWkAddr[0]={4'b0,cnt[3:0]}; attWkData[0]=layer_Wk[0][cnt[3:2]][cnt[1:0]]; end
                else if (cnt[5:4]==2'b10) begin attWvEn[0]=1; attWvAddr[0]={4'b0,cnt[3:0]}; attWvData[0]=layer_Wv[0][cnt[3:2]][cnt[1:0]]; end
                else                       begin attWoEn[0]=1; attWoAddr[0]={4'b0,cnt[3:0]}; attWoData[0]=layer_Wo[0][cnt[3:2]][cnt[1:0]]; end
            end
            // LN1 layer 0 feed
            FWD_LN10_FEED: begin
                for (int d = 0; d < D; d++) begin
                    lnX[0][d*32+:32] = x_reg[ln_row][d];
                    lnG[0][d*32+:32] = layer_ln1_gamma[0][d];
                    lnB[0][d*32+:32] = layer_ln1_beta[0][d];
                end
                lnStart[0] = 1'b1;
            end
            // ATT0 X load
            FWD_ATT0_XLOAD: begin attXEn[0]=1; attXAddr[0]={4'b0,cnt[3:0]}; attXData[0]=xnorm1_0_buf[cnt[3:2]][cnt[1:0]]; end
            FWD_ATT0_START: attStart[0] = 1'b1;

            // Layer 0: load Wff1/Wff2 into u_mlp_inst[0]
            WL_MLP_0: begin
                if (!cnt[4]) begin mlpW1En[0]=1; mlpW1Addr[0]={4'b0,cnt[3:0]}; mlpW1Data[0]=layer_Wff1[0][cnt[3:2]][cnt[1:0]]; end
                else         begin mlpW2En[0]=1; mlpW2Addr[0]={4'b0,cnt[3:0]}; mlpW2Data[0]=layer_Wff2[0][cnt[3:2]][cnt[1:0]]; end
            end
            // LN2 layer 0 feed
            FWD_LN20_FEED: begin
                for (int d = 0; d < D; d++) begin
                    lnX[1][d*32+:32] = x_reg[ln_row][d];
                    lnG[1][d*32+:32] = layer_ln2_gamma[0][d];
                    lnB[1][d*32+:32] = layer_ln2_beta[0][d];
                end
                lnStart[1] = 1'b1;
            end
            FWD_MLP0_XLOAD: begin mlpXEn[0]=1; mlpXAddr[0]={4'b0,cnt[3:0]}; mlpXData[0]=xnorm2_0_buf[cnt[3:2]][cnt[1:0]]; end
            FWD_MLP0_START: mlpStart[0] = 1'b1;

            // -- Layer 1: same pattern, index 1 --
            WL_ATT_1: begin
                if      (cnt[5:4]==2'b00) begin attWqEn[1]=1; attWqAddr[1]={4'b0,cnt[3:0]}; attWqData[1]=layer_Wq[1][cnt[3:2]][cnt[1:0]]; end
                else if (cnt[5:4]==2'b01) begin attWkEn[1]=1; attWkAddr[1]={4'b0,cnt[3:0]}; attWkData[1]=layer_Wk[1][cnt[3:2]][cnt[1:0]]; end
                else if (cnt[5:4]==2'b10) begin attWvEn[1]=1; attWvAddr[1]={4'b0,cnt[3:0]}; attWvData[1]=layer_Wv[1][cnt[3:2]][cnt[1:0]]; end
                else                       begin attWoEn[1]=1; attWoAddr[1]={4'b0,cnt[3:0]}; attWoData[1]=layer_Wo[1][cnt[3:2]][cnt[1:0]]; end
            end
            FWD_LN11_FEED: begin
                for (int d = 0; d < D; d++) begin
                    lnX[2][d*32+:32] = x_reg[ln_row][d];
                    lnG[2][d*32+:32] = layer_ln1_gamma[1][d];
                    lnB[2][d*32+:32] = layer_ln1_beta[1][d];
                end
                lnStart[2] = 1'b1;
            end
            FWD_ATT1_XLOAD: begin attXEn[1]=1; attXAddr[1]={4'b0,cnt[3:0]}; attXData[1]=xnorm1_1_buf[cnt[3:2]][cnt[1:0]]; end
            FWD_ATT1_START: attStart[1] = 1'b1;
            WL_MLP_1: begin
                if (!cnt[4]) begin mlpW1En[1]=1; mlpW1Addr[1]={4'b0,cnt[3:0]}; mlpW1Data[1]=layer_Wff1[1][cnt[3:2]][cnt[1:0]]; end
                else         begin mlpW2En[1]=1; mlpW2Addr[1]={4'b0,cnt[3:0]}; mlpW2Data[1]=layer_Wff2[1][cnt[3:2]][cnt[1:0]]; end
            end
            FWD_LN21_FEED: begin
                for (int d = 0; d < D; d++) begin
                    lnX[3][d*32+:32] = x_reg[ln_row][d];
                    lnG[3][d*32+:32] = layer_ln2_gamma[1][d];
                    lnB[3][d*32+:32] = layer_ln2_beta[1][d];
                end
                lnStart[3] = 1'b1;
            end
            FWD_MLP1_XLOAD: begin mlpXEn[1]=1; mlpXAddr[1]={4'b0,cnt[3:0]}; mlpXData[1]=xnorm2_1_buf[cnt[3:2]][cnt[1:0]]; end
            FWD_MLP1_START: mlpStart[1] = 1'b1;

            // Output projection
            PROJ_LOAD_A: begin projAEn=1; projAAddr={4'b0,cnt[3:0]}; projAData=x_reg[cnt[3:2]][cnt[1:0]]; end
            PROJ_LOAD_B: begin
                projBEn=1; projBAddr={2'b0,cnt[5:0]};
                projBData = emb_reg[cnt[3:0]][cnt[5:4]];
            end
            PROJ_START: projStart = 1'b1;

            // ---- Backward layer 1 ----
            // MLP1 dy load: dx_buf → mlp_inst[1] dy
            BWD_L1_MLP_DY: begin
                mlpDyEn[1]=1; mlpDyAddr[1]={4'b0,cnt[3:0]};
                mlpDyData[1]=dx_buf[cnt[3:2]][cnt[1:0]];
            end
            BWD_L1_MLP_START: mlpBwdStart[1]=1;
            // LN2_1 dy load: dX_tmp → ln_inst[3] dy
            BWD_L1_LN2_DY: begin
                lnDyEn[3]=1; lnDyAddr[3]={4'b0,cnt[3:0]};
                lnDyData[3]=dX_tmp[cnt[3:2]][cnt[1:0]];
            end
            BWD_L1_LN2_START: lnBwdStart[3]=1;
            // ATT1 dy load: dAttn_tmp → att_inst[1] dy
            BWD_L1_ATT_DY: begin
                attDyEn[1]=1; attDyAddr[1]={4'b0,cnt[3:0]};
                attDyData[1]=dAttn_tmp[cnt[3:2]][cnt[1:0]];
            end
            BWD_L1_ATT_START: attBwdStart[1]=1;
            // LN1_1 dy load: dX_tmp → ln_inst[2] dy
            BWD_L1_LN1_DY: begin
                lnDyEn[2]=1; lnDyAddr[2]={4'b0,cnt[3:0]};
                lnDyData[2]=dX_tmp[cnt[3:2]][cnt[1:0]];
            end
            BWD_L1_LN1_START: lnBwdStart[2]=1;

            // ---- Backward layer 0 ----
            BWD_L0_MLP_DY: begin
                mlpDyEn[0]=1; mlpDyAddr[0]={4'b0,cnt[3:0]};
                mlpDyData[0]=dx_buf[cnt[3:2]][cnt[1:0]];
            end
            BWD_L0_MLP_START: mlpBwdStart[0]=1;
            BWD_L0_LN2_DY: begin
                lnDyEn[1]=1; lnDyAddr[1]={4'b0,cnt[3:0]};
                lnDyData[1]=dX_tmp[cnt[3:2]][cnt[1:0]];
            end
            BWD_L0_LN2_START: lnBwdStart[1]=1;
            BWD_L0_ATT_DY: begin
                attDyEn[0]=1; attDyAddr[0]={4'b0,cnt[3:0]};
                attDyData[0]=dAttn_tmp[cnt[3:2]][cnt[1:0]];
            end
            BWD_L0_ATT_START: attBwdStart[0]=1;
            BWD_L0_LN1_DY: begin
                lnDyEn[0]=1; lnDyAddr[0]={4'b0,cnt[3:0]};
                lnDyData[0]=dX_tmp[cnt[3:2]][cnt[1:0]];
            end
            BWD_L0_LN1_START: lnBwdStart[0]=1;

            ADAM_START: adam_all_start=1;

            default: ;
        endcase
    end

    // -----------------------------------------------------------------------
    // FSM sequential
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE; cnt <= '0; ln_row <= '0;
            done <= 1'b0; out_valid <= 1'b0; out_row <= '0; out_row_idx <= '0;
            adam_done <= 1'b0;
        end else if (en) begin
            done <= 1'b0; out_valid <= 1'b0; adam_done <= 1'b0;

            case (state)
                IDLE: if (start) begin
                    for (int t = 0; t < T; t++) for (int d = 0; d < D; d++)
                        x_reg[t][d] <= x_init_reg[t][d];
                    cnt <= '0; ln_row <= '0; state <= WL_ATT_0;
                end

                // ---- Layer 0 forward ----
                WL_ATT_0:  begin cnt<=cnt+1; if (cnt==7'd63) begin cnt<='0; ln_row<='0; state<=FWD_LN10_FEED; end end
                FWD_LN10_FEED: state <= FWD_LN10_WAIT;
                FWD_LN10_WAIT: if (lnValid[0]) begin
                    for (int d = 0; d < D; d++) xnorm1_0_buf[ln_row][d] <= lnY[0][d*32+:32];
                    if (ln_row==2'(Tm1)) begin ln_row<='0; cnt<='0; state<=FWD_ATT0_XLOAD; end
                    else                 begin ln_row<=ln_row+1; state<=FWD_LN10_FEED; end
                end
                FWD_ATT0_XLOAD: begin cnt<=cnt+1; if (cnt==7'(TD1)) begin cnt<='0; state<=FWD_ATT0_START; end end
                FWD_ATT0_START: state <= FWD_ATT0_WAIT;
                FWD_ATT0_WAIT: if (attOutValid[0]) begin
                    for (int d = 0; d < D; d++) attn_buf[attOutIdx[0]][d] <= attOutRow[0][d*32+:32];
                    if (attOutIdx[0]==2'(Tm1)) state <= FWD_RESID1_0;
                end
                FWD_RESID1_0: begin
                    for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) begin
                        automatic shortreal xa, xb;
                        xa=$bitstoshortreal(x_reg[t][d]); xb=$bitstoshortreal(attn_buf[t][d]);
                        x_reg[t][d] <= $shortrealtobits(xa+xb);
                    end
                    cnt<='0; ln_row<='0; state<=WL_MLP_0;
                end

                WL_MLP_0:  begin cnt<=cnt+1; if (cnt==7'd31) begin cnt<='0; ln_row<='0; state<=FWD_LN20_FEED; end end
                FWD_LN20_FEED: state <= FWD_LN20_WAIT;
                FWD_LN20_WAIT: if (lnValid[1]) begin
                    for (int d = 0; d < D; d++) xnorm2_0_buf[ln_row][d] <= lnY[1][d*32+:32];
                    if (ln_row==2'(Tm1)) begin ln_row<='0; cnt<='0; state<=FWD_MLP0_XLOAD; end
                    else                 begin ln_row<=ln_row+1; state<=FWD_LN20_FEED; end
                end
                FWD_MLP0_XLOAD: begin cnt<=cnt+1; if (cnt==7'(TD1)) begin cnt<='0; state<=FWD_MLP0_START; end end
                FWD_MLP0_START: state <= FWD_MLP0_WAIT;
                FWD_MLP0_WAIT: if (mlpOutValid[0]) begin
                    for (int d = 0; d < D; d++) mlp_buf[mlpOutIdx[0]][d] <= mlpOutRow[0][d*32+:32];
                    if (mlpOutIdx[0]==2'(Tm1)) state <= FWD_RESID2_0;
                end
                FWD_RESID2_0: begin
                    for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) begin
                        automatic shortreal xa, xb;
                        xa=$bitstoshortreal(x_reg[t][d]); xb=$bitstoshortreal(mlp_buf[t][d]);
                        x_reg[t][d] <= $shortrealtobits(xa+xb);
                    end
                    cnt<='0; ln_row<='0; state<=WL_ATT_1;
                end

                // ---- Layer 1 forward ----
                WL_ATT_1:  begin cnt<=cnt+1; if (cnt==7'd63) begin cnt<='0; ln_row<='0; state<=FWD_LN11_FEED; end end
                FWD_LN11_FEED: state <= FWD_LN11_WAIT;
                FWD_LN11_WAIT: if (lnValid[2]) begin
                    for (int d = 0; d < D; d++) xnorm1_1_buf[ln_row][d] <= lnY[2][d*32+:32];
                    if (ln_row==2'(Tm1)) begin ln_row<='0; cnt<='0; state<=FWD_ATT1_XLOAD; end
                    else                 begin ln_row<=ln_row+1; state<=FWD_LN11_FEED; end
                end
                FWD_ATT1_XLOAD: begin cnt<=cnt+1; if (cnt==7'(TD1)) begin cnt<='0; state<=FWD_ATT1_START; end end
                FWD_ATT1_START: state <= FWD_ATT1_WAIT;
                FWD_ATT1_WAIT: if (attOutValid[1]) begin
                    for (int d = 0; d < D; d++) attn_buf[attOutIdx[1]][d] <= attOutRow[1][d*32+:32];
                    if (attOutIdx[1]==2'(Tm1)) state <= FWD_RESID1_1;
                end
                FWD_RESID1_1: begin
                    for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) begin
                        automatic shortreal xa, xb;
                        xa=$bitstoshortreal(x_reg[t][d]); xb=$bitstoshortreal(attn_buf[t][d]);
                        x_reg[t][d] <= $shortrealtobits(xa+xb);
                    end
                    cnt<='0; ln_row<='0; state<=WL_MLP_1;
                end

                WL_MLP_1:  begin cnt<=cnt+1; if (cnt==7'd31) begin cnt<='0; ln_row<='0; state<=FWD_LN21_FEED; end end
                FWD_LN21_FEED: state <= FWD_LN21_WAIT;
                FWD_LN21_WAIT: if (lnValid[3]) begin
                    for (int d = 0; d < D; d++) xnorm2_1_buf[ln_row][d] <= lnY[3][d*32+:32];
                    if (ln_row==2'(Tm1)) begin ln_row<='0; cnt<='0; state<=FWD_MLP1_XLOAD; end
                    else                 begin ln_row<=ln_row+1; state<=FWD_LN21_FEED; end
                end
                FWD_MLP1_XLOAD: begin cnt<=cnt+1; if (cnt==7'(TD1)) begin cnt<='0; state<=FWD_MLP1_START; end end
                FWD_MLP1_START: state <= FWD_MLP1_WAIT;
                FWD_MLP1_WAIT: if (mlpOutValid[1]) begin
                    for (int d = 0; d < D; d++) mlp_buf[mlpOutIdx[1]][d] <= mlpOutRow[1][d*32+:32];
                    if (mlpOutIdx[1]==2'(Tm1)) state <= FWD_RESID2_1;
                end
                FWD_RESID2_1: begin
                    for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) begin
                        automatic shortreal xa, xb;
                        xa=$bitstoshortreal(x_reg[t][d]); xb=$bitstoshortreal(mlp_buf[t][d]);
                        x_reg[t][d] <= $shortrealtobits(xa+xb);
                    end
                    cnt<='0; state<=PROJ_LOAD_A;
                end

                // Output projection
                PROJ_LOAD_A: begin cnt<=cnt+1; if (cnt==7'(TD1)) begin cnt<='0; state<=PROJ_LOAD_B; end end
                PROJ_LOAD_B: begin cnt<=cnt+1; if (cnt==7'(DV1)) begin cnt<='0; state<=PROJ_START; end end
                PROJ_START:  state <= PROJ_WAIT;
                PROJ_WAIT: begin
                    out_valid <= projCValid; out_row <= projCRow; out_row_idx <= projCIdx;
                    if (projCValid) begin
                        // Buffer logits for internal cross-entropy computation
                        for (int v = 0; v < V; v++)
                            logits_reg[projCIdx][v] <= projCRow[v*32 +: 32];
                        if (projCIdx==2'(Tm1)) state <= DONE_ST;
                    end
                end

                DONE_ST: begin done <= 1'b1; state <= IDLE; end

                // ================================================================
                // BACKWARD pass — entered from IDLE when train_start received.
                // Testbench must write dX_init, dWout, lr, bc1, bc2 before asserting.
                // ================================================================

                // ---- Layer 1 backward ----
                // MLP1 backward: feed dx_buf (gradient from above) into u_mlp_inst[1]
                BWD_L1_MLP_DY: begin
                    cnt<=cnt+1;
                    if (cnt==7'(TD1)) begin cnt<='0; state<=BWD_L1_MLP_START; end
                end
                BWD_L1_MLP_START: state <= BWD_L1_MLP_WAIT;
                BWD_L1_MLP_WAIT: if (mlpDxValid[1]) begin
                    for (int d = 0; d < D; d++) dX_tmp[mlpDxIdx[1]][d] <= mlpDxRow[1][d*32+:32];
                    if (mlpDxIdx[1]==2'(Tm1)) begin cnt<='0; state<=BWD_L1_LN2_DY; end
                end

                // LN2_1 backward: feed dX_tmp into u_ln_inst[3]
                BWD_L1_LN2_DY: begin
                    cnt<=cnt+1;
                    if (cnt==7'(TD1)) begin cnt<='0; state<=BWD_L1_LN2_START; end
                end
                BWD_L1_LN2_START: state <= BWD_L1_LN2_WAIT;
                BWD_L1_LN2_WAIT: if (lnDxValid[3]) begin
                    for (int d = 0; d < D; d++) begin
                        // Also compute dAttn_tmp = dX_ln2 + dx_buf (FF skip) on last row
                        automatic shortreal xa, xb;
                        xa=$bitstoshortreal(lnDx[3][d*32+:32]);
                        xb=$bitstoshortreal(dx_buf[lnDxIdx[3]][d]);
                        dAttn_tmp[lnDxIdx[3]][d] <= $shortrealtobits(xa+xb);
                    end
                    if (lnDxIdx[3]==2'(Tm1)) state<=BWD_L1_LN2_RESID;
                end
                BWD_L1_LN2_RESID: begin
                    // dAttn_tmp is already computed above; just move to ATT DY
                    cnt<='0; state<=BWD_L1_ATT_DY;
                end

                // ATT1 backward: feed dAttn_tmp into u_att_inst[1]
                BWD_L1_ATT_DY: begin
                    cnt<=cnt+1;
                    if (cnt==7'(TD1)) begin cnt<='0; state<=BWD_L1_ATT_START; end
                end
                BWD_L1_ATT_START: state <= BWD_L1_ATT_WAIT;
                BWD_L1_ATT_WAIT: if (attDxValid[1]) begin
                    for (int d = 0; d < D; d++) dX_tmp[attDxIdx[1]][d] <= attDxRow[1][d*32+:32];
                    if (attDxIdx[1]==2'(Tm1)) begin cnt<='0; state<=BWD_L1_LN1_DY; end
                end

                // LN1_1 backward: feed dX_tmp into u_ln_inst[2]
                BWD_L1_LN1_DY: begin
                    cnt<=cnt+1;
                    if (cnt==7'(TD1)) begin cnt<='0; state<=BWD_L1_LN1_START; end
                end
                BWD_L1_LN1_START: state <= BWD_L1_LN1_WAIT;
                BWD_L1_LN1_WAIT: if (lnDxValid[2]) begin
                    // Compute dx_buf = dX_ln1 + dAttn_tmp (ATT skip)
                    for (int d = 0; d < D; d++) begin
                        automatic shortreal xa, xb;
                        xa=$bitstoshortreal(lnDx[2][d*32+:32]);
                        xb=$bitstoshortreal(dAttn_tmp[lnDxIdx[2]][d]);
                        dx_buf[lnDxIdx[2]][d] <= $shortrealtobits(xa+xb);
                    end
                    if (lnDxIdx[2]==2'(Tm1)) state<=BWD_L1_LN1_RESID;
                end
                BWD_L1_LN1_RESID: begin cnt<='0; state<=BWD_L0_MLP_DY; end  // dx_buf ready

                // ---- Layer 0 backward ----
                BWD_L0_MLP_DY: begin
                    cnt<=cnt+1;
                    if (cnt==7'(TD1)) begin cnt<='0; state<=BWD_L0_MLP_START; end
                end
                BWD_L0_MLP_START: state <= BWD_L0_MLP_WAIT;
                BWD_L0_MLP_WAIT: if (mlpDxValid[0]) begin
                    for (int d = 0; d < D; d++) dX_tmp[mlpDxIdx[0]][d] <= mlpDxRow[0][d*32+:32];
                    if (mlpDxIdx[0]==2'(Tm1)) begin cnt<='0; state<=BWD_L0_LN2_DY; end
                end

                BWD_L0_LN2_DY: begin
                    cnt<=cnt+1;
                    if (cnt==7'(TD1)) begin cnt<='0; state<=BWD_L0_LN2_START; end
                end
                BWD_L0_LN2_START: state <= BWD_L0_LN2_WAIT;
                BWD_L0_LN2_WAIT: if (lnDxValid[1]) begin
                    for (int d = 0; d < D; d++) begin
                        automatic shortreal xa, xb;
                        xa=$bitstoshortreal(lnDx[1][d*32+:32]);
                        xb=$bitstoshortreal(dx_buf[lnDxIdx[1]][d]);
                        dAttn_tmp[lnDxIdx[1]][d] <= $shortrealtobits(xa+xb);
                    end
                    if (lnDxIdx[1]==2'(Tm1)) state<=BWD_L0_LN2_RESID;
                end
                BWD_L0_LN2_RESID: begin cnt<='0; state<=BWD_L0_ATT_DY; end

                BWD_L0_ATT_DY: begin
                    cnt<=cnt+1;
                    if (cnt==7'(TD1)) begin cnt<='0; state<=BWD_L0_ATT_START; end
                end
                BWD_L0_ATT_START: state <= BWD_L0_ATT_WAIT;
                BWD_L0_ATT_WAIT: if (attDxValid[0]) begin
                    for (int d = 0; d < D; d++) dX_tmp[attDxIdx[0]][d] <= attDxRow[0][d*32+:32];
                    if (attDxIdx[0]==2'(Tm1)) begin cnt<='0; state<=BWD_L0_LN1_DY; end
                end

                BWD_L0_LN1_DY: begin
                    cnt<=cnt+1;
                    if (cnt==7'(TD1)) begin cnt<='0; state<=BWD_L0_LN1_START; end
                end
                BWD_L0_LN1_START: state <= BWD_L0_LN1_WAIT;
                BWD_L0_LN1_WAIT: if (lnDxValid[0]) begin
                    for (int d = 0; d < D; d++) begin
                        automatic shortreal xa, xb;
                        xa=$bitstoshortreal(lnDx[0][d*32+:32]);
                        xb=$bitstoshortreal(dAttn_tmp[lnDxIdx[0]][d]);
                        dx_buf[lnDxIdx[0]][d] <= $shortrealtobits(xa+xb);
                    end
                    if (lnDxIdx[0]==2'(Tm1)) state<=BWD_L0_LN1_RESID;
                end
                BWD_L0_LN1_RESID: begin state<=ADAM_START; end

                // ---- Adam ----
                ADAM_START: state <= ADAM_WAIT;  // adam_all_start driven combinatorially

                ADAM_WAIT: if (adam_all_done) state <= ADAM_DONE_ST;

                ADAM_DONE_ST: begin adam_done <= 1'b1; state <= IDLE; end

                // =============================================================
                // CROSS-ENTROPY STATES — compute dLogits, dX_init, dWout inside RTL
                // =============================================================

                // Per-row softmax over V=16 logits:
                //   Step 1: find row max (numerical stability)
                CE_ROW_MAX: begin
                    begin
                        automatic shortreal lv = $bitstoshortreal(logits_reg[ln_row][cnt[3:0]]);
                        if (cnt[3:0] == '0) max_logit_sr <= lv;
                        else if (lv > max_logit_sr) max_logit_sr <= lv;
                    end
                    if (cnt[3:0] == 4'(V-1)) begin cnt <= '0; exp_sum_sr <= 0.0; state <= CE_ROW_EXP; end
                    else cnt <= cnt + 1;
                end

                //   Step 2: compute exp(logit - max), accumulate sum
                CE_ROW_EXP: begin
                    begin
                        automatic shortreal lv = $bitstoshortreal(logits_reg[ln_row][cnt[3:0]]);
                        automatic shortreal e  = $exp(lv - max_logit_sr);
                        exp_buf_sr[cnt[3:0]] <= e;
                        exp_sum_sr <= exp_sum_sr + e;
                    end
                    if (cnt[3:0] == 4'(V-1)) begin cnt <= '0; state <= CE_ROW_NORM; end
                    else cnt <= cnt + 1;
                end

                //   Step 3: dLogits[t][v] = exp/sum; subtract 1 at target
                CE_ROW_NORM: begin
                    begin
                        automatic shortreal dl = exp_buf_sr[cnt[3:0]] / exp_sum_sr;
                        if (cnt[3:0] == 4'(targets_reg[ln_row])) dl = dl - 1.0;
                        dLogits_reg[ln_row][cnt[3:0]] <= $shortrealtobits(dl);
                    end
                    if (cnt[3:0] == 4'(V-1)) begin
                        if (ln_row == 2'(Tm1)) begin
                            cnt <= '0; state <= CE_DIV_T;   // all rows done
                        end else begin
                            ln_row <= ln_row + 1;
                            cnt <= '0; max_logit_sr <= 0.0; exp_sum_sr <= 0.0;
                            state <= CE_ROW_MAX;
                        end
                    end else cnt <= cnt + 1;
                end

                // Divide all dLogits by T (seqLen normalisation), flat counter 0..T*V-1
                CE_DIV_T: begin
                    begin
                        automatic shortreal dl = $bitstoshortreal(dLogits_reg[cnt[5:4]][cnt[3:0]]);
                        dLogits_reg[cnt[5:4]][cnt[3:0]] <= $shortrealtobits(dl / shortreal'(T));
                    end
                    if (cnt[5:0] == 6'(T*V-1)) begin cnt <= '0; grad_norm_sq_sr <= 0.0; state <= CE_GRADNORM; end
                    else cnt <= cnt + 1;
                end

                // Compute sum(dLogits^2) for grad norm clipping
                CE_GRADNORM: begin
                    begin
                        automatic shortreal dl = $bitstoshortreal(dLogits_reg[cnt[5:4]][cnt[3:0]]);
                        grad_norm_sq_sr <= grad_norm_sq_sr + dl * dl;
                    end
                    if (cnt[5:0] == 6'(T*V-1)) begin cnt <= '0; state <= CE_CLIP; end
                    else cnt <= cnt + 1;
                end

                // Compute clip_scale (1 cycle)
                CE_CLIP: begin
                    begin
                        automatic shortreal gnorm = $sqrt(grad_norm_sq_sr);
                        clip_scale_sr <= (gnorm > 1.0) ? (1.0 / gnorm) : 1.0;
                    end
                    cnt <= '0; state <= CE_CLIP_APPLY;
                end

                // Apply clip to all T*V dLogits
                CE_CLIP_APPLY: begin
                    begin
                        automatic shortreal dl = $bitstoshortreal(dLogits_reg[cnt[5:4]][cnt[3:0]]);
                        dLogits_reg[cnt[5:4]][cnt[3:0]] <= $shortrealtobits(dl * clip_scale_sr);
                    end
                    if (cnt[5:0] == 6'(T*V-1)) begin cnt <= '0; state <= CE_DX_INIT; end
                    else cnt <= cnt + 1;
                end

                // dX_init[t][d] = sum_v dLogits[t][v] * emb[v][d]
                // For each v in parallel over all (t,d) — V=16 cycles
                CE_DX_INIT: begin
                    for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) begin
                        automatic shortreal dl = $bitstoshortreal(dLogits_reg[t][cnt[3:0]]);
                        automatic shortreal ew = $bitstoshortreal(emb_reg[cnt[3:0]][d]);
                        if (cnt[3:0] == '0)
                            dX_init_reg[t][d] <= $shortrealtobits(dl * ew);
                        else
                            dX_init_reg[t][d] <= $shortrealtobits(
                                $bitstoshortreal(dX_init_reg[t][d]) + dl * ew);
                    end
                    if (cnt[3:0] == 4'(V-1)) begin cnt <= '0; state <= CE_DWOUT; end
                    else cnt <= cnt + 1;
                end

                // dWout[v][d] = sum_t dLogits[t][v] * x_reg[t][d]
                // For each t in parallel over all (v,d) — T=4 cycles
                CE_DWOUT: begin
                    for (int v = 0; v < V; v++) for (int d = 0; d < D; d++) begin
                        automatic shortreal dl = $bitstoshortreal(dLogits_reg[cnt[1:0]][v]);
                        automatic shortreal xv = $bitstoshortreal(x_reg[cnt[1:0]][d]);
                        if (cnt[1:0] == '0)
                            dWout_reg[v][d] <= $shortrealtobits(dl * xv);
                        else
                            dWout_reg[v][d] <= $shortrealtobits(
                                $bitstoshortreal(dWout_reg[v][d]) + dl * xv);
                    end
                    if (cnt[1:0] == 2'(Tm1)) begin
                        // Load dx_buf from dX_init_reg, start backward pass
                        for (int t = 0; t < T; t++) for (int d = 0; d < D; d++)
                            dx_buf[t][d] <= dX_init_reg[t][d];
                        cnt <= '0; state <= BWD_L1_MLP_DY;
                    end else cnt <= cnt + 1;
                end

                default: state <= IDLE;
            endcase

            // ---- IDLE: train_start triggers CE → backward → Adam ----
            if (state == IDLE && train_start && !start) begin
                cnt <= '0; ln_row <= '0; max_logit_sr <= 0.0;
                state <= CE_ROW_MAX;
            end
        end
    end

endmodule
