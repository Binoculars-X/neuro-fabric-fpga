// transformer.sv
// Forward-only transformer pass (Phase 1: step 7a).
//
// Matches TransformerBus.Forward() in C#:
//   x = embedding_lookup(tokens) + PE   [loaded as x_init via write bus]
//   for l in 0..L-1:
//     xn1 = LayerNorm1(x)
//     attn = AttentionCore.Forward(xn1)
//     x    = x + attn
//     xn2  = LayerNorm2(x)
//     mlp  = MlpCore.Forward(xn2)
//     x    = x + mlp
//   logits = x * emb_reg^T            [fp32_matmul, T*D . D*V -> T*V]
//
// Model configuration (hardcoded):
//   T=4, D=4, DH=4 (numHeads=1), FF=4, L=2, V=16
//
// Write bus address map (wr_addr [8:0]):
//   0x000..0x03F  emb_reg[v][d]    FP32
//   0x040..0x04F  x_init[t][d]     FP32
//   Layer l (base: l=0 -> 0x050, l=1 -> 0x0C0, stride 0x070; 112 entries each):
//     loffset 0x00..0x0F: ln1_gamma[l], ln1_beta[l], ln2_gamma[l], ln2_beta[l] (4 FP32 each)
//     loffset 0x10..0x1F: Wq  BF16 (wr_data[15:0])
//     loffset 0x20..0x2F: Wk  BF16
//     loffset 0x30..0x3F: Wv  BF16
//     loffset 0x40..0x4F: Wo  BF16
//     loffset 0x50..0x5F: Wff1 BF16
//     loffset 0x60..0x6F: Wff2 BF16
//   Total write bus entries: 64 + 16 + 2*112 = 304
//
// Note: NO final LayerNorm -- matches C# TransformerBus.Forward() exactly.

`timescale 1ns/1ps

module transformer #(
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

    input  logic        wr_en,
    input  logic [8:0]  wr_addr,
    input  logic [31:0] wr_data,

    input  logic start,

    output logic [V*32-1:0] out_row,
    output logic             out_valid,
    output logic [1:0]       out_row_idx,

    output logic done
);

    // -----------------------------------------------------------------------
    // Registers
    // -----------------------------------------------------------------------
    logic [31:0] emb_reg     [0:V-1][0:D-1];
    logic [31:0] x_init_reg  [0:T-1][0:D-1];
    logic [31:0] x_reg       [0:T-1][0:D-1];
    logic [31:0] xnorm1_buf  [0:T-1][0:D-1];
    logic [31:0] xnorm2_buf  [0:T-1][0:D-1];
    logic [31:0] attn_buf    [0:T-1][0:D-1];
    logic [31:0] mlp_buf     [0:T-1][0:D-1];

    logic [31:0] layer_ln1_gamma [0:L-1][0:D-1];
    logic [31:0] layer_ln1_beta  [0:L-1][0:D-1];
    logic [31:0] layer_ln2_gamma [0:L-1][0:D-1];
    logic [31:0] layer_ln2_beta  [0:L-1][0:D-1];
    logic [15:0] layer_Wq        [0:L-1][0:D-1][0:DH-1];
    logic [15:0] layer_Wk        [0:L-1][0:D-1][0:DH-1];
    logic [15:0] layer_Wv        [0:L-1][0:D-1][0:DH-1];
    logic [15:0] layer_Wo        [0:L-1][0:DH-1][0:D-1];
    logic [15:0] layer_Wff1      [0:L-1][0:D-1][0:FF-1];
    logic [15:0] layer_Wff2      [0:L-1][0:FF-1][0:D-1];

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

            // Layer 0: addr 0x050..0x0BF  loffset = addr[7:0] - 0x50
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

            // Layer 1: addr 0x0C0..0x12F  loffset = addr[7:0] - 0xC0
            // (addr 0x100..0x12F: addr[7:0]=0x00..0x2F, 0x00-0xC0=0x40 in 8-bit unsigned)
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
        end
    end

    // -----------------------------------------------------------------------
    // Sub-module signals
    // -----------------------------------------------------------------------
    logic [D*32-1:0] ln_x_in, ln_gamma, ln_beta, ln_y_out;
    logic            ln_start, ln_out_valid;

    logic        att_x_wr_en;  logic [7:0] att_x_wr_addr;  logic [31:0] att_x_wr_data;
    logic        att_wq_wr_en; logic [7:0] att_wq_wr_addr; logic [15:0] att_wq_wr_data;
    logic        att_wk_wr_en; logic [7:0] att_wk_wr_addr; logic [15:0] att_wk_wr_data;
    logic        att_wv_wr_en; logic [7:0] att_wv_wr_addr; logic [15:0] att_wv_wr_data;
    logic        att_wo_wr_en; logic [7:0] att_wo_wr_addr; logic [15:0] att_wo_wr_data;
    logic        att_start;
    logic [D*32-1:0] att_out_row; logic att_out_valid; logic [1:0] att_out_row_idx;

    logic        mlp_x_wr_en;    logic [7:0] mlp_x_wr_addr;    logic [31:0] mlp_x_wr_data;
    logic        mlp_wff1_wr_en; logic [7:0] mlp_wff1_wr_addr; logic [15:0] mlp_wff1_wr_data;
    logic        mlp_wff2_wr_en; logic [7:0] mlp_wff2_wr_addr; logic [15:0] mlp_wff2_wr_data;
    logic        mlp_start;
    logic [D*32-1:0] mlp_out_row; logic mlp_out_valid; logic [1:0] mlp_out_row_idx;

    logic        proj_a_wr_en; logic [7:0] proj_a_wr_addr; logic [31:0] proj_a_wr_data;
    logic        proj_b_wr_en; logic [7:0] proj_b_wr_addr; logic [31:0] proj_b_wr_data;
    logic        proj_start;
    logic [V*32-1:0] proj_c_row; logic proj_c_valid; logic [1:0] proj_c_row_idx;

    // -----------------------------------------------------------------------
    // Sub-module instances
    // -----------------------------------------------------------------------
    layernorm #(.D(D), .T(T)) u_ln (
        .clk(clk), .rst(rst), .en(en),
        .x_in(ln_x_in), .gamma(ln_gamma), .beta(ln_beta),
        .start(ln_start), .y_out(ln_y_out), .out_valid(ln_out_valid),
        // Backward ports — not yet wired (7b-iv)
        .dy_wr_en(1'b0), .dy_wr_addr(8'h0), .dy_wr_data(32'h0),
        .bwd_start(1'b0),
        .dx_row(), .dx_valid(), .dx_row_idx(),
        .dGamma_flat(), .dBeta_flat()
    );

    attention_core #(
        .T(T), .D(D), .DH(DH), .MAC_LAT(MAC_LAT), .MUL_LAT(MUL_LAT),
        .EXP_LAT(EXP_LAT), .LUT_SIZE(LUT_SIZE), .LUT_FILE(LUT_FILE)
    ) u_att (
        .clk(clk), .rst(rst), .en(en),
        .x_wr_en(att_x_wr_en), .x_wr_addr(att_x_wr_addr), .x_wr_data(att_x_wr_data),
        .wq_wr_en(att_wq_wr_en), .wq_wr_addr(att_wq_wr_addr), .wq_wr_data(att_wq_wr_data),
        .wk_wr_en(att_wk_wr_en), .wk_wr_addr(att_wk_wr_addr), .wk_wr_data(att_wk_wr_data),
        .wv_wr_en(att_wv_wr_en), .wv_wr_addr(att_wv_wr_addr), .wv_wr_data(att_wv_wr_data),
        .wo_wr_en(att_wo_wr_en), .wo_wr_addr(att_wo_wr_addr), .wo_wr_data(att_wo_wr_data),
        .start(att_start),
        .out_row(att_out_row), .out_valid(att_out_valid), .out_row_idx(att_out_row_idx),
        // Backward ports — not yet wired (7b-iv)
        .dy_wr_en(1'b0), .dy_wr_addr(8'h0), .dy_wr_data(32'h0),
        .bwd_start(1'b0),
        .dx_row(), .dx_valid(), .dx_row_idx(),
        .dWq_flat(), .dWk_flat(), .dWv_flat(), .dWo_flat()
    );

    mlp_core #(
        .T(T), .D(D), .FF(FF), .MAC_LAT(MAC_LAT), .EXP_LAT(EXP_LAT),
        .LUT_SIZE(LUT_SIZE), .LUT_FILE(LUT_FILE)
    ) u_mlp (
        .clk(clk), .rst(rst), .en(en),
        .x_wr_en(mlp_x_wr_en), .x_wr_addr(mlp_x_wr_addr), .x_wr_data(mlp_x_wr_data),
        .wff1_wr_en(mlp_wff1_wr_en), .wff1_wr_addr(mlp_wff1_wr_addr), .wff1_wr_data(mlp_wff1_wr_data),
        .wff2_wr_en(mlp_wff2_wr_en), .wff2_wr_addr(mlp_wff2_wr_addr), .wff2_wr_data(mlp_wff2_wr_data),
        .start(mlp_start),
        .out_row(mlp_out_row), .out_valid(mlp_out_valid), .out_row_idx(mlp_out_row_idx),
        // Backward ports — not yet wired (7b-iv will connect)
        .dy_wr_en(1'b0), .dy_wr_addr(8'h0), .dy_wr_data(32'h0),
        .bwd_start(1'b0),
        .dx_row(), .dx_valid(), .dx_row_idx(),
        .dWff1_flat(), .dWff2_flat()
    );

    fp32_matmul #(.M(T), .K(D), .N(V), .MUL_LATENCY(MUL_LAT)) u_proj (
        .clk(clk), .rst(rst), .en(en),
        .a_wr_en(proj_a_wr_en), .a_wr_addr(proj_a_wr_addr), .a_wr_data(proj_a_wr_data),
        .b_wr_en(proj_b_wr_en), .b_wr_addr(proj_b_wr_addr), .b_wr_data(proj_b_wr_data),
        .start(proj_start),
        .c_row(proj_c_row), .c_valid(proj_c_valid), .c_row_idx(proj_c_row_idx)
    );

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    localparam int TD1  = T*D - 1;    // 15
    localparam int W31  = 31;          // D*DH*4 - 1 = 31
    localparam int DV1  = D*V - 1;    // 63
    localparam int Tm1  = T - 1;       // 3

    typedef enum logic [4:0] {
        IDLE,
        WL_ATT,
        LN1_FEED, LN1_WAIT,
        ATT_XLOAD, ATT_START, ATT_WAIT,
        RESID1,
        WL_MLP,
        LN2_FEED, LN2_WAIT,
        MLP_XLOAD, MLP_START, MLP_WAIT,
        RESID2,
        PROJ_LOAD_A, PROJ_LOAD_B, PROJ_START, PROJ_WAIT,
        DONE_ST
    } state_t;

    state_t     state;
    logic [6:0] cnt;
    logic [1:0] ln_row;
    logic [1:0] layer_idx;

    // -----------------------------------------------------------------------
    // Combinatorial drive
    // -----------------------------------------------------------------------
    always_comb begin
        ln_x_in = '0; ln_gamma = '0; ln_beta = '0; ln_start = 1'b0;
        att_x_wr_en = 1'b0; att_x_wr_addr = '0; att_x_wr_data = '0;
        att_wq_wr_en = 1'b0; att_wq_wr_addr = '0; att_wq_wr_data = '0;
        att_wk_wr_en = 1'b0; att_wk_wr_addr = '0; att_wk_wr_data = '0;
        att_wv_wr_en = 1'b0; att_wv_wr_addr = '0; att_wv_wr_data = '0;
        att_wo_wr_en = 1'b0; att_wo_wr_addr = '0; att_wo_wr_data = '0;
        att_start = 1'b0;
        mlp_x_wr_en = 1'b0; mlp_x_wr_addr = '0; mlp_x_wr_data = '0;
        mlp_wff1_wr_en = 1'b0; mlp_wff1_wr_addr = '0; mlp_wff1_wr_data = '0;
        mlp_wff2_wr_en = 1'b0; mlp_wff2_wr_addr = '0; mlp_wff2_wr_data = '0;
        mlp_start = 1'b0;
        proj_a_wr_en = 1'b0; proj_a_wr_addr = '0; proj_a_wr_data = '0;
        proj_b_wr_en = 1'b0; proj_b_wr_addr = '0; proj_b_wr_data = '0;
        proj_start = 1'b0;

        case (state)
            WL_ATT: begin
                if      (cnt[5:4] == 2'b00) begin att_wq_wr_en = 1; att_wq_wr_addr = {4'b0,cnt[3:0]}; att_wq_wr_data = layer_Wq[layer_idx][cnt[3:2]][cnt[1:0]]; end
                else if (cnt[5:4] == 2'b01) begin att_wk_wr_en = 1; att_wk_wr_addr = {4'b0,cnt[3:0]}; att_wk_wr_data = layer_Wk[layer_idx][cnt[3:2]][cnt[1:0]]; end
                else if (cnt[5:4] == 2'b10) begin att_wv_wr_en = 1; att_wv_wr_addr = {4'b0,cnt[3:0]}; att_wv_wr_data = layer_Wv[layer_idx][cnt[3:2]][cnt[1:0]]; end
                else                         begin att_wo_wr_en = 1; att_wo_wr_addr = {4'b0,cnt[3:0]}; att_wo_wr_data = layer_Wo[layer_idx][cnt[3:2]][cnt[1:0]]; end
            end

            LN1_FEED: begin
                for (int d = 0; d < D; d++) begin
                    ln_x_in [d*32+:32] = x_reg[ln_row][d];
                    ln_gamma[d*32+:32] = layer_ln1_gamma[layer_idx][d];
                    ln_beta [d*32+:32] = layer_ln1_beta [layer_idx][d];
                end
                ln_start = 1'b1;
            end

            ATT_XLOAD: begin att_x_wr_en = 1; att_x_wr_addr = {4'b0,cnt[3:0]}; att_x_wr_data = xnorm1_buf[cnt[3:2]][cnt[1:0]]; end

            ATT_START: att_start = 1'b1;

            WL_MLP: begin
                if (!cnt[4]) begin mlp_wff1_wr_en = 1; mlp_wff1_wr_addr = {4'b0,cnt[3:0]}; mlp_wff1_wr_data = layer_Wff1[layer_idx][cnt[3:2]][cnt[1:0]]; end
                else         begin mlp_wff2_wr_en = 1; mlp_wff2_wr_addr = {4'b0,cnt[3:0]}; mlp_wff2_wr_data = layer_Wff2[layer_idx][cnt[3:2]][cnt[1:0]]; end
            end

            LN2_FEED: begin
                for (int d = 0; d < D; d++) begin
                    ln_x_in [d*32+:32] = x_reg[ln_row][d];
                    ln_gamma[d*32+:32] = layer_ln2_gamma[layer_idx][d];
                    ln_beta [d*32+:32] = layer_ln2_beta [layer_idx][d];
                end
                ln_start = 1'b1;
            end

            MLP_XLOAD: begin mlp_x_wr_en = 1; mlp_x_wr_addr = {4'b0,cnt[3:0]}; mlp_x_wr_data = xnorm2_buf[cnt[3:2]][cnt[1:0]]; end

            MLP_START: mlp_start = 1'b1;

            // Output projection: A = x_reg row-major, B = emb_reg transposed
            PROJ_LOAD_A: begin proj_a_wr_en = 1; proj_a_wr_addr = {4'b0,cnt[3:0]}; proj_a_wr_data = x_reg[cnt[3:2]][cnt[1:0]]; end
            PROJ_LOAD_B: begin
                // B[d][v] = emb_reg[v][d]; addr = d*V+v; cnt = d*V+v so d=cnt[5:4], v=cnt[3:0]
                proj_b_wr_en = 1; proj_b_wr_addr = {2'b0,cnt[5:0]}; proj_b_wr_data = emb_reg[cnt[3:0]][cnt[5:4]];
            end
            PROJ_START: proj_start = 1'b1;

            default: ;
        endcase
    end

    // -----------------------------------------------------------------------
    // FSM sequential
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE; cnt <= '0; ln_row <= '0; layer_idx <= '0;
            done <= 1'b0; out_valid <= 1'b0; out_row <= '0; out_row_idx <= '0;
        end else if (en) begin
            done <= 1'b0; out_valid <= 1'b0;

            case (state)
                IDLE: if (start) begin
                    for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) x_reg[t][d] <= x_init_reg[t][d];
                    layer_idx <= '0; cnt <= '0; ln_row <= '0; state <= WL_ATT;
                end

                WL_ATT:  begin cnt <= cnt+1; if (cnt==7'd63)         begin cnt<='0; state<=LN1_FEED; end end

                LN1_FEED: state <= LN1_WAIT;

                LN1_WAIT: if (ln_out_valid) begin
                    for (int d = 0; d < D; d++) xnorm1_buf[ln_row][d] <= ln_y_out[d*32+:32];
                    if (ln_row==T-1) begin ln_row<='0; cnt<='0; state<=ATT_XLOAD; end
                    else             begin ln_row<=ln_row+1;     state<=LN1_FEED;  end
                end

                ATT_XLOAD: begin cnt <= cnt+1; if (cnt==7'(TD1)) begin cnt<='0; state<=ATT_START; end end

                ATT_START: state <= ATT_WAIT;

                ATT_WAIT: if (att_out_valid) begin
                    for (int d = 0; d < D; d++) attn_buf[att_out_row_idx][d] <= att_out_row[d*32+:32];
                    if (att_out_row_idx == 2'(Tm1)) state <= RESID1;
                end

                RESID1: begin
                    for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) begin
                        shortreal xa, xb; xa=$bitstoshortreal(x_reg[t][d]); xb=$bitstoshortreal(attn_buf[t][d]);
                        x_reg[t][d] <= $shortrealtobits(xa+xb);
                    end
                    cnt<='0; state<=WL_MLP;
                end

                WL_MLP: begin cnt <= cnt+1; if (cnt==7'd31) begin cnt<='0; state<=LN2_FEED; end end

                LN2_FEED: state <= LN2_WAIT;

                LN2_WAIT: if (ln_out_valid) begin
                    for (int d = 0; d < D; d++) xnorm2_buf[ln_row][d] <= ln_y_out[d*32+:32];
                    if (ln_row==T-1) begin ln_row<='0; cnt<='0; state<=MLP_XLOAD; end
                    else             begin ln_row<=ln_row+1;     state<=LN2_FEED;  end
                end

                MLP_XLOAD: begin cnt <= cnt+1; if (cnt==7'(TD1)) begin cnt<='0; state<=MLP_START; end end

                MLP_START: state <= MLP_WAIT;

                MLP_WAIT: if (mlp_out_valid) begin
                    for (int d = 0; d < D; d++) mlp_buf[mlp_out_row_idx][d] <= mlp_out_row[d*32+:32];
                    if (mlp_out_row_idx == 2'(Tm1)) state <= RESID2;
                end

                RESID2: begin
                    for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) begin
                        shortreal xa, xb; xa=$bitstoshortreal(x_reg[t][d]); xb=$bitstoshortreal(mlp_buf[t][d]);
                        x_reg[t][d] <= $shortrealtobits(xa+xb);
                    end
                    if (layer_idx == L-1) begin cnt<='0; state<=PROJ_LOAD_A; end
                    else                  begin layer_idx<=layer_idx+1; cnt<='0; ln_row<='0; state<=WL_ATT; end
                end

                PROJ_LOAD_A: begin cnt <= cnt+1; if (cnt==7'(TD1))   begin cnt<='0; state<=PROJ_LOAD_B; end end
                PROJ_LOAD_B: begin cnt <= cnt+1; if (cnt==7'(DV1))   begin cnt<='0; state<=PROJ_START;  end end
                PROJ_START:  state <= PROJ_WAIT;

                PROJ_WAIT: begin
                    out_valid <= proj_c_valid; out_row <= proj_c_row; out_row_idx <= proj_c_row_idx;
                    if (proj_c_valid && proj_c_row_idx == 2'(Tm1)) state <= DONE_ST;
                end

                DONE_ST: begin done <= 1'b1; state <= IDLE; end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
