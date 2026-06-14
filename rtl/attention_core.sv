// Scaled dot-product attention forward pass (single head, BF16 weights)
//
// Algorithm (matches AttentionCore.Forward() in C#):
//   Q = X · Wq          [T×D × D×DH → T×DH]   bf16w_matmul
//   K = X · Wk          [T×D × D×DH → T×DH]   bf16w_matmul
//   V = X · Wv          [T×D × D×DH → T×DH]   bf16w_matmul
//   S = Q · Kᵀ / √DH   [T×DH × DH×T → T×T]   fp32_matmul + scale
//   A = softmax(S)       [T×T]                  softmax (causal mask per row)
//   Z = A · V            [T×T × T×DH → T×DH]   fp32_matmul
//   Y = Z · Wo           [T×DH × DH×D → T×D]   bf16w_matmul
//
// Constraint: T = D = DH = 4 required for register-file reuse across pipeline stages.
//             (The single bf16w_matmul and fp32_matmul instances are reused for all
//              respective matmul operations; this works because all shapes are 4×4×4.)
//
// Interface:
//   Weight load:  wq_wr_*, wk_wr_*, wv_wr_*, wo_wr_*  — BF16, row-major
//   Input load:   x_wr_*                               — FP32, row-major
//   start         — 1-cycle pulse to begin forward pass
//   out_row       — [D*32-1:0] one row per clock when out_valid is high
//   out_valid     — T-cycle pulse (one row per clock, rows 0..T-1 in order)
//   out_row_idx   — 0-based output row index
//
// Sub-module instances (each reused across multiple operations):
//   u_bwm   — bf16w_matmul  (Q, K, V projections; output projection Y)
//   u_fpm   — fp32_matmul   (score matrix S; weighted sum Z)
//   u_smx   — softmax       (all T rows, processed sequentially)
//
// Latency (T=4, D=4, DH=4, MAC_LAT=3, MUL_LAT=3, EXP_LAT=4):
//   7 × matmul phases × ~38 cycles + 4 softmax rows × ~14 cycles ≈ 320 cycles
//   Exact latency is non-critical for correctness.
//
// Parameters:
//   T, D, DH      — sequence length, embed dim, head dim (must all equal 4)
//   MAC_LAT       — bf16w_matmul MAC pipeline depth (default 3)
//   MUL_LAT       — fp32_matmul multiply pipeline depth (default 3)
//   EXP_LAT       — softmax exp_lut pipeline depth (default 4)
//   LUT_SIZE      — softmax exp LUT entries (default 256)
//   LUT_FILE      — softmax exp LUT BRAM init hex (relative to xsim work dir)

`timescale 1ns/1ps

module attention_core #(
    parameter int    T        = 4,
    parameter int    D        = 4,
    parameter int    DH       = 4,
    parameter int    MAC_LAT  = 3,
    parameter int    MUL_LAT  = 3,
    parameter int    EXP_LAT  = 4,
    parameter int    LUT_SIZE = 256,
    parameter string LUT_FILE = "exp_lut_init.hex"
)(
    input  logic clk,
    input  logic rst,
    input  logic en,

    // Load X [T×D] FP32 row-major (addr = row*D + col)
    input  logic        x_wr_en,
    input  logic [7:0]  x_wr_addr,
    input  logic [31:0] x_wr_data,

    // Load Wq, Wk, Wv [D×DH] BF16 row-major (addr = row*DH + col)
    input  logic        wq_wr_en,
    input  logic [7:0]  wq_wr_addr,
    input  logic [15:0] wq_wr_data,
    input  logic        wk_wr_en,
    input  logic [7:0]  wk_wr_addr,
    input  logic [15:0] wk_wr_data,
    input  logic        wv_wr_en,
    input  logic [7:0]  wv_wr_addr,
    input  logic [15:0] wv_wr_data,

    // Load Wo [DH×D] BF16 row-major (addr = row*D + col)
    input  logic        wo_wr_en,
    input  logic [7:0]  wo_wr_addr,
    input  logic [15:0] wo_wr_data,

    input  logic start,

    // Output: T rows × D FP32 cols, one row per clock
    output logic [D*32-1:0]      out_row,
    output logic                  out_valid,
    output logic [$clog2(T)-1:0] out_row_idx
);

    // -----------------------------------------------------------------------
    // Localparams
    // -----------------------------------------------------------------------
    localparam int TD   = T * D;    // 16
    localparam int DDH  = D * DH;   // 16  (D rows × DH cols for Wq/Wk/Wv)
    localparam int TDH  = T * DH;   // 16  (T rows × DH cols for Q/K/V/attnOut)
    localparam int TT   = T * T;    // 16  (T rows × T cols for scores/attn)
    localparam int DHD  = DH * D;   // 16  (DH rows × D cols for Wo/output)

    // Scale 1/√DH hardcoded for DH=4; used in SC_WAIT to normalise Q·Kᵀ
    localparam shortreal SCALE_SR = 1.0 / 2.0;   // 1/√4 = 0.5

    // -----------------------------------------------------------------------
    // Weight and input register files (loaded from outside before start)
    // -----------------------------------------------------------------------
    logic [31:0] X_reg  [0:T-1][0:D-1];
    logic [15:0] Wq_reg [0:D-1][0:DH-1];
    logic [15:0] Wk_reg [0:D-1][0:DH-1];
    logic [15:0] Wv_reg [0:D-1][0:DH-1];
    logic [15:0] Wo_reg [0:DH-1][0:D-1];

    always_ff @(posedge clk) begin
        if (x_wr_en)  X_reg [x_wr_addr  / D ][x_wr_addr  % D ] <= x_wr_data;
        if (wq_wr_en) Wq_reg[wq_wr_addr / DH][wq_wr_addr % DH] <= wq_wr_data;
        if (wk_wr_en) Wk_reg[wk_wr_addr / DH][wk_wr_addr % DH] <= wk_wr_data;
        if (wv_wr_en) Wv_reg[wv_wr_addr / DH][wv_wr_addr % DH] <= wv_wr_data;
        if (wo_wr_en) Wo_reg[wo_wr_addr / D ][wo_wr_addr % D ] <= wo_wr_data;
    end

    // -----------------------------------------------------------------------
    // Intermediate result registers
    // -----------------------------------------------------------------------
    logic [31:0] Q_reg       [0:T-1][0:DH-1];
    logic [31:0] K_reg       [0:T-1][0:DH-1];
    logic [31:0] V_reg       [0:T-1][0:DH-1];
    logic [31:0] scores_reg  [0:T-1][0:T-1];
    logic [31:0] attn_reg    [0:T-1][0:T-1];
    logic [31:0] attnOut_reg [0:T-1][0:DH-1];

    // -----------------------------------------------------------------------
    // FSM states
    // -----------------------------------------------------------------------
    typedef enum logic [4:0] {
        IDLE,
        Q_LD_A,  Q_LD_B,  Q_RUN,  Q_WAIT,
        K_LD_A,  K_LD_B,  K_RUN,  K_WAIT,
        V_LD_A,  V_LD_B,  V_RUN,  V_WAIT,
        SC_LD_A, SC_LD_B, SC_RUN, SC_WAIT,
        SM_FEED, SM_WAIT,
        AV_LD_A, AV_LD_B, AV_RUN, AV_WAIT,
        OUT_LD_A,OUT_LD_B,OUT_RUN,OUT_WAIT
    } state_t;

    state_t state;
    int     cnt;
    int     sm_row;

    // -----------------------------------------------------------------------
    // bf16w_matmul sub-module signals (reused for Q, K, V, out projections)
    // DIM: M=T=4, K=D=4, N=DH=4 — all register files are 4×4 = 16 entries
    // -----------------------------------------------------------------------
    logic        bwm_a_wr_en;
    logic [7:0]  bwm_a_wr_addr;
    logic [31:0] bwm_a_wr_data;
    logic        bwm_b_wr_en;
    logic [7:0]  bwm_b_wr_addr;
    logic [15:0] bwm_b_wr_data;
    logic        bwm_start;
    logic [DH*32-1:0] bwm_c_row;    // DH = D = 4, so [127:0]
    logic              bwm_c_valid;
    logic [1:0]        bwm_c_row_idx;

    bf16w_matmul #(.M(T), .K(D), .N(DH), .MAC_LATENCY(MAC_LAT)) u_bwm (
        .clk        (clk),
        .rst        (rst),
        .en         (en),
        .a_wr_en    (bwm_a_wr_en),
        .a_wr_addr  (bwm_a_wr_addr),
        .a_wr_data  (bwm_a_wr_data),
        .b_wr_en    (bwm_b_wr_en),
        .b_wr_addr  (bwm_b_wr_addr),
        .b_wr_data  (bwm_b_wr_data),
        .start      (bwm_start),
        .c_row      (bwm_c_row),
        .c_valid    (bwm_c_valid),
        .c_row_idx  (bwm_c_row_idx)
    );

    // -----------------------------------------------------------------------
    // fp32_matmul sub-module signals (reused for scores S and attnOut Z)
    // DIM: M=T=4, K=4, N=4 — all outputs are [T*32-1:0] = [127:0]
    // -----------------------------------------------------------------------
    logic        fpm_a_wr_en;
    logic [7:0]  fpm_a_wr_addr;
    logic [31:0] fpm_a_wr_data;
    logic        fpm_b_wr_en;
    logic [7:0]  fpm_b_wr_addr;
    logic [31:0] fpm_b_wr_data;
    logic        fpm_start;
    logic [T*32-1:0] fpm_c_row;
    logic             fpm_c_valid;
    logic [1:0]       fpm_c_row_idx;

    fp32_matmul #(.M(T), .K(DH), .N(T), .MUL_LATENCY(MUL_LAT)) u_fpm (
        .clk        (clk),
        .rst        (rst),
        .en         (en),
        .a_wr_en    (fpm_a_wr_en),
        .a_wr_addr  (fpm_a_wr_addr),
        .a_wr_data  (fpm_a_wr_data),
        .b_wr_en    (fpm_b_wr_en),
        .b_wr_addr  (fpm_b_wr_addr),
        .b_wr_data  (fpm_b_wr_data),
        .start      (fpm_start),
        .c_row      (fpm_c_row),
        .c_valid    (fpm_c_valid),
        .c_row_idx  (fpm_c_row_idx)
    );

    // -----------------------------------------------------------------------
    // softmax sub-module signals
    // -----------------------------------------------------------------------
    logic [T*32-1:0]      sm_scores_in;
    logic [$clog2(T)-1:0] sm_row_in;
    logic                  sm_start;
    logic [T*32-1:0]      sm_probs_out;
    logic                  sm_valid_out;

    softmax #(
        .T       (T),
        .EXP_LAT (EXP_LAT),
        .LUT_SIZE(LUT_SIZE),
        .LUT_FILE(LUT_FILE)
    ) u_smx (
        .clk       (clk),
        .rst       (rst),
        .en        (en),
        .scores_in (sm_scores_in),
        .row_idx   (sm_row_in),
        .start     (sm_start),
        .probs_out (sm_probs_out),
        .valid_out (sm_valid_out)
    );

    // -----------------------------------------------------------------------
    // Combinatorial sub-module input mux (driven by state + cnt + registers)
    // -----------------------------------------------------------------------
    always_comb begin
        // Defaults — all sub-module inputs idle
        bwm_a_wr_en   = 1'b0;
        bwm_a_wr_addr = 8'h0;
        bwm_a_wr_data = 32'h0;
        bwm_b_wr_en   = 1'b0;
        bwm_b_wr_addr = 8'h0;
        bwm_b_wr_data = 16'h0;
        bwm_start     = 1'b0;
        fpm_a_wr_en   = 1'b0;
        fpm_a_wr_addr = 8'h0;
        fpm_a_wr_data = 32'h0;
        fpm_b_wr_en   = 1'b0;
        fpm_b_wr_addr = 8'h0;
        fpm_b_wr_data = 32'h0;
        fpm_start     = 1'b0;
        sm_scores_in  = '0;
        sm_row_in     = '0;
        sm_start      = 1'b0;

        unique case (state)
            // ---- Q = X · Wq ------------------------------------------------
            Q_LD_A: begin
                bwm_a_wr_en   = 1'b1;
                bwm_a_wr_addr = cnt[7:0];
                bwm_a_wr_data = X_reg[cnt / D][cnt % D];
            end
            Q_LD_B: begin
                bwm_b_wr_en   = 1'b1;
                bwm_b_wr_addr = cnt[7:0];
                bwm_b_wr_data = Wq_reg[cnt / DH][cnt % DH];
            end
            Q_RUN:  bwm_start = 1'b1;

            // ---- K = X · Wk ------------------------------------------------
            K_LD_A: begin
                bwm_a_wr_en   = 1'b1;
                bwm_a_wr_addr = cnt[7:0];
                bwm_a_wr_data = X_reg[cnt / D][cnt % D];
            end
            K_LD_B: begin
                bwm_b_wr_en   = 1'b1;
                bwm_b_wr_addr = cnt[7:0];
                bwm_b_wr_data = Wk_reg[cnt / DH][cnt % DH];
            end
            K_RUN:  bwm_start = 1'b1;

            // ---- V = X · Wv ------------------------------------------------
            V_LD_A: begin
                bwm_a_wr_en   = 1'b1;
                bwm_a_wr_addr = cnt[7:0];
                bwm_a_wr_data = X_reg[cnt / D][cnt % D];
            end
            V_LD_B: begin
                bwm_b_wr_en   = 1'b1;
                bwm_b_wr_addr = cnt[7:0];
                bwm_b_wr_data = Wv_reg[cnt / DH][cnt % DH];
            end
            V_RUN:  bwm_start = 1'b1;

            // ---- S = Q · Kᵀ / √DH ----------------------------------------
            // A = Q [T×DH], addr = t*DH + dh (natural row-major)
            SC_LD_A: begin
                fpm_a_wr_en   = 1'b1;
                fpm_a_wr_addr = cnt[7:0];
                fpm_a_wr_data = Q_reg[cnt / DH][cnt % DH];
            end
            // B = Kᵀ [DH×T]; addr = dh*T + t → Kᵀ[dh][t] = K[t][dh]
            // cnt = k*T + j  where k = cnt/T, j = cnt%T → K[j][k]
            SC_LD_B: begin
                fpm_b_wr_en   = 1'b1;
                fpm_b_wr_addr = cnt[7:0];
                fpm_b_wr_data = K_reg[cnt % T][cnt / T];
            end
            SC_RUN:  fpm_start = 1'b1;

            // ---- softmax row sm_row ----------------------------------------
            SM_FEED: begin
                sm_start = 1'b1;
                sm_row_in = sm_row[$clog2(T)-1:0];
                for (int j = 0; j < T; j++)
                    sm_scores_in[j*32 +: 32] = scores_reg[sm_row][j];
            end

            // ---- Z = A · V ------------------------------------------------
            // A = attn [T×T], addr = t0*T + t1
            AV_LD_A: begin
                fpm_a_wr_en   = 1'b1;
                fpm_a_wr_addr = cnt[7:0];
                fpm_a_wr_data = attn_reg[cnt / T][cnt % T];
            end
            // B = V [T×DH], addr = t*DH + dh (natural row-major)
            AV_LD_B: begin
                fpm_b_wr_en   = 1'b1;
                fpm_b_wr_addr = cnt[7:0];
                fpm_b_wr_data = V_reg[cnt / DH][cnt % DH];
            end
            AV_RUN:  fpm_start = 1'b1;

            // ---- Y = Z · Wo -----------------------------------------------
            // A = attnOut [T×DH], addr = t*DH + dh
            OUT_LD_A: begin
                bwm_a_wr_en   = 1'b1;
                bwm_a_wr_addr = cnt[7:0];
                bwm_a_wr_data = attnOut_reg[cnt / DH][cnt % DH];
            end
            // B = Wo [DH×D], addr = dh*D + d
            OUT_LD_B: begin
                bwm_b_wr_en   = 1'b1;
                bwm_b_wr_addr = cnt[7:0];
                bwm_b_wr_data = Wo_reg[cnt / D][cnt % D];
            end
            OUT_RUN: bwm_start = 1'b1;

            default: ;
        endcase
    end

    // -----------------------------------------------------------------------
    // Output registers (set during OUT_WAIT from bwm outputs)
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            out_valid   <= 1'b0;
            out_row     <= '0;
            out_row_idx <= '0;
        end else if (en) begin
            out_valid <= 1'b0;  // default: not valid this cycle
            if (state == OUT_WAIT && bwm_c_valid) begin
                out_row     <= bwm_c_row;
                out_valid   <= 1'b1;
                out_row_idx <= bwm_c_row_idx[$clog2(T)-1:0];
            end
        end
    end

    // -----------------------------------------------------------------------
    // Main FSM
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state  <= IDLE;
            cnt    <= 0;
            sm_row <= 0;
        end else if (en) begin
            case (state)

                // ---- IDLE --------------------------------------------------
                IDLE: begin
                    if (start) begin
                        state <= Q_LD_A;
                        cnt   <= 0;
                    end
                end

                // ---- Q = X · Wq --------------------------------------------
                Q_LD_A: begin
                    cnt <= cnt + 1;
                    if (cnt == TD - 1) begin state <= Q_LD_B; cnt <= 0; end
                end
                Q_LD_B: begin
                    cnt <= cnt + 1;
                    if (cnt == DDH - 1) begin state <= Q_RUN; cnt <= 0; end
                end
                Q_RUN: state <= Q_WAIT;
                Q_WAIT: begin
                    if (bwm_c_valid) begin
                        for (int j = 0; j < DH; j++)
                            Q_reg[int'(bwm_c_row_idx)][j] <= bwm_c_row[j*32 +: 32];
                        if (int'(bwm_c_row_idx) == T - 1) begin
                            state <= K_LD_A;
                            cnt   <= 0;
                        end
                    end
                end

                // ---- K = X · Wk --------------------------------------------
                K_LD_A: begin
                    cnt <= cnt + 1;
                    if (cnt == TD - 1) begin state <= K_LD_B; cnt <= 0; end
                end
                K_LD_B: begin
                    cnt <= cnt + 1;
                    if (cnt == DDH - 1) begin state <= K_RUN; cnt <= 0; end
                end
                K_RUN: state <= K_WAIT;
                K_WAIT: begin
                    if (bwm_c_valid) begin
                        for (int j = 0; j < DH; j++)
                            K_reg[int'(bwm_c_row_idx)][j] <= bwm_c_row[j*32 +: 32];
                        if (int'(bwm_c_row_idx) == T - 1) begin
                            state <= V_LD_A;
                            cnt   <= 0;
                        end
                    end
                end

                // ---- V = X · Wv --------------------------------------------
                V_LD_A: begin
                    cnt <= cnt + 1;
                    if (cnt == TD - 1) begin state <= V_LD_B; cnt <= 0; end
                end
                V_LD_B: begin
                    cnt <= cnt + 1;
                    if (cnt == DDH - 1) begin state <= V_RUN; cnt <= 0; end
                end
                V_RUN: state <= V_WAIT;
                V_WAIT: begin
                    if (bwm_c_valid) begin
                        for (int j = 0; j < DH; j++)
                            V_reg[int'(bwm_c_row_idx)][j] <= bwm_c_row[j*32 +: 32];
                        if (int'(bwm_c_row_idx) == T - 1) begin
                            state <= SC_LD_A;
                            cnt   <= 0;
                        end
                    end
                end

                // ---- S = Q · Kᵀ / √DH ------------------------------------
                // SC_LD_B loops over DH*T = T*DH = TDH entries (DH=T=4 → same count)
                SC_LD_A: begin
                    cnt <= cnt + 1;
                    if (cnt == TDH - 1) begin state <= SC_LD_B; cnt <= 0; end
                end
                SC_LD_B: begin
                    cnt <= cnt + 1;
                    if (cnt == TDH - 1) begin state <= SC_RUN; cnt <= 0; end
                end
                SC_RUN: state <= SC_WAIT;
                SC_WAIT: begin
                    if (fpm_c_valid) begin
                        for (int j = 0; j < T; j++)
                            scores_reg[int'(fpm_c_row_idx)][j] <=
                                $shortrealtobits($bitstoshortreal(fpm_c_row[j*32 +: 32]) * SCALE_SR);
                        if (int'(fpm_c_row_idx) == T - 1) begin
                            state  <= SM_FEED;
                            sm_row <= 0;
                        end
                    end
                end

                // ---- A = softmax(S) ----------------------------------------
                // SM_FEED: combinatorial block drives sm_start=1 and sm_scores_in
                SM_FEED: state <= SM_WAIT;
                SM_WAIT: begin
                    if (sm_valid_out) begin
                        for (int j = 0; j < T; j++)
                            attn_reg[sm_row][j] <= sm_probs_out[j*32 +: 32];
                        if (sm_row == T - 1) begin
                            state <= AV_LD_A;
                            cnt   <= 0;
                        end else begin
                            sm_row <= sm_row + 1;
                            state  <= SM_FEED;
                        end
                    end
                end

                // ---- Z = A · V --------------------------------------------
                AV_LD_A: begin
                    cnt <= cnt + 1;
                    if (cnt == TT - 1) begin state <= AV_LD_B; cnt <= 0; end
                end
                AV_LD_B: begin
                    cnt <= cnt + 1;
                    if (cnt == TDH - 1) begin state <= AV_RUN; cnt <= 0; end
                end
                AV_RUN: state <= AV_WAIT;
                AV_WAIT: begin
                    if (fpm_c_valid) begin
                        for (int j = 0; j < DH; j++)
                            attnOut_reg[int'(fpm_c_row_idx)][j] <= fpm_c_row[j*32 +: 32];
                        if (int'(fpm_c_row_idx) == T - 1) begin
                            state <= OUT_LD_A;
                            cnt   <= 0;
                        end
                    end
                end

                // ---- Y = Z · Wo -------------------------------------------
                OUT_LD_A: begin
                    cnt <= cnt + 1;
                    if (cnt == TDH - 1) begin state <= OUT_LD_B; cnt <= 0; end
                end
                OUT_LD_B: begin
                    cnt <= cnt + 1;
                    if (cnt == DHD - 1) begin state <= OUT_RUN; cnt <= 0; end
                end
                OUT_RUN: state <= OUT_WAIT;
                OUT_WAIT: begin
                    if (bwm_c_valid && int'(bwm_c_row_idx) == T - 1)
                        state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
