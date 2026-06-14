// MLP feed-forward block forward pass (BF16 weights, FP32 activations)
//
// Algorithm (matches AttentionLayer.Forward() FF sub-block in C#):
//   H1 = X  · Wff1     [T×D × D×FF → T×FF]   bf16w_matmul
//   G  = GeLU(H1)       [T×FF]                 gelu (element-wise, serialised per row)
//   Y  = G  · Wff2     [T×FF × FF×D → T×D]    bf16w_matmul
//
// Parameters:
//   T       — sequence length (default 4)
//   D       — embedding dim (default 4)
//   FF      — feed-forward hidden dim (default 16)
//   MAC_LAT — bf16w_matmul MAC pipeline depth (default 3)
//   EXP_LAT — gelu exp_lut pipeline depth (default 4)
//   LUT_SIZE — gelu exp LUT entries (default 256)
//   LUT_FILE — gelu exp LUT BRAM init hex
//
// Constraint: D and FF must both equal 4 for this prototype (register file sizes).
//             Phase 2 will generalise with parameterised reg files.
//
// Interface:
//   Weight load: wff1_wr_* [D×FF BF16], wff2_wr_* [FF×D BF16]
//   Input load:  x_wr_*    [T×D FP32]
//   start        — 1-cycle pulse to begin forward pass
//   out_row      — [D*32-1:0] one output row per clock when out_valid is high
//   out_valid    — T-cycle pulse (rows 0..T-1 in order)
//   out_row_idx  — 0-based output row index
//
// Sub-module instances (sequential reuse, Phase 1 prototype):
//   u_bwm   — bf16w_matmul  (Wff1 pass; Wff2 pass)
//   u_gelu  — gelu          (one element per clock, serialised over T×FF elements)
//
// FSM states:
//   IDLE → LD_W1 → RUN_W1 → WAIT_W1 → GELU_FEED → GELU_COLLECT →
//   LD_W2 → RUN_W2 → WAIT_W2 → OUTPUT → IDLE
//
// Latency (T=4, D=4, FF=4, MAC_LAT=3, EXP_LAT=4):
//   Wff1 matmul: ~38 cycles; GeLU T×FF elements: T*FF + EXP_LAT+2 cycles;
//   Wff2 matmul: ~38 cycles. Total ~120 cycles.

`timescale 1ns/1ps

module mlp_core #(
    parameter int    T        = 4,
    parameter int    D        = 4,
    parameter int    FF       = 4,
    parameter int    MAC_LAT  = 3,
    parameter int    EXP_LAT  = 4,
    parameter int    LUT_SIZE = 256,
    parameter string LUT_FILE = "exp_lut_init.hex"
)(
    input  logic clk,
    input  logic rst,
    input  logic en,

    // Load X [T×D] FP32 row-major
    input  logic        x_wr_en,
    input  logic [7:0]  x_wr_addr,
    input  logic [31:0] x_wr_data,

    // Load Wff1 [D×FF] BF16 row-major
    input  logic        wff1_wr_en,
    input  logic [7:0]  wff1_wr_addr,
    input  logic [15:0] wff1_wr_data,

    // Load Wff2 [FF×D] BF16 row-major
    input  logic        wff2_wr_en,
    input  logic [7:0]  wff2_wr_addr,
    input  logic [15:0] wff2_wr_data,

    input  logic start,

    output logic [D*32-1:0]      out_row,
    output logic                  out_valid,
    output logic [$clog2(T)-1:0] out_row_idx
);

    // -----------------------------------------------------------------------
    // Localparams
    // -----------------------------------------------------------------------
    localparam int TD  = T * D;
    localparam int DFF = D * FF;
    localparam int TFF = T * FF;
    localparam int FFD = FF * D;
    localparam int GELU_LAT = EXP_LAT + 2;

    // -----------------------------------------------------------------------
    // Weight and input register files
    // -----------------------------------------------------------------------
    logic [31:0] X_reg    [0:T-1][0:D-1];
    logic [15:0] Wff1_reg [0:D-1][0:FF-1];
    logic [15:0] Wff2_reg [0:FF-1][0:D-1];

    always_ff @(posedge clk) begin
        if (x_wr_en)    X_reg   [x_wr_addr    / D ][x_wr_addr    % D ] <= x_wr_data;
        if (wff1_wr_en) Wff1_reg[wff1_wr_addr / FF][wff1_wr_addr % FF] <= wff1_wr_data;
        if (wff2_wr_en) Wff2_reg[wff2_wr_addr / D ][wff2_wr_addr % D ] <= wff2_wr_data;
    end

    // -----------------------------------------------------------------------
    // Intermediate result registers
    // -----------------------------------------------------------------------
    logic [31:0] H1_reg [0:T-1][0:FF-1];   // pre-GeLU
    logic [31:0] G_reg  [0:T-1][0:FF-1];   // post-GeLU

    // -----------------------------------------------------------------------
    // bf16w_matmul instance — shared for Wff1 and Wff2 passes
    // -----------------------------------------------------------------------
    logic        bwm_a_wr_en;
    logic [7:0]  bwm_a_wr_addr;
    logic [31:0] bwm_a_wr_data;
    logic        bwm_b_wr_en;
    logic [7:0]  bwm_b_wr_addr;
    logic [15:0] bwm_b_wr_data;
    logic        bwm_start;
    logic [FF*32-1:0] bwm_c_row_ff;   // wide enough for FF cols (Wff1 pass)
    logic [D*32-1:0]  bwm_c_row_d;    // wide enough for D cols  (Wff2 pass)
    logic        bwm_c_valid;
    logic [1:0]  bwm_c_row_idx;

    // The matmul N parameter differs between passes (FF for Wff1, D for Wff2).
    // We use FF for the instantiation (FF=D=4 in this prototype).
    // Output wire is cast appropriately per FSM state.
    logic [FF*32-1:0] bwm_c_row_raw;

    bf16w_matmul #(
        .M          (T),
        .K          (D),    // K=D for Wff1 pass; K=FF=D for Wff2 pass (D==FF in prototype)
        .N          (FF),
        .MAC_LATENCY(MAC_LAT)
    ) u_bwm (
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
        .c_row      (bwm_c_row_raw),
        .c_valid    (bwm_c_valid),
        .c_row_idx  (bwm_c_row_idx)
    );

    // -----------------------------------------------------------------------
    // gelu instance — T*FF elements processed one per clock
    // -----------------------------------------------------------------------
    logic        gelu_valid_in;
    logic [31:0] gelu_x_fp32;
    logic [31:0] gelu_result;
    logic        gelu_valid_out;

    gelu #(
        .EXP_LAT (EXP_LAT),
        .LUT_SIZE(LUT_SIZE),
        .LUT_FILE(LUT_FILE)
    ) u_gelu (
        .clk        (clk),
        .rst        (rst),
        .en         (en),
        .x_fp32     (gelu_x_fp32),
        .valid_in   (gelu_valid_in),
        .result_fp32(gelu_result),
        .valid_out  (gelu_valid_out)
    );

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    typedef enum logic [3:0] {
        IDLE,
        LD_W1,       // load X into bwm A, Wff1 into bwm B
        RUN_W1,      // pulse bwm_start
        WAIT_W1,     // collect T rows of H1
        GELU_FEED,   // feed T*FF elements to gelu serially
        GELU_COLLECT,// collect T*FF GeLU results
        LD_W2,       // load G into bwm A, Wff2 into bwm B
        RUN_W2,      // pulse bwm_start
        WAIT_W2,     // collect T rows of Y → output
        OUTPUT,      // emit T output rows
        DONE
    } state_t;

    state_t state;

    int load_idx;   // general-purpose load counter
    int h1_rows;    // H1 rows collected so far
    int gelu_in_cnt;
    int gelu_out_cnt;
    int out_cnt;
    int wait_cyc;

    localparam int TOTAL_LAT_W = MAC_LAT + 2;  // matmul total latency after last row fed

    // -----------------------------------------------------------------------
    // FSM transitions and datapath
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            state        <= IDLE;
            bwm_a_wr_en  <= 0;
            bwm_b_wr_en  <= 0;
            bwm_start    <= 0;
            gelu_valid_in <= 0;
            out_valid    <= 0;
        end else if (en) begin
            // Defaults
            bwm_a_wr_en  <= 0;
            bwm_b_wr_en  <= 0;
            bwm_start    <= 0;
            gelu_valid_in <= 0;
            out_valid    <= 0;

            case (state)

                // ── IDLE ──────────────────────────────────────────────────
                IDLE: begin
                    if (start) begin
                        load_idx <= 0;
                        state    <= LD_W1;
                    end
                end

                // ── LD_W1: stream X→A and Wff1→B into matmul reg files ───
                LD_W1: begin
                    if (load_idx < TD) begin
                        bwm_a_wr_en   <= 1;
                        bwm_a_wr_addr <= load_idx[7:0];
                        bwm_a_wr_data <= X_reg[load_idx / D][load_idx % D];
                        bwm_b_wr_en   <= 1;
                        bwm_b_wr_addr <= load_idx[7:0];
                        bwm_b_wr_data <= Wff1_reg[load_idx / FF][load_idx % FF];
                        load_idx <= load_idx + 1;
                    end else begin
                        h1_rows  <= 0;
                        state    <= RUN_W1;
                    end
                end

                // ── RUN_W1 ────────────────────────────────────────────────
                RUN_W1: begin
                    bwm_start <= 1;
                    state     <= WAIT_W1;
                end

                // ── WAIT_W1: collect T rows into H1_reg ───────────────────
                WAIT_W1: begin
                    if (bwm_c_valid) begin
                        for (int j = 0; j < FF; j++)
                            H1_reg[bwm_c_row_idx][j] <= bwm_c_row_raw[j*32 +: 32];
                        h1_rows <= h1_rows + 1;
                        if (h1_rows == T - 1) begin
                            gelu_in_cnt  <= 0;
                            gelu_out_cnt <= 0;
                            state        <= GELU_FEED;
                        end
                    end
                end

                // ── GELU_FEED: feed T*FF elements to gelu one per clock ───
                GELU_FEED: begin
                    if (gelu_in_cnt < TFF) begin
                        gelu_valid_in <= 1;
                        gelu_x_fp32   <= H1_reg[gelu_in_cnt / FF][gelu_in_cnt % FF];
                        gelu_in_cnt   <= gelu_in_cnt + 1;
                    end
                    if (gelu_valid_out) begin
                        G_reg[gelu_out_cnt / FF][gelu_out_cnt % FF] <= gelu_result;
                        gelu_out_cnt <= gelu_out_cnt + 1;
                    end
                    // Transition once all inputs sent; keep collecting in GELU_COLLECT
                    if (gelu_in_cnt == TFF)
                        state <= GELU_COLLECT;
                end

                // ── GELU_COLLECT: drain remaining gelu outputs ─────────────
                GELU_COLLECT: begin
                    if (gelu_valid_out) begin
                        G_reg[gelu_out_cnt / FF][gelu_out_cnt % FF] <= gelu_result;
                        gelu_out_cnt <= gelu_out_cnt + 1;
                    end
                    if (gelu_out_cnt == TFF) begin
                        load_idx <= 0;
                        state    <= LD_W2;
                    end
                end

                // ── LD_W2: stream G→A and Wff2→B into matmul reg files ───
                LD_W2: begin
                    if (load_idx < TFF) begin
                        bwm_a_wr_en   <= 1;
                        bwm_a_wr_addr <= load_idx[7:0];
                        bwm_a_wr_data <= G_reg[load_idx / FF][load_idx % FF];
                        bwm_b_wr_en   <= 1;
                        bwm_b_wr_addr <= load_idx[7:0];
                        bwm_b_wr_data <= Wff2_reg[load_idx / D][load_idx % D];
                        load_idx <= load_idx + 1;
                    end else begin
                        out_cnt <= 0;
                        state   <= RUN_W2;
                    end
                end

                // ── RUN_W2 ────────────────────────────────────────────────
                RUN_W2: begin
                    bwm_start <= 1;
                    state     <= WAIT_W2;
                end

                // ── WAIT_W2: collect T rows → drive output directly ───────
                WAIT_W2: begin
                    if (bwm_c_valid) begin
                        out_row     <= bwm_c_row_raw[D*32-1:0];
                        out_row_idx <= bwm_c_row_idx[$clog2(T)-1:0];
                        out_valid   <= 1;
                        out_cnt     <= out_cnt + 1;
                        if (out_cnt == T - 1)
                            state <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
