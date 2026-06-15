// MLP feed-forward block forward + backward pass (BF16 weights, FP32 activations)
//
// Algorithm (matches AttentionLayer.Forward() FF sub-block in C#):
//   H1 = X  · Wff1     [T×D × D×FF → T×FF]   bf16w_matmul
//   G  = GeLU(H1)       [T×FF]                 gelu (element-wise, serialised per row)
//   Y  = G  · Wff2     [T×FF × FF×D → T×D]    bf16w_matmul
//
// Backward pass (matches AttentionLayer.Backward() FF sub-block in C#):
//   dWff2 = Gᵀ · dY    [FF×T × T×D → FF×D]   fp32_matmul (A transposed)
//   dG    = dY · Wff2ᵀ [T×D × D×FF → T×FF]   fp32_matmul (B transposed; FP32 weights decoded)
//   dH1   = dG ⊙ GeLU'(H1)                    element-wise shortreal (no new primitives)
//   dWff1 = Xᵀ · dH1   [D×T × T×FF → D×FF]   fp32_matmul (A transposed)
//   dX    = dH1 · Wff1ᵀ [T×FF × FF×D → T×D]  fp32_matmul (B transposed; FP32 weights decoded)
//
// GeLU derivative (matches AttentionLayer.GeluBackward, c = 0.7978845608, c2 = 0.1070322244):
//   d(GeLU)/dx = 0.5*(1+tanh(inner)) + 0.5*x*sech²(inner)*(c + c2*x²)
//   where inner = c*(x + 0.044715*x³),  sech²(inner) = 1 - tanh²(inner)
//   tanh is computed via shortreal exp: tanh(z) = (e^2z - 1)/(e^2z + 1)
//
// Parameters:
//   T       — sequence length (default 4)
//   D       — embedding dim (default 4)
//   FF      — feed-forward hidden dim (default 4)
//   MAC_LAT — bf16w_matmul MAC pipeline depth (default 3)
//   EXP_LAT — gelu exp_lut pipeline depth (default 4)
//   LUT_SIZE — gelu exp LUT entries (default 256)
//   LUT_FILE — gelu exp LUT BRAM init hex
//
// Constraint: D and FF must both equal 4 for this prototype (register file sizes).
//             Phase 2 will generalise with parameterised reg files.
//
// Interface:
//   Forward:   x_wr_*, wff1_wr_*, wff2_wr_*, start → out_row, out_valid, out_row_idx
//   Backward:  bwd_start, dy_wr_* → dx_row, dx_valid, dx_row_idx, dWff1, dWff2 (flat)
//
// Sub-module instances:
//   u_bwm   — bf16w_matmul  (Wff1/Wff2 forward passes)
//   u_fpm   — fp32_matmul   (all 5 backward gradient matmuls; M=K=N=4 always)
//   u_gelu  — gelu          (forward GeLU)
//
// FSM forward states:
//   IDLE → LD_W1 → RUN_W1 → WAIT_W1 → GELU_FEED → GELU_COLLECT → LD_W2 → RUN_W2 → WAIT_W2 → IDLE
// FSM backward states (entered on bwd_start after forward completes):
//   BWD_LD_DW2 → BWD_RUN_DW2 → BWD_WAIT_DW2 →
//   BWD_LD_DG  → BWD_RUN_DG  → BWD_WAIT_DG  →
//   BWD_GELU_GRAD →
//   BWD_LD_DW1 → BWD_RUN_DW1 → BWD_WAIT_DW1 →
//   BWD_LD_DX  → BWD_RUN_DX  → BWD_WAIT_DX  → IDLE

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
    output logic [$clog2(T)-1:0] out_row_idx,

    // ── Backward pass ────────────────────────────────────────────────────
    // Load dY [T×D] FP32 row-major (upstream gradient from next layer/loss)
    input  logic        dy_wr_en,
    input  logic [7:0]  dy_wr_addr,
    input  logic [31:0] dy_wr_data,

    input  logic bwd_start,    // 1-cycle pulse: begin backward pass

    // dX output — T rows streamed row-by-row (same protocol as out_row)
    output logic [D*32-1:0]      dx_row,
    output logic                  dx_valid,
    output logic [$clog2(T)-1:0] dx_row_idx,

    // Gradient accumulator outputs (flat, whole matrix, valid one cycle after dx_valid T-th row)
    output logic [D*FF*32-1:0]  dWff1_flat,   // [D×FF] FP32 row-major
    output logic [FF*D*32-1:0]  dWff2_flat    // [FF×D] FP32 row-major
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
    // Backward registers
    // -----------------------------------------------------------------------
    logic [31:0] dY_reg   [0:T-1][0:D-1];   // upstream gradient
    logic [31:0] dG_reg   [0:T-1][0:FF-1];  // gradient after Wff2ᵀ
    logic [31:0] dH1_reg  [0:T-1][0:FF-1];  // gradient after GeLU backward
    logic [31:0] dWff1_reg[0:D-1][0:FF-1];  // weight gradient for Wff1
    logic [31:0] dWff2_reg[0:FF-1][0:D-1];  // weight gradient for Wff2

    always_ff @(posedge clk)
        if (dy_wr_en)
            dY_reg[dy_wr_addr / D][dy_wr_addr % D] <= dy_wr_data;

    // Pack gradient registers to flat outputs (combinational)
    always_comb begin
        for (int i = 0; i < D; i++)
            for (int j = 0; j < FF; j++)
                dWff1_flat[(i*FF+j)*32 +: 32] = dWff1_reg[i][j];
        for (int i = 0; i < FF; i++)
            for (int j = 0; j < D; j++)
                dWff2_flat[(i*D+j)*32 +: 32] = dWff2_reg[i][j];
    end

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
    // fp32_matmul instance — used for all 5 backward gradient matmuls
    // Instantiated with M=T, K=T, N=FF (4×4×4 covers all backward shapes)
    // For each backward matmul the load loops treat A/B as the right shapes.
    // -----------------------------------------------------------------------
    logic        fpm_a_wr_en;
    logic [7:0]  fpm_a_wr_addr;
    logic [31:0] fpm_a_wr_data;
    logic        fpm_b_wr_en;
    logic [7:0]  fpm_b_wr_addr;
    logic [31:0] fpm_b_wr_data;
    logic        fpm_start;
    logic [FF*32-1:0] fpm_c_row;   // FF==D in prototype
    logic        fpm_c_valid;
    logic [1:0]  fpm_c_row_idx;

    fp32_matmul #(
        .M(T),
        .K(T),
        .N(FF)
    ) u_fpm (
        .clk       (clk),
        .rst       (rst),
        .en        (en),
        .a_wr_en   (fpm_a_wr_en),
        .a_wr_addr (fpm_a_wr_addr),
        .a_wr_data (fpm_a_wr_data),
        .b_wr_en   (fpm_b_wr_en),
        .b_wr_addr (fpm_b_wr_addr),
        .b_wr_data (fpm_b_wr_data),
        .start     (fpm_start),
        .c_row     (fpm_c_row),
        .c_valid   (fpm_c_valid),
        .c_row_idx (fpm_c_row_idx)
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
    typedef enum logic [4:0] {
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
        DONE,
        // ── Backward states ───────────────────────────────────────────────
        // dWff2 = Gᵀ · dY  [FF×T × T×D → FF×D]  (A=Gᵀ, B=dY, K=T, M=FF, N=D)
        BWD_LD_DW2,  BWD_RUN_DW2,  BWD_WAIT_DW2,
        // dG   = dY · Wff2ᵀ [T×D × D×FF → T×FF]  (A=dY, B=Wff2ᵀ decoded to FP32, K=D, M=T, N=FF)
        BWD_LD_DG,   BWD_RUN_DG,   BWD_WAIT_DG,
        // dH1  = dG ⊙ GeLU'(H1)  element-wise
        BWD_GELU_GRAD,
        // dWff1 = Xᵀ · dH1  [D×T × T×FF → D×FF]  (A=Xᵀ, B=dH1, K=T, M=D, N=FF)
        BWD_LD_DW1,  BWD_RUN_DW1,  BWD_WAIT_DW1,
        // dX   = dH1 · Wff1ᵀ [T×FF × FF×D → T×D]  (A=dH1, B=Wff1ᵀ decoded, K=FF, M=T, N=D)
        BWD_LD_DX,   BWD_RUN_DX,   BWD_WAIT_DX
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
            fpm_a_wr_en  <= 0;
            fpm_b_wr_en  <= 0;
            fpm_start    <= 0;
            dx_valid     <= 0;
        end else if (en) begin
            // Defaults
            bwm_a_wr_en  <= 0;
            bwm_b_wr_en  <= 0;
            bwm_start    <= 0;
            gelu_valid_in <= 0;
            out_valid    <= 0;
            fpm_a_wr_en  <= 0;
            fpm_b_wr_en  <= 0;
            fpm_start    <= 0;
            dx_valid     <= 0;

            case (state)

                // ── IDLE ──────────────────────────────────────────────────
                IDLE: begin
                    if (start) begin
                        load_idx <= 0;
                        state    <= LD_W1;
                    end else if (bwd_start) begin
                        load_idx <= 0;
                        state    <= BWD_LD_DW2;
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

                // ══════════════════════════════════════════════════════════
                // BACKWARD STATES
                // ══════════════════════════════════════════════════════════
                // All backward matmuls use u_fpm (fp32_matmul, M=T K=T N=FF).
                // For non-square shapes we only write the needed sub-region.
                //
                // Notation for u_fpm A/B addressing:
                //   A is M×K stored row-major, addr = row*K + col
                //   B is K×N stored row-major, addr = row*N + col
                //   C output is M×N rows streamed via c_valid/c_row/c_row_idx
                //
                // ── BWD_LD_DW2: dWff2 = Gᵀ · dY  (M=FF, K=T, N=D) ──────
                //   A = Gᵀ [FF×T] row-major, addr i*T+k → G_reg[k][i]
                //   B = dY  [T×D]  row-major, addr k*D+j → dY_reg[k][j]
                //   FF*T == T*D == 16 in prototype (all 4)
                BWD_LD_DW2: begin
                    if (load_idx < FF * T) begin
                        automatic int i_aw2 = load_idx / T;
                        automatic int k_aw2 = load_idx % T;
                        automatic int k_bw2 = load_idx / D;
                        automatic int j_bw2 = load_idx % D;
                        fpm_a_wr_en   <= 1;
                        fpm_a_wr_addr <= load_idx[7:0];
                        fpm_a_wr_data <= G_reg[k_aw2][i_aw2];  // Gᵀ[i][k] = G[k][i]
                        fpm_b_wr_en   <= 1;
                        fpm_b_wr_addr <= load_idx[7:0];
                        fpm_b_wr_data <= dY_reg[k_bw2][j_bw2];
                        load_idx <= load_idx + 1;
                    end else begin
                        state <= BWD_RUN_DW2;
                    end
                end

                BWD_RUN_DW2: begin
                    fpm_start <= 1;
                    out_cnt   <= 0;
                    state     <= BWD_WAIT_DW2;
                end

                BWD_WAIT_DW2: begin
                    if (fpm_c_valid) begin
                        // dWff2[fpm_c_row_idx][j] for j in 0..D-1
                        for (int j = 0; j < D; j++)
                            dWff2_reg[fpm_c_row_idx][j] <= fpm_c_row[j*32 +: 32];
                        out_cnt <= out_cnt + 1;
                        if (out_cnt == FF - 1) begin
                            load_idx <= 0;
                            state    <= BWD_LD_DG;
                        end
                    end
                end

                // ── BWD_LD_DG: dG = dY · Wff2ᵀ  (M=T, K=D, N=FF) ───────
                //   A = dY    [T×D]:  addr i*D+k   → dY_reg[i][k]
                //   B = Wff2ᵀ [D×FF]: addr k*FF+j  → Wff2_reg[j][k] decoded BF16→FP32
                //   T*D == D*FF == 16 in prototype (all 4)
                BWD_LD_DG: begin
                    if (load_idx < T * D) begin
                        automatic int          i_dg     = load_idx / D;
                        automatic int          k_dg     = load_idx % D;
                        automatic int          b_addr_dg = k_dg * FF + i_dg;
                        automatic logic [15:0] bf16_dg  = Wff2_reg[i_dg][k_dg]; // Wff2ᵀ[k][j]=Wff2[j][k]
                        fpm_a_wr_en   <= 1;
                        fpm_a_wr_addr <= load_idx[7:0];
                        fpm_a_wr_data <= dY_reg[i_dg][k_dg];
                        fpm_b_wr_en   <= 1;
                        fpm_b_wr_addr <= b_addr_dg[7:0];
                        fpm_b_wr_data <= {bf16_dg, 16'h0000};  // BF16→FP32
                        load_idx <= load_idx + 1;
                    end else begin
                        state <= BWD_RUN_DG;
                    end
                end

                BWD_RUN_DG: begin
                    fpm_start <= 1;
                    out_cnt   <= 0;
                    state     <= BWD_WAIT_DG;
                end

                BWD_WAIT_DG: begin
                    if (fpm_c_valid) begin
                        for (int j = 0; j < FF; j++)
                            dG_reg[fpm_c_row_idx][j] <= fpm_c_row[j*32 +: 32];
                        out_cnt <= out_cnt + 1;
                        if (out_cnt == T - 1)
                            state <= BWD_GELU_GRAD;
                    end
                end

                // ── BWD_GELU_GRAD: dH1 = dG ⊙ GeLU'(H1) ─────────────────
                // Computed entirely in shortreal (no modules needed).
                // c  = 0.7978845608,  c2 = 0.1070322244
                // inner = c*(x + 0.044715*x³)
                // tanh_v = (e^{2*inner}-1)/(e^{2*inner}+1)
                // sech2  = 1 - tanh_v²
                // dGeLU  = 0.5*(1+tanh_v) + 0.5*x*sech2*(c + c2*x²)
                // dH1[i][j] = dG[i][j] * dGeLU(H1[i][j])
                BWD_GELU_GRAD: begin
                    for (int ti = 0; ti < T; ti++) begin
                        for (int fi = 0; fi < FF; fi++) begin
                            automatic shortreal x_sr    = $bitstoshortreal(H1_reg[ti][fi]);
                            automatic shortreal dg_sr   = $bitstoshortreal(dG_reg[ti][fi]);
                            automatic shortreal c_sr    = 0.7978845608;
                            automatic shortreal c2_sr   = 0.1070322244;
                            automatic shortreal x2_sr   = x_sr * x_sr;
                            automatic shortreal inner_sr = c_sr * (x_sr + 0.044715 * x_sr * x2_sr);
                            automatic shortreal e2       = $exp(2.0 * inner_sr);
                            automatic shortreal tanh_sr  = (e2 - 1.0) / (e2 + 1.0);
                            automatic shortreal sech2_sr = 1.0 - tanh_sr * tanh_sr;
                            automatic shortreal dgelu_sr = 0.5 * (1.0 + tanh_sr)
                                                         + 0.5 * x_sr * sech2_sr * (c_sr + c2_sr * x2_sr);
                            dH1_reg[ti][fi] <= $shortrealtobits(dg_sr * dgelu_sr);
                        end
                    end
                    load_idx <= 0;
                    state    <= BWD_LD_DW1;
                end

                // ── BWD_LD_DW1: dWff1 = Xᵀ · dH1  (M=D, K=T, N=FF) ─────
                //   A = Xᵀ  [D×T]: addr i*T+k → X_reg[k][i]
                //   B = dH1 [T×FF]: addr k*FF+j → dH1_reg[k][j]
                //   D*T == T*FF == 16 in prototype; loop addr=i_w1*T+k_w1; B addr=k_w1*FF+i_w1
                BWD_LD_DW1: begin
                    if (load_idx < D * T) begin
                        automatic int i_w1     = load_idx / T;
                        automatic int k_w1     = load_idx % T;
                        automatic int b_addr_w1 = k_w1 * FF + i_w1;
                        fpm_a_wr_en   <= 1;
                        fpm_a_wr_addr <= load_idx[7:0];
                        fpm_a_wr_data <= X_reg[k_w1][i_w1];    // Xᵀ[i][k] = X[k][i]
                        fpm_b_wr_en   <= 1;
                        fpm_b_wr_addr <= b_addr_w1[7:0];
                        fpm_b_wr_data <= dH1_reg[k_w1][i_w1];  // dH1[k][j], j==i_w1 (D==FF)
                        load_idx <= load_idx + 1;
                    end else begin
                        state <= BWD_RUN_DW1;
                    end
                end

                BWD_RUN_DW1: begin
                    fpm_start <= 1;
                    out_cnt   <= 0;
                    state     <= BWD_WAIT_DW1;
                end

                BWD_WAIT_DW1: begin
                    if (fpm_c_valid) begin
                        for (int j = 0; j < FF; j++)
                            dWff1_reg[fpm_c_row_idx][j] <= fpm_c_row[j*32 +: 32];
                        out_cnt <= out_cnt + 1;
                        if (out_cnt == D - 1) begin
                            load_idx <= 0;
                            state    <= BWD_LD_DX;
                        end
                    end
                end

                // ── BWD_LD_DX: dX = dH1 · Wff1ᵀ  (M=T, K=FF, N=D) ──────
                //   A = dH1  [T×FF]:  addr i*FF+k  → dH1_reg[i][k]
                //   B = Wff1ᵀ[FF×D]: addr k*D+j   → Wff1_reg[j][k] decoded BF16→FP32
                //   T*FF == FF*D == 16 in prototype (all 4); j=i_dx, k=k_dx (D==FF)
                BWD_LD_DX: begin
                    if (load_idx < T * FF) begin
                        automatic int          i_dx      = load_idx / FF;
                        automatic int          k_dx      = load_idx % FF;
                        automatic int          b_addr_dx = k_dx * D + i_dx;
                        automatic logic [15:0] bf16_dx   = Wff1_reg[i_dx][k_dx]; // Wff1ᵀ[k][j]=Wff1[j][k]
                        fpm_a_wr_en   <= 1;
                        fpm_a_wr_addr <= load_idx[7:0];
                        fpm_a_wr_data <= dH1_reg[i_dx][k_dx];
                        fpm_b_wr_en   <= 1;
                        fpm_b_wr_addr <= b_addr_dx[7:0];
                        fpm_b_wr_data <= {bf16_dx, 16'h0000};  // BF16→FP32
                        load_idx <= load_idx + 1;
                    end else begin
                        state <= BWD_RUN_DX;
                    end
                end

                BWD_RUN_DX: begin
                    fpm_start <= 1;
                    out_cnt   <= 0;
                    state     <= BWD_WAIT_DX;
                end

                BWD_WAIT_DX: begin
                    if (fpm_c_valid) begin
                        dx_row     <= fpm_c_row[D*32-1:0];
                        dx_row_idx <= fpm_c_row_idx[$clog2(T)-1:0];
                        dx_valid   <= 1;
                        out_cnt    <= out_cnt + 1;
                        if (out_cnt == T - 1)
                            state <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
