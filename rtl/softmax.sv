// Per-row FP32 softmax with causal mask — forward pass for scaled dot-product attention
//
// Algorithm (matches AttentionCore.Softmax + ExpLutHelper with LUT-256):
//   For row i (0-based), positions j > i are causal-masked (score → -∞)
//   1. Apply mask: masked positions set to NEG_INF
//   2. Find max over all T positions (masked positions don't affect max)
//   3. Subtract max from each score (keeps numerics stable)
//   4. exp_lut applied to each shifted score serially (LATENCY=4 cycles each)
//   5. fp32_add_tree sums the T exp values
//   6. fp32_div normalises each exp value by sum
//
// Pipeline:
//   IDLE → FIND_MAX (1 cycle) → EXP_FEED (T cycles) →
//   EXP_COLLECT (T cycles, collecting valid_out from exp_lut) →
//   SUM_DIV (1 cycle) → OUTPUT_VALID (1 cycle, valid_out=1) → IDLE
//
//   Total latency from start pulse: 2 + 2*T + EXP_LAT cycles
//   For T=4, EXP_LAT=4: 2 + 8 + 4 = 14 cycles
//   Note: first exp output arrives exactly when EXP_COLLECT begins (EXP_LAT == T)
//
// Interface:
//   scores_in  — T packed FP32 scores, valid on the start cycle
//   row_idx    — 0-based row index (used to compute causal mask)
//   start      — 1-cycle pulse to begin processing
//   probs_out  — T packed FP32 probabilities (valid when valid_out=1)
//   valid_out  — 1-cycle pulse when probs_out is ready
//
// Sub-modules instantiated:
//   exp_lut      — per-element exp (serialised, 1 element per cycle)
//   fp32_add_tree — reduction sum over T exp values
//   fp32_div (×T) — normalise each exp value by sum
//
// Parameters:
//   T        — sequence length (4)
//   EXP_LAT  — exp_lut pipeline depth (4)
//   LUT_SIZE — exp LUT entries (256)
//   LUT_FILE — BRAM init hex (relative to xsim working directory)

`timescale 1ns/1ps

module softmax #(
    parameter int    T        = 4,
    parameter int    EXP_LAT  = 4,
    parameter int    LUT_SIZE = 256,
    parameter string LUT_FILE = "exp_lut_init.hex"
)(
    input  logic clk,
    input  logic rst,
    input  logic en,

    input  logic [T*32-1:0]       scores_in,
    input  logic [$clog2(T)-1:0]  row_idx,
    input  logic                   start,

    output logic [T*32-1:0]       probs_out,
    output logic                   valid_out
);

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------
    localparam logic [31:0] NEG_INF_BITS = 32'hFF800000; // -inf

    // -----------------------------------------------------------------------
    // exp_lut instance — serialised: one element per clock
    // -----------------------------------------------------------------------
    logic [31:0] exp_x_reg;
    logic        exp_valid_in_reg;
    logic [31:0] exp_result;
    logic        exp_valid_out;

    exp_lut #(
        .LUT_SIZE(LUT_SIZE),
        .LUT_FILE(LUT_FILE)
    ) u_exp (
        .clk         (clk),
        .rst         (rst),
        .en          (en),
        .x_fp32      (exp_x_reg),
        .valid_in    (exp_valid_in_reg),
        .result_fp32 (exp_result),
        .valid_out   (exp_valid_out)
    );

    // -----------------------------------------------------------------------
    // fp32_add_tree — combinatorial, driven from exp_buf
    // -----------------------------------------------------------------------
    logic [T*32-1:0] exp_buf_packed;
    logic [31:0]     sum_bits;

    fp32_add_tree #(.T(T)) u_add_tree (
        .in_vec  (exp_buf_packed),
        .sum_out (sum_bits)
    );

    // -----------------------------------------------------------------------
    // fp32_div — T instances, each normalising one exp value by sum
    // -----------------------------------------------------------------------
    logic [T*32-1:0] probs_comb;

    genvar gi;
    generate
        for (gi = 0; gi < T; gi++) begin : gen_div
            fp32_div u_div (
                .a_fp32      (exp_buf_packed[gi*32 +: 32]),
                .b_fp32      (sum_bits),
                .result_fp32 (probs_comb[gi*32 +: 32])
            );
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Internal registers
    // -----------------------------------------------------------------------
    logic [31:0] score_buf  [0:T-1];  // masked scores
    logic [31:0] shifted_buf[0:T-1];  // score - max
    logic [31:0] exp_buf    [0:T-1];  // collected exp outputs
    logic [31:0] probs_reg  [0:T-1];  // final probabilities (registered)

    // Pack exp_buf into exp_buf_packed for add_tree / fp32_div wiring
    always_comb begin
        for (int j = 0; j < T; j++)
            exp_buf_packed[j*32 +: 32] = exp_buf[j];
    end

    // Pack probs_reg into probs_out
    always_comb begin
        for (int j = 0; j < T; j++)
            probs_out[j*32 +: 32] = probs_reg[j];
    end

    // -----------------------------------------------------------------------
    // Combinatorial max over score_buf (masked -inf values never win)
    // -----------------------------------------------------------------------
    shortreal max_comb;
    always_comb begin
        max_comb = $bitstoshortreal(score_buf[0]);
        for (int j = 1; j < T; j++) begin
            shortreal v = $bitstoshortreal(score_buf[j]);
            if (v > max_comb) max_comb = v;
        end
    end

    // -----------------------------------------------------------------------
    // State machine
    // -----------------------------------------------------------------------
    typedef enum logic [2:0] {
        IDLE,
        FIND_MAX,
        EXP_FEED,
        EXP_COLLECT,
        SUM_DIV,
        OUTPUT_VALID
    } state_t;

    state_t      state;
    int unsigned feed_cnt;
    int unsigned collect_cnt;

    always_ff @(posedge clk) begin
        if (rst) begin
            state            <= IDLE;
            feed_cnt         <= 0;
            collect_cnt      <= 0;
            valid_out        <= 1'b0;
            exp_valid_in_reg <= 1'b0;
            for (int j = 0; j < T; j++) begin
                score_buf  [j] <= '0;
                shifted_buf[j] <= '0;
                exp_buf    [j] <= '0;
                probs_reg  [j] <= '0;
            end
        end else if (en) begin
            valid_out        <= 1'b0;
            exp_valid_in_reg <= 1'b0;

            case (state)

                // ---- Wait for start pulse --------------------------------
                IDLE: begin
                    if (start) begin
                        // Latch inputs with causal mask applied
                        for (int j = 0; j < T; j++) begin
                            if (j > int'(row_idx))
                                score_buf[j] <= NEG_INF_BITS;
                            else
                                score_buf[j] <= scores_in[j*32 +: 32];
                        end
                        state <= FIND_MAX;
                    end
                end

                // ---- Register (score[j] - max) into shifted_buf ---------
                // max_comb is combinatorial from score_buf (set in previous cycle)
                FIND_MAX: begin
                    for (int j = 0; j < T; j++) begin
                        shortreal s = $bitstoshortreal(score_buf[j]);
                        shifted_buf[j] <= $shortrealtobits(s - max_comb);
                    end
                    feed_cnt    <= 0;
                    collect_cnt <= 0;
                    state       <= EXP_FEED;
                end

                // ---- Feed shifted scores through exp_lut one per cycle ---
                EXP_FEED: begin
                    exp_x_reg        <= shifted_buf[feed_cnt];
                    exp_valid_in_reg <= 1'b1;
                    feed_cnt         <= feed_cnt + 1;
                    if (feed_cnt == T - 1)
                        state <= EXP_COLLECT;
                end

                // ---- Collect exp_lut outputs (arrives EXP_LAT after each feed) --
                // For T==EXP_LAT==4 the first valid_out arrives on the first cycle here.
                EXP_COLLECT: begin
                    if (exp_valid_out) begin
                        exp_buf[collect_cnt] <= exp_result;
                        collect_cnt          <= collect_cnt + 1;
                        if (collect_cnt == T - 1)
                            state <= SUM_DIV;
                    end
                end

                // ---- Combinatorial sum+div results are valid now ----------
                // fp32_add_tree and fp32_div are combinatorial from exp_buf.
                // Register their outputs (probs_comb) into probs_reg.
                SUM_DIV: begin
                    for (int j = 0; j < T; j++)
                        probs_reg[j] <= probs_comb[j*32 +: 32];
                    state <= OUTPUT_VALID;
                end

                // ---- Assert valid_out for 1 cycle, then return to IDLE ---
                OUTPUT_VALID: begin
                    valid_out <= 1'b1;
                    state     <= IDLE;
                end

            endcase
        end
    end

endmodule
