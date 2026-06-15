// Cross-entropy gradient computation — standalone hardware-accurate module
//
// Computes dX_init and dWout from raw logits, targets, embedding table, and
// the last-layer activations x, matching TransformerBus.TrainStep CE path.
//
// Write bus input layout (address → data):
//   0x000..0x03F : logits_reg[T][V]  FP32  addr = t*V + v   (T=4, V=16 → 64 entries)
//   0x040..0x043 : targets_reg[T]    int   data[3:0] = token index 0..V-1
//   0x044..0x083 : emb_reg[V][D]     FP32  addr = 0x044 + v*D + d (V=16, D=4 → 64 entries)
//   0x084..0x093 : x_reg[T][D]       FP32  addr = 0x084 + t*D + d (T=4, D=4 → 16 entries)
//
// Output ports (valid when done=1, held until next start):
//   dX_init_out [T*D*32-1:0]  : dX_init[t][d] at bit offset (t*D+d)*32
//   dWout_out   [V*D*32-1:0]  : dWout  [v][d] at bit offset (v*D+d)*32
//
// Algorithm (matches TransformerBus.TrainStep + ExpLutHelper exactly):
//   1. Softmax per row (max-stable): exp via exp_lut.sv, sum via shortreal add
//   2. Subtract 1 at target
//   3. Divide by T (seqLen norm)
//   4. Grad norm clip to 1.0: 1/sqrt via fp32_sqrt.sv
//   5. dX_init[t][d] = sum_v dLogits[t][v] * emb[v][d]
//   6. dWout  [v][d] = sum_t dLogits[t][v] * x  [t][d]
//
// Hardware primitives used:
//   exp_lut.sv    — pipelined exp (EXP_LAT cycles), matches ExpLutHelper.Exp()
//   fp32_sqrt.sv  — combinatorial 1/sqrt, matches ExpLutHelper.RecipSqrt()
//   shortreal     — add/multiply only (no $exp/$sqrt)
//
// C# reference: FpgaCeGradVecGen.Build() — uses ExpLutHelper.Exp + RecipSqrt.
// Tolerance: VsSoftwareRelTol = 0.01%.

`timescale 1ns/1ps

module ce_grad #(
    parameter int    T        = 4,
    parameter int    D        = 4,
    parameter int    V        = 16,
    parameter int    EXP_LAT  = 4,
    parameter int    LUT_SIZE = 256,
    parameter string LUT_FILE = "exp_lut_init.hex"
)(
    input  logic        clk,
    input  logic        rst,

    // Write bus
    input  logic        wr_en,
    input  logic [8:0]  wr_addr,
    input  logic [31:0] wr_data,

    // Control
    input  logic        start,
    output logic        done,

    // Outputs (flat packed, valid when done=1)
    output logic [T*D*32-1:0] dX_init_out,
    output logic [V*D*32-1:0] dWout_out
);
    // -----------------------------------------------------------------------
    // Input registers
    // -----------------------------------------------------------------------
    logic [31:0] logits_reg  [0:T-1][0:V-1];
    logic [3:0]  targets_reg [0:T-1];
    logic [31:0] emb_reg     [0:V-1][0:D-1];
    logic [31:0] x_reg       [0:T-1][0:D-1];

    // Output registers (FP32 bits)
    logic [31:0] dX_init_reg [0:T-1][0:D-1];
    logic [31:0] dWout_reg   [0:V-1][0:D-1];

    // Working registers
    shortreal     max_logit_sr;
    logic [31:0]  exp_buf_fp32 [0:V-1];   // raw FP32 output from exp_lut
    shortreal     exp_sum_sr;
    shortreal     dLogits_sr   [0:T-1][0:V-1];
    shortreal     grad_norm_sq_sr;
    shortreal     clip_scale_sr;

    // -----------------------------------------------------------------------
    // exp_lut instance — one element per cycle, EXP_LAT pipeline stages
    // -----------------------------------------------------------------------
    logic [31:0] exp_x_reg;
    logic        exp_valid_in_reg;
    logic [31:0] exp_result_fp32;
    logic        exp_valid_out;

    exp_lut #(.LUT_SIZE(LUT_SIZE), .LUT_FILE(LUT_FILE)) u_exp (
        .clk         (clk),
        .rst         (rst),
        .en          (1'b1),
        .x_fp32      (exp_x_reg),
        .valid_in    (exp_valid_in_reg),
        .result_fp32 (exp_result_fp32),
        .valid_out   (exp_valid_out)
    );

    // -----------------------------------------------------------------------
    // fp32_sqrt instance — combinatorial 1/sqrt(grad_norm_sq)
    // -----------------------------------------------------------------------
    logic [31:0] sqrt_in_fp32;
    logic [31:0] sqrt_result_fp32;

    assign sqrt_in_fp32 = $shortrealtobits(grad_norm_sq_sr);

    fp32_sqrt u_sqrt (
        .x_fp32      (sqrt_in_fp32),
        .result_fp32 (sqrt_result_fp32)
    );

    // -----------------------------------------------------------------------
    // Output pack
    // -----------------------------------------------------------------------
    genvar gt, gd, gv;
    generate
        for (gt = 0; gt < T; gt++) for (gd = 0; gd < D; gd++)
            assign dX_init_out[(gt*D+gd)*32 +: 32] = dX_init_reg[gt][gd];
        for (gv = 0; gv < V; gv++) for (gd = 0; gd < D; gd++)
            assign dWout_out[(gv*D+gd)*32 +: 32] = dWout_reg[gv][gd];
    endgenerate

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    typedef enum logic [3:0] {
        IDLE,
        CE_ROW_MAX,         // find max logit per row (V cycles)
        CE_ROW_EXP_FEED,    // feed logit-max to exp_lut (V cycles)
        CE_ROW_EXP_DRAIN,   // drain last EXP_LAT results (EXP_LAT cycles)
        CE_ROW_NORM,        // dLogits = exp_buf/sum, subtract 1 at target (V cycles)
        CE_DIV_T,           // divide all T×V dLogits by T (T*V cycles)
        CE_GRADNORM,        // accumulate ||dLogits||^2 (T*V cycles)
        CE_CLIP,            // compute clip_scale from fp32_sqrt (1 cycle)
        CE_CLIP_APPLY,      // apply clip to all T×V dLogits (T*V cycles)
        CE_DX_INIT,         // dX_init = dLogits · emb (V cycles, parallel over T×D)
        CE_DWOUT,           // dWout   = dLogitsᵀ · x  (T cycles, parallel over V×D)
        DONE_ST
    } state_t;

    state_t     state;
    logic [6:0] cnt;       // feed counter (up to max(V-1,T*V-1)=63)
    logic [1:0] ln_row;    // current token row (0..T-1)
    logic [4:0] coll_idx;  // exp collection index (0..V-1); 5-bit to detect V

    // -----------------------------------------------------------------------
    // Write bus
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (wr_en) begin
            if (wr_addr <= 9'h03F)
                logits_reg[wr_addr[5:4]][wr_addr[3:0]] <= wr_data;
            else if (wr_addr >= 9'h040 && wr_addr <= 9'h043)
                targets_reg[wr_addr[1:0]] <= wr_data[3:0];
            else if (wr_addr >= 9'h044 && wr_addr <= 9'h083)
                emb_reg[(wr_addr - 9'h044) >> 2][(wr_addr - 9'h044) & 2'h3] <= wr_data;
            else if (wr_addr >= 9'h084 && wr_addr <= 9'h093)
                x_reg[(wr_addr - 9'h084) >> 2][(wr_addr - 9'h084) & 2'h3] <= wr_data;
        end
    end

    // -----------------------------------------------------------------------
    // Collect exp_lut results whenever valid_out fires (any state)
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            coll_idx   <= '0;
            exp_sum_sr <= 0.0;
        end else begin
            if (exp_valid_out && coll_idx < 5'(V)) begin
                exp_buf_fp32[coll_idx[3:0]] <= exp_result_fp32;
                exp_sum_sr                  <= exp_sum_sr + $bitstoshortreal(exp_result_fp32);
                coll_idx                    <= coll_idx + 1;
            end
        end
    end

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state         <= IDLE;
            done          <= 0;
            cnt           <= '0;
            ln_row        <= '0;
            max_logit_sr  <= 0.0;
            grad_norm_sq_sr <= 0.0;
            clip_scale_sr <= 1.0;
            exp_valid_in_reg <= 0;
            exp_x_reg        <= '0;
        end else begin
            done             <= 0;
            exp_valid_in_reg <= 0;   // default: no feed

            case (state)
                IDLE: begin
                    if (start) begin
                        cnt <= '0; ln_row <= '0; max_logit_sr <= 0.0;
                        state <= CE_ROW_MAX;
                    end
                end

                // ----------------------------------------------------------
                // Step 1 — find max logit per row (shortreal comparison only)
                // ----------------------------------------------------------
                CE_ROW_MAX: begin
                    begin
                        automatic shortreal lv = $bitstoshortreal(logits_reg[ln_row][cnt[3:0]]);
                        if (cnt[3:0] == '0) max_logit_sr <= lv;
                        else if (lv > max_logit_sr) max_logit_sr <= lv;
                    end
                    if (cnt[3:0] == 4'(V-1)) begin
                        cnt          <= '0;
                        coll_idx     <= '0;
                        exp_sum_sr   <= 0.0;
                        state        <= CE_ROW_EXP_FEED;
                    end else cnt <= cnt + 1;
                end

                // ----------------------------------------------------------
                // Step 2a — feed V elements to exp_lut (V cycles)
                //   Simultaneously collect whenever exp_valid_out fires
                // ----------------------------------------------------------
                CE_ROW_EXP_FEED: begin
                    begin
                        automatic shortreal shifted =
                            $bitstoshortreal(logits_reg[ln_row][cnt[3:0]]) - max_logit_sr;
                        exp_x_reg        <= $shortrealtobits(shifted);
                        exp_valid_in_reg <= 1'b1;
                    end
                    if (cnt[3:0] == 4'(V-1)) begin
                        cnt   <= '0;
                        state <= CE_ROW_EXP_DRAIN;
                    end else cnt <= cnt + 1;
                end

                // ----------------------------------------------------------
                // Step 2b — drain last EXP_LAT results (EXP_LAT cycles)
                // ----------------------------------------------------------
                CE_ROW_EXP_DRAIN: begin
                    if (cnt == 7'(EXP_LAT-1)) begin
                        cnt   <= '0;
                        state <= CE_ROW_NORM;
                    end else cnt <= cnt + 1;
                end

                // ----------------------------------------------------------
                // Step 3 — dLogits[t][v] = exp_buf[v]/sum, subtract 1 at target
                // ----------------------------------------------------------
                CE_ROW_NORM: begin
                    begin
                        automatic shortreal dl =
                            $bitstoshortreal(exp_buf_fp32[cnt[3:0]]) / exp_sum_sr;
                        if (cnt[3:0] == 4'(targets_reg[ln_row])) dl = dl - 1.0;
                        dLogits_sr[ln_row][cnt[3:0]] <= dl;
                    end
                    if (cnt[3:0] == 4'(V-1)) begin
                        if (ln_row == 2'(T-1)) begin
                            cnt <= '0; state <= CE_DIV_T;
                        end else begin
                            ln_row       <= ln_row + 1;
                            coll_idx     <= '0;
                            exp_sum_sr   <= 0.0;
                            max_logit_sr <= 0.0;
                            cnt          <= '0;
                            state        <= CE_ROW_MAX;
                        end
                    end else cnt <= cnt + 1;
                end

                // ----------------------------------------------------------
                // Step 4 — divide all T×V dLogits by T  (cnt[5:4]=t, cnt[3:0]=v)
                // ----------------------------------------------------------
                CE_DIV_T: begin
                    dLogits_sr[cnt[5:4]][cnt[3:0]] <=
                        dLogits_sr[cnt[5:4]][cnt[3:0]] / shortreal'(T);
                    if (cnt[5:0] == 6'(T*V-1)) begin
                        cnt <= '0; grad_norm_sq_sr <= 0.0; state <= CE_GRADNORM;
                    end else cnt <= cnt + 1;
                end

                // ----------------------------------------------------------
                // Step 5 — accumulate ||dLogits||^2
                // ----------------------------------------------------------
                CE_GRADNORM: begin
                    begin
                        automatic shortreal dl = dLogits_sr[cnt[5:4]][cnt[3:0]];
                        grad_norm_sq_sr <= grad_norm_sq_sr + dl * dl;
                    end
                    if (cnt[5:0] == 6'(T*V-1)) begin
                        cnt <= '0; state <= CE_CLIP;
                    end else cnt <= cnt + 1;
                end

                // ----------------------------------------------------------
                // Step 6 — clip_scale from fp32_sqrt (combinatorial, 1 cycle)
                //   fp32_sqrt gives 1/sqrt(grad_norm_sq_sr)
                //   clip = 1/gnorm  when gnorm > 1  (i.e., norm_sq > 1)
                // ----------------------------------------------------------
                CE_CLIP: begin
                    if (grad_norm_sq_sr > 1.0)
                        clip_scale_sr <= $bitstoshortreal(sqrt_result_fp32);
                    else
                        clip_scale_sr <= 1.0;
                    cnt   <= '0;
                    state <= CE_CLIP_APPLY;
                end

                // ----------------------------------------------------------
                // Step 7 — apply clip
                // ----------------------------------------------------------
                CE_CLIP_APPLY: begin
                    dLogits_sr[cnt[5:4]][cnt[3:0]] <=
                        dLogits_sr[cnt[5:4]][cnt[3:0]] * clip_scale_sr;
                    if (cnt[5:0] == 6'(T*V-1)) begin
                        cnt <= '0;
                        // initialise output accumulators
                        for (int ti = 0; ti < T; ti++)
                            for (int di = 0; di < D; di++)
                                dX_init_reg[ti][di] <= '0;
                        state <= CE_DX_INIT;
                    end else cnt <= cnt + 1;
                end

                // ----------------------------------------------------------
                // Step 8 — dX_init[t][d] = sum_v dLogits[t][v] * emb[v][d]
                //   cnt[3:0] = v, parallel over all (t,d)
                // ----------------------------------------------------------
                CE_DX_INIT: begin
                    for (int ti = 0; ti < T; ti++) for (int di = 0; di < D; di++) begin
                        automatic shortreal dl  = dLogits_sr[ti][cnt[3:0]];
                        automatic shortreal ew  = $bitstoshortreal(emb_reg[cnt[3:0]][di]);
                        automatic shortreal acc = $bitstoshortreal(dX_init_reg[ti][di]);
                        dX_init_reg[ti][di] <= $shortrealtobits(acc + dl * ew);
                    end
                    if (cnt[3:0] == 4'(V-1)) begin
                        cnt <= '0;
                        for (int vi = 0; vi < V; vi++)
                            for (int di = 0; di < D; di++)
                                dWout_reg[vi][di] <= '0;
                        state <= CE_DWOUT;
                    end else cnt <= cnt + 1;
                end

                // ----------------------------------------------------------
                // Step 9 — dWout[v][d] = sum_t dLogits[t][v] * x[t][d]
                //   cnt[1:0] = t, parallel over all (v,d)
                // ----------------------------------------------------------
                CE_DWOUT: begin
                    for (int vi = 0; vi < V; vi++) for (int di = 0; di < D; di++) begin
                        automatic shortreal dl  = dLogits_sr[cnt[1:0]][vi];
                        automatic shortreal xv  = $bitstoshortreal(x_reg[cnt[1:0]][di]);
                        automatic shortreal acc = $bitstoshortreal(dWout_reg[vi][di]);
                        dWout_reg[vi][di] <= $shortrealtobits(acc + dl * xv);
                    end
                    if (cnt[1:0] == 2'(T-1)) state <= DONE_ST;
                    else cnt <= cnt + 1;
                end

                DONE_ST: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
//
// Computes dX_init and dWout from raw logits, targets, embedding table, and
// the last-layer activations x, matching TransformerBus.TrainStep CE path.
//
// Write bus input layout (address → data):
//   0x000..0x03F : logits_reg[T][V]  FP32  addr = t*V + v   (T=4, V=16 → 64 entries)
//   0x040..0x043 : targets_reg[T]    int   data[3:0] = token index 0..V-1
//   0x044..0x083 : emb_reg[V][D]     FP32  addr = 0x044 + v*D + d (V=16, D=4 → 64 entries)
//   0x084..0x093 : x_reg[T][D]       FP32  addr = 0x084 + t*D + d (T=4, D=4 → 16 entries)
//
// Output ports (valid when done=1, held until next start):
//   dX_init_out [T*D*32-1:0]  : dX_init[t][d] at bit offset (t*D+d)*32
//   dWout_out   [V*D*32-1:0]  : dWout  [v][d] at bit offset (v*D+d)*32
//
// Algorithm (matches C# TransformerBus.TrainStep):
//   1. Softmax per row (max-stable): dLogits[t][v] = exp(l-max)/sum
//   2. Subtract 1 at target:         dLogits[t][target] -= 1
//   3. Divide by T (seqLen norm)
//   4. Clip grad norm to 1.0:        clip_scale = min(1, 1/||dLogits||)
//   5. dX_init[t][d] = sum_v dLogits[t][v] * emb[v][d]
//   6. dWout  [v][d] = sum_t dLogits[t][v] * x  [t][d]
//
// Arithmetic: shortreal + $exp + $sqrt — simulation only (Phase 1 correctness proof).
// C# reference: FpgaCeGradVecGen.Build() in Neuro.Attention.XSim.LocalTests.
// Tolerance: VsSoftwareRelTol = 0.01%.

`timescale 1ns/1ps

module ce_grad #(
    parameter int T = 4,
    parameter int D = 4,
    parameter int V = 16
)(
    input  logic        clk,
    input  logic        rst,

    // Write bus
    input  logic        wr_en,
    input  logic [8:0]  wr_addr,
    input  logic [31:0] wr_data,

    // Control
    input  logic        start,
    output logic        done,

    // Outputs (flat packed, valid when done=1)
    output logic [T*D*32-1:0] dX_init_out,
    output logic [V*D*32-1:0] dWout_out
);
    // -----------------------------------------------------------------------
    // Input registers
    // -----------------------------------------------------------------------
    logic [31:0] logits_reg  [0:T-1][0:V-1];
    logic [3:0]  targets_reg [0:T-1];
    logic [31:0] emb_reg     [0:V-1][0:D-1];
    logic [31:0] x_reg       [0:T-1][0:D-1];

    // Output registers (FP32 bits)
    logic [31:0] dX_init_reg [0:T-1][0:D-1];
    logic [31:0] dWout_reg   [0:V-1][0:D-1];

    // Working shortreal registers
    shortreal max_logit_sr, exp_sum_sr;
    shortreal exp_buf_sr   [0:V-1];
    shortreal dLogits_sr   [0:T-1][0:V-1];
    shortreal grad_norm_sq_sr, clip_scale_sr;

    // -----------------------------------------------------------------------
    // Output pack
    // -----------------------------------------------------------------------
    genvar gt, gd, gv;
    generate
        for (gt = 0; gt < T; gt++) for (gd = 0; gd < D; gd++)
            assign dX_init_out[(gt*D+gd)*32 +: 32] = dX_init_reg[gt][gd];
        for (gv = 0; gv < V; gv++) for (gd = 0; gd < D; gd++)
            assign dWout_out[(gv*D+gd)*32 +: 32] = dWout_reg[gv][gd];
    endgenerate

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    typedef enum logic [3:0] {
        IDLE,
        CE_ROW_MAX,     // find max logit per row (V cycles)
        CE_ROW_EXP,     // exp(logit-max), accumulate sum (V cycles)
        CE_ROW_NORM,    // normalise → dLogits, subtract 1 at target (V cycles)
        CE_DIV_T,       // divide all T×V dLogits by T (T*V cycles)
        CE_GRADNORM,    // accumulate ||dLogits||^2 (T*V cycles)
        CE_CLIP,        // compute clip_scale (1 cycle)
        CE_CLIP_APPLY,  // apply clip to all T×V dLogits (T*V cycles)
        CE_DX_INIT,     // dX_init = dLogits · emb (V cycles, parallel over T×D)
        CE_DWOUT,       // dWout   = dLogitsᵀ · x  (T cycles, parallel over V×D)
        DONE_ST
    } state_t;

    state_t     state;
    logic [6:0] cnt;      // up to T*V-1 = 63
    logic [1:0] ln_row;   // current token row (0..T-1)

    // -----------------------------------------------------------------------
    // Write bus
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (wr_en) begin
            // logits[T][V]: addr = t*V + v, addr 0x000..0x03F
            if (wr_addr <= 9'h03F)
                logits_reg[wr_addr[5:4]][wr_addr[3:0]] <= wr_data;
            // targets[T]: addr 0x040..0x043
            else if (wr_addr >= 9'h040 && wr_addr <= 9'h043)
                targets_reg[wr_addr[1:0]] <= wr_data[3:0];
            // emb[V][D]: addr = 0x044 + v*D + d  → offset = addr - 0x44
            //   v = offset[5:2],  d = offset[1:0]  (D=4 so /4 = >>2)
            else if (wr_addr >= 9'h044 && wr_addr <= 9'h083)
                emb_reg[(wr_addr - 9'h044) >> 2][(wr_addr - 9'h044) & 2'h3] <= wr_data;
            // x[T][D]: addr = 0x084 + t*D + d
            //   t = offset[3:2], d = offset[1:0]
            else if (wr_addr >= 9'h084 && wr_addr <= 9'h093)
                x_reg[(wr_addr - 9'h084) >> 2][(wr_addr - 9'h084) & 2'h3] <= wr_data;
        end
    end

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE; done <= 0; cnt <= '0; ln_row <= '0;
            max_logit_sr <= 0.0; exp_sum_sr <= 0.0;
            grad_norm_sq_sr <= 0.0; clip_scale_sr <= 1.0;
        end else begin
            done <= 0;

            case (state)
                IDLE: begin
                    if (start) begin
                        cnt <= '0; ln_row <= '0; max_logit_sr <= 0.0;
                        state <= CE_ROW_MAX;
                    end
                end

                // ----------------------------------------------------------
                // Step 1 — find max logit per token row
                // ----------------------------------------------------------
                CE_ROW_MAX: begin
                    begin
                        automatic shortreal lv = $bitstoshortreal(logits_reg[ln_row][cnt[3:0]]);
                        if (cnt[3:0] == '0) max_logit_sr <= lv;
                        else if (lv > max_logit_sr) max_logit_sr <= lv;
                    end
                    if (cnt[3:0] == 4'(V-1)) begin
                        cnt <= '0; exp_sum_sr <= 0.0; state <= CE_ROW_EXP;
                    end else cnt <= cnt + 1;
                end

                // ----------------------------------------------------------
                // Step 2 — exp(logit - max), accumulate sum
                // ----------------------------------------------------------
                CE_ROW_EXP: begin
                    begin
                        automatic shortreal lv = $bitstoshortreal(logits_reg[ln_row][cnt[3:0]]);
                        automatic shortreal e  = $exp(lv - max_logit_sr);
                        exp_buf_sr[cnt[3:0]] <= e;
                        exp_sum_sr <= exp_sum_sr + e;
                    end
                    if (cnt[3:0] == 4'(V-1)) begin
                        cnt <= '0; state <= CE_ROW_NORM;
                    end else cnt <= cnt + 1;
                end

                // ----------------------------------------------------------
                // Step 3 — normalise, subtract 1 at target
                // ----------------------------------------------------------
                CE_ROW_NORM: begin
                    begin
                        automatic shortreal dl = exp_buf_sr[cnt[3:0]] / exp_sum_sr;
                        if (cnt[3:0] == 4'(targets_reg[ln_row])) dl = dl - 1.0;
                        dLogits_sr[ln_row][cnt[3:0]] <= dl;
                    end
                    if (cnt[3:0] == 4'(V-1)) begin
                        if (ln_row == 2'(T-1)) begin
                            cnt <= '0; state <= CE_DIV_T;
                        end else begin
                            ln_row <= ln_row + 1;
                            cnt <= '0; max_logit_sr <= 0.0; exp_sum_sr <= 0.0;
                            state <= CE_ROW_MAX;
                        end
                    end else cnt <= cnt + 1;
                end

                // ----------------------------------------------------------
                // Step 4 — divide all T×V dLogits by T (seqLen norm)
                //   cnt[5:4] = t,  cnt[3:0] = v
                // ----------------------------------------------------------
                CE_DIV_T: begin
                    dLogits_sr[cnt[5:4]][cnt[3:0]] <=
                        dLogits_sr[cnt[5:4]][cnt[3:0]] / shortreal'(T);
                    if (cnt[5:0] == 6'(T*V-1)) begin
                        cnt <= '0; grad_norm_sq_sr <= 0.0; state <= CE_GRADNORM;
                    end else cnt <= cnt + 1;
                end

                // ----------------------------------------------------------
                // Step 5 — accumulate ||dLogits||^2 for grad norm clipping
                // ----------------------------------------------------------
                CE_GRADNORM: begin
                    begin
                        automatic shortreal dl = dLogits_sr[cnt[5:4]][cnt[3:0]];
                        grad_norm_sq_sr <= grad_norm_sq_sr + dl * dl;
                    end
                    if (cnt[5:0] == 6'(T*V-1)) begin
                        cnt <= '0; state <= CE_CLIP;
                    end else cnt <= cnt + 1;
                end

                // ----------------------------------------------------------
                // Step 6 — compute clip_scale (1 cycle)
                // ----------------------------------------------------------
                CE_CLIP: begin
                    begin
                        automatic shortreal gnorm = $sqrt(grad_norm_sq_sr);
                        clip_scale_sr <= (gnorm > 1.0) ? (1.0 / gnorm) : 1.0;
                    end
                    cnt <= '0; state <= CE_CLIP_APPLY;
                end

                // ----------------------------------------------------------
                // Step 7 — apply clip to all T×V dLogits
                // ----------------------------------------------------------
                CE_CLIP_APPLY: begin
                    dLogits_sr[cnt[5:4]][cnt[3:0]] <=
                        dLogits_sr[cnt[5:4]][cnt[3:0]] * clip_scale_sr;
                    if (cnt[5:0] == 6'(T*V-1)) begin
                        cnt <= '0; state <= CE_DX_INIT;
                        // Initialise dX_init_reg to 0 before accumulation
                        for (int ti = 0; ti < T; ti++)
                            for (int di = 0; di < D; di++)
                                dX_init_reg[ti][di] <= '0;
                    end else cnt <= cnt + 1;
                end

                // ----------------------------------------------------------
                // Step 8 — dX_init[t][d] = sum_v dLogits[t][v] * emb[v][d]
                //   cnt[3:0] = v, parallel over all (t,d)
                // ----------------------------------------------------------
                CE_DX_INIT: begin
                    for (int ti = 0; ti < T; ti++) for (int di = 0; di < D; di++) begin
                        automatic shortreal dl  = dLogits_sr[ti][cnt[3:0]];
                        automatic shortreal ew  = $bitstoshortreal(emb_reg[cnt[3:0]][di]);
                        automatic shortreal acc = $bitstoshortreal(dX_init_reg[ti][di]);
                        dX_init_reg[ti][di] <= $shortrealtobits(acc + dl * ew);
                    end
                    if (cnt[3:0] == 4'(V-1)) begin
                        cnt <= '0; state <= CE_DWOUT;
                        // Initialise dWout_reg to 0 before accumulation
                        for (int vi = 0; vi < V; vi++)
                            for (int di = 0; di < D; di++)
                                dWout_reg[vi][di] <= '0;
                    end else cnt <= cnt + 1;
                end

                // ----------------------------------------------------------
                // Step 9 — dWout[v][d] = sum_t dLogits[t][v] * x[t][d]
                //   cnt[1:0] = t, parallel over all (v,d)
                // ----------------------------------------------------------
                CE_DWOUT: begin
                    for (int vi = 0; vi < V; vi++) for (int di = 0; di < D; di++) begin
                        automatic shortreal dl  = dLogits_sr[cnt[1:0]][vi];
                        automatic shortreal xv  = $bitstoshortreal(x_reg[cnt[1:0]][di]);
                        automatic shortreal acc = $bitstoshortreal(dWout_reg[vi][di]);
                        dWout_reg[vi][di] <= $shortrealtobits(acc + dl * xv);
                    end
                    if (cnt[1:0] == 2'(T-1)) begin
                        state <= DONE_ST;
                    end else cnt <= cnt + 1;
                end

                DONE_ST: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
