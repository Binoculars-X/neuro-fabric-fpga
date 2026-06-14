// FP32×FP32 Matrix Multiply: C[M×N] = A[M×K] · B[K×N]
//
// Both operands are FP32 activations (no weight decoding).
// Used in the attention forward pass for:
//   - Score matrix:    Q[T×d] × Kᵀ[d×T]  → scores[T×T]
//   - Weighted sum:    attn[T×T] × V[T×d] → out[T×d]
//
// Architecture: identical pipeline structure to bf16w_matmul.
//   K×N shortreal multiply units (pure product, c=0)
//   2-stage FP32 adder tree (K=4): (p0+p1) + (p2+p3)
//   Total latency = MUL_LATENCY + 2 cycles
//
// Matrix load protocol (same as bf16w_matmul):
//   A: a_wr_en/a_wr_addr/a_wr_data  — FP32, addr = row*K + col
//   B: b_wr_en/b_wr_addr/b_wr_data  — FP32, addr = row*N + col
//
// Output:
//   c_row[j*32 +: 32] = C[cur_row][j]  (FP32, N values per clock)
//   c_valid asserted for M consecutive clocks (one row per clock)
//   c_row_idx gives the 0-based output row index

`timescale 1ns/1ps

module fp32_matmul #(
    parameter int M           = 4,
    parameter int K           = 4,
    parameter int N           = 4,
    parameter int MUL_LATENCY = 3    // pipeline depth of the multiply stage
)(
    input  logic clk,
    input  logic rst,
    input  logic en,

    // Load A (FP32, row-major)
    input  logic        a_wr_en,
    input  logic [7:0]  a_wr_addr,
    input  logic [31:0] a_wr_data,

    // Load B (FP32, row-major)
    input  logic        b_wr_en,
    input  logic [7:0]  b_wr_addr,
    input  logic [31:0] b_wr_data,

    input  logic start,

    output logic [N*32-1:0] c_row,
    output logic            c_valid,
    output logic [1:0]      c_row_idx
);

    localparam int ADD_STAGES = 2;
    localparam int TOTAL_LAT  = MUL_LATENCY + ADD_STAGES;

    // -----------------------------------------------------------------------
    // Register files
    // -----------------------------------------------------------------------
    logic [31:0] A_reg [0:M-1][0:K-1];
    logic [31:0] B_reg [0:K-1][0:N-1];

    always_ff @(posedge clk) begin
        if (a_wr_en) A_reg[a_wr_addr / K][a_wr_addr % K] <= a_wr_data;
        if (b_wr_en) B_reg[b_wr_addr / N][b_wr_addr % N] <= b_wr_data;
    end

    // -----------------------------------------------------------------------
    // Row feed control
    // -----------------------------------------------------------------------
    logic [1:0] row_cnt;
    logic       feeding;

    always_ff @(posedge clk) begin
        if (rst) begin
            feeding <= 0;
            row_cnt <= 0;
        end else if (en) begin
            if (start) begin
                feeding <= 1;
                row_cnt <= 2'd1;
            end else if (feeding) begin
                if (row_cnt == M - 1) feeding <= 0;
                else row_cnt <= row_cnt + 2'd1;
            end
        end
    end

    logic [1:0] cur_row;
    logic       cur_valid;
    always_comb cur_row   = start ? 2'd0 : row_cnt;
    assign      cur_valid = start | feeding;

    // -----------------------------------------------------------------------
    // Multiply array: K×N shortreal multiply units (pipelined)
    // -----------------------------------------------------------------------
    // Each unit: p[k][n] = A[cur_row][k] * B[k][n], latched MUL_LATENCY times.
    logic [31:0] mul_prod  [0:K-1][0:N-1];
    logic        mul_valid [0:K-1][0:N-1];

    genvar gk, gn;
    generate
        for (gk = 0; gk < K; gk++) begin : gen_k
            for (gn = 0; gn < N; gn++) begin : gen_n

                // Pipeline registers for product and valid
                logic [31:0] prod_pipe [0:MUL_LATENCY-1];
                logic        vld_pipe  [0:MUL_LATENCY-1];

                always_ff @(posedge clk) begin
                    if (rst) begin
                        for (int s = 0; s < MUL_LATENCY; s++) begin
                            prod_pipe[s] <= 32'h0;
                            vld_pipe[s]  <= 1'b0;
                        end
                    end else if (en) begin
                        begin
                            shortreal fa, fb, fp;
                            fa = $bitstoshortreal(A_reg[cur_row][gk]);
                            fb = $bitstoshortreal(B_reg[gk][gn]);
                            fp = fa * fb;
                            prod_pipe[0] <= $shortrealtobits(fp);
                        end
                        vld_pipe[0] <= cur_valid;
                        for (int s = 1; s < MUL_LATENCY; s++) begin
                            prod_pipe[s] <= prod_pipe[s-1];
                            vld_pipe[s]  <= vld_pipe[s-1];
                        end
                    end
                end

                assign mul_prod[gk][gn]  = prod_pipe[MUL_LATENCY-1];
                assign mul_valid[gk][gn] = vld_pipe[MUL_LATENCY-1];
            end
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Adder tree (K=4, 2 stages): (p0+p1) + (p2+p3)
    // -----------------------------------------------------------------------
    logic [31:0] add1 [0:N-1][0:1];
    logic        add1_v;
    logic [31:0] add2 [0:N-1];
    logic        add2_v;

    shortreal tmp_p, tmp_q, tmp_r;

    always_ff @(posedge clk) begin
        if (rst) begin
            add1_v <= 0;
            add2_v <= 0;
        end else if (en) begin
            add1_v <= mul_valid[0][0];
            for (int jj = 0; jj < N; jj++) begin
                tmp_p = $bitstoshortreal(mul_prod[0][jj]);
                tmp_q = $bitstoshortreal(mul_prod[1][jj]);
                tmp_r = tmp_p + tmp_q;
                add1[jj][0] <= $shortrealtobits(tmp_r);

                tmp_p = $bitstoshortreal(mul_prod[2][jj]);
                tmp_q = $bitstoshortreal(mul_prod[3][jj]);
                tmp_r = tmp_p + tmp_q;
                add1[jj][1] <= $shortrealtobits(tmp_r);
            end

            add2_v <= add1_v;
            for (int jj = 0; jj < N; jj++) begin
                tmp_p = $bitstoshortreal(add1[jj][0]);
                tmp_q = $bitstoshortreal(add1[jj][1]);
                tmp_r = tmp_p + tmp_q;
                add2[jj] <= $shortrealtobits(tmp_r);
            end
        end
    end

    // -----------------------------------------------------------------------
    // Row index pipeline
    // -----------------------------------------------------------------------
    logic [1:0] row_pipe [0:TOTAL_LAT-1];

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int ii = 0; ii < TOTAL_LAT; ii++) row_pipe[ii] <= 0;
        end else if (en) begin
            row_pipe[0] <= cur_row;
            for (int ii = 1; ii < TOTAL_LAT; ii++)
                row_pipe[ii] <= row_pipe[ii-1];
        end
    end

    // -----------------------------------------------------------------------
    // Outputs
    // -----------------------------------------------------------------------
    always_comb begin
        for (int jj = 0; jj < N; jj++)
            c_row[jj*32 +: 32] = add2[jj];
    end

    assign c_valid   = add2_v;
    assign c_row_idx = row_pipe[TOTAL_LAT-1];

endmodule
