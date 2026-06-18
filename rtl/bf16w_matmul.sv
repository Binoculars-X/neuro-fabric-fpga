// FP32×BF16 Matrix Multiply: C[M×N] = A[M×K] · B[K×N]
//
// bf16w training path: A = FP32 activations, B = BF16 weights (decoded on-the-fly).
// Matches AdamBF16WeightsAttentionCore.Forward():
//   MatMul(x_fp32, Bf16.Decode(w_bf16[i,j]))
//
// Architecture: identical to bf16_matmul but uses bf16w_mac internally.
//   K×N bf16w_mac units in parallel (c_fp32=0: pure product)
//   2-stage FP32 adder tree (K=4): (p0+p1) + (p2+p3)
//   Total latency = MAC_LATENCY + ADD_TREE_LAT = 7 + 8 = 15 cycles (defaults)
//
// Matrix load protocol:
//   A: a_wr_en/a_wr_addr/a_wr_data  — FP32, addr = row*K + col
//   B: b_wr_en/b_wr_addr/b_wr_data  — BF16, addr = row*N + col
//
// Output:
//   c_row[j*32 +: 32] = C[row][j]  (FP32, N values per clock)
//   c_valid, c_row_idx as in bf16_matmul

`timescale 1ns/1ps

module bf16w_matmul #(
    parameter int M           = 4,
    parameter int K           = 4,
    parameter int N           = 4,
    parameter int MAC_LATENCY  = 7,
    parameter int ADD_TREE_LAT = 8
)(
    input  logic clk,
    input  logic rst,
    input  logic en,

    // Load A (FP32, row-major)
    input  logic        a_wr_en,
    input  logic [7:0]  a_wr_addr,
    input  logic [31:0] a_wr_data,   // FP32 — wider than bf16_matmul

    // Load B (BF16, row-major)
    input  logic        b_wr_en,
    input  logic [7:0]  b_wr_addr,
    input  logic [15:0] b_wr_data,

    input  logic start,

    output logic [N*32-1:0] c_row,
    output logic            c_valid,
    output logic [1:0]      c_row_idx
);

    localparam int TOTAL_LAT  = MAC_LATENCY + ADD_TREE_LAT;

    // -----------------------------------------------------------------------
    // Register files
    // -----------------------------------------------------------------------
    logic [31:0] A_reg [0:M-1][0:K-1];   // FP32
    logic [15:0] B_reg [0:K-1][0:N-1];   // BF16

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
    // MAC array: K×N bf16w_mac instances
    // -----------------------------------------------------------------------
    logic [31:0] mac_prod  [0:K-1][0:N-1];
    logic        mac_valid [0:K-1][0:N-1];

    genvar gk, gn;
    generate
        for (gk = 0; gk < K; gk++) begin : gen_k
            for (gn = 0; gn < N; gn++) begin : gen_n
                bf16w_mac #(.LATENCY(MAC_LATENCY)) u_mac (
                    .clk        (clk),
                    .rst        (rst),
                    .en         (en),
                    .a_fp32     (A_reg[cur_row][gk]),
                    .b_bf16     (B_reg[gk][gn]),
                    .c_fp32     (32'h0000_0000),
                    .valid_in   (cur_valid),
                    .result_fp32(mac_prod[gk][gn]),
                    .valid_out  (mac_valid[gk][gn])
                );
            end
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Adder tree: N fp32_add_tree instances (one per output column)
    // Each sums K=4 products: mac_prod[0..3][j]; latency = 8 cycles
    // -----------------------------------------------------------------------
    logic [31:0] tree_sum   [0:N-1];
    logic        tree_valid [0:N-1];

    genvar gt;
    generate
        for (gt = 0; gt < N; gt++) begin : gen_tree
            fp32_add_tree #(.T(4)) u_tree (
                .clk      (clk),
                .rst      (rst),
                .valid_in (mac_valid[0][gt]),
                .in_vec   ({mac_prod[3][gt], mac_prod[2][gt],
                            mac_prod[1][gt], mac_prod[0][gt]}),
                .sum_out  (tree_sum[gt]),
                .valid_out(tree_valid[gt])
            );
        end
    endgenerate

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
    // Output
    // -----------------------------------------------------------------------
    always_comb begin
        c_valid   = tree_valid[0];
        c_row_idx = row_pipe[TOTAL_LAT-1];
        for (int jj = 0; jj < N; jj++)
            c_row[jj*32 +: 32] = tree_sum[jj];
    end

endmodule
