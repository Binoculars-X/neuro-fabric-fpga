// BF16 Matrix Multiply: C[M×N] = A[M×K] · B[K×N]
//
// Algorithm:
//   For each output row i and column j:
//     C[i][j] = sum(k=0..K-1) A[i][k] * B[k][j]
//
//   Arithmetic: BF16 inputs decoded to FP32 (top 16 bits → zero-extend),
//   products accumulated in FP32 via an adder tree.
//   Matches C# reference using Bf16.Encode/Decode + float32 arithmetic.
//
// Architecture (K=4 specific, parameterisable M and N):
//   - K×N bf16_mac units in parallel (c_fp32=0 → pure product)
//   - 2-stage FP32 adder tree sums K=4 products per output column:
//       Stage 1: add_s1[j][0] = prod[0][j] + prod[1][j]
//                add_s1[j][1] = prod[2][j] + prod[3][j]
//       Stage 2: add_s2[j]    = add_s1[j][0] + add_s1[j][1]
//   - One row of C output per clock cycle, starting MAC_LATENCY+ADD_STAGES after start
//
// Matrix load protocol (before pulsing start):
//   Write A elements via a_wr_en/a_wr_addr/a_wr_data  (addr = row*K + col)
//   Write B elements via b_wr_en/b_wr_addr/b_wr_data  (addr = row*N + col)
//
// Output:
//   c_row[j*32 +: 32] = C[current_row][j]  (FP32, row-major, N per clock)
//   c_valid = 1 for M consecutive cycles starting TOTAL_LAT after start
//   c_row_idx = row index of current output (0..M-1)
//
// NOTE: ADD_STAGES = ceil(log2(K)) is hardcoded as 2 (valid for K=4 only).
//       Generalise adder tree with generate if K changes.
//
// Parameters:
//   M           — output rows    (default 4)
//   K           — inner dimension (must be 4 — adder tree hardcoded for K=4)
//   N           — output columns (default 4)
//   MAC_LATENCY — pipeline depth of bf16_mac (must match bf16_mac's LATENCY param)

`timescale 1ns/1ps

module bf16_matmul #(
    parameter int M           = 4,
    parameter int K           = 4,   // inner dimension; adder tree requires K=4
    parameter int N           = 4,
    parameter int MAC_LATENCY  = 7,   // must match bf16_mac LATENCY
    parameter int ADD_TREE_LAT = 8    // must match fp32_add_tree latency
)(
    input  logic clk,
    input  logic rst,
    input  logic en,

    // Load A (BF16, row-major): a_wr_addr = row*K + col
    input  logic        a_wr_en,
    input  logic [7:0]  a_wr_addr,
    input  logic [15:0] a_wr_data,

    // Load B (BF16, row-major): b_wr_addr = row*N + col
    input  logic        b_wr_en,
    input  logic [7:0]  b_wr_addr,
    input  logic [15:0] b_wr_data,

    // Pulse start once to begin computation
    input  logic start,

    // One row of C (FP32) per clock, valid for M consecutive cycles
    output logic [N*32-1:0] c_row,       // c_row[j*32 +: 32] = C[row][j]
    output logic            c_valid,
    output logic [1:0]      c_row_idx    // row index of c_row
);

    localparam int TOTAL_LAT = MAC_LATENCY + ADD_TREE_LAT;  // 15 for defaults

    // -----------------------------------------------------------------------
    // Register files: A[M][K] and B[K][N], BF16
    // -----------------------------------------------------------------------
    logic [15:0] A_reg [0:M-1][0:K-1];
    logic [15:0] B_reg [0:K-1][0:N-1];

    always_ff @(posedge clk) begin
        if (a_wr_en) A_reg[a_wr_addr / K][a_wr_addr % K] <= a_wr_data;
        if (b_wr_en) B_reg[b_wr_addr / N][b_wr_addr % N] <= b_wr_data;
    end

    // -----------------------------------------------------------------------
    // Row feed control
    //
    // start pulse: presents row 0 this cycle, then feeding=1 for rows 1..M-1.
    // cur_row mux: forces row 0 on the start cycle regardless of row_cnt state,
    // allowing safe re-use after a previous run (row_cnt would be M-1 otherwise).
    // -----------------------------------------------------------------------
    logic [1:0]  row_cnt;   // next row for feeding=1 cycles
    logic        feeding;

    always_ff @(posedge clk) begin
        if (rst) begin
            feeding <= 0;
            row_cnt <= 0;
        end else if (en) begin
            if (start) begin
                feeding <= 1;
                row_cnt <= 2'd1;   // row 0 is THIS cycle; row 1 is next
            end else if (feeding) begin
                if (row_cnt == M - 1)
                    feeding <= 0;
                else
                    row_cnt <= row_cnt + 2'd1;
            end
        end
    end

    logic [1:0]  cur_row;
    logic        cur_valid;
    always_comb cur_row   = start ? 2'd0 : row_cnt;
    assign       cur_valid = start | feeding;

    // -----------------------------------------------------------------------
    // MAC array: K×N bf16_mac instances
    // mac_prod[k][j] = A[cur_row][k] * B[k][j]   (c_fp32=0: pure multiply)
    // All instances share the same timing (valid_out follows valid_in by MAC_LATENCY)
    // -----------------------------------------------------------------------
    logic [31:0] mac_prod  [0:K-1][0:N-1];
    logic        mac_valid [0:K-1][0:N-1];

    genvar gk, gn;
    generate
        for (gk = 0; gk < K; gk++) begin : gen_k
            for (gn = 0; gn < N; gn++) begin : gen_n
                bf16_mac #(.LATENCY(MAC_LATENCY)) u_mac (
                    .clk        (clk),
                    .rst        (rst),
                    .en         (en),
                    .a_bf16     (A_reg[cur_row][gk]),
                    .b_bf16     (B_reg[gk][gn]),
                    .c_fp32     (32'h0000_0000),
                    .valid_in   (cur_valid),
                    .result_fp32(mac_prod[gk][gn]),
                    .result_bf16(),                  // unused
                    .valid_out  (mac_valid[gk][gn])
                );
            end
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Adder tree: N fp32_add_tree instances (one per output column)
    // Each sums K=4 products: mac_prod[0..3][j]
    // fp32_add_tree latency = 8 cycles
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
    // Row-index pipeline: shift TOTAL_LAT stages so c_row_idx matches c_row
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
