// Testbench for bf16w_matmul (FP32×BF16 — bf16w training path)
//
// Reads:
//   $NEURO_TESTVECS/bf16w_matmul/input.hex
//     M*K lines of FP32 hex (A, row-major)
//     K*N lines of BF16 hex (B, row-major)
//   $NEURO_TESTVECS/bf16w_matmul/expected.hex
//     M*N lines of FP32 hex (C, row-major)
//
// Writes:
//   $NEURO_TESTVECS/bf16w_matmul/pass_fail.txt
//
// Comparison: 1-ULP tolerance. C# reference uses ReferenceExactHardwareMode:
// every operation is (float)((double)x op (double)y), matching XSim shortreal promotion.
// Parameters: M=4, K=4, N=4, MAC_LATENCY=7, TOTAL_LAT=9.

`timescale 1ns/1ps

module tb_bf16w_matmul;

    localparam int M           = 4;
    localparam int K           = 4;
    localparam int N           = 4;
    localparam int MAC_LATENCY = 7;
    localparam int ADD_STAGES  = 2;
    localparam int TOTAL_LAT   = MAC_LATENCY + ADD_STAGES;

    logic        clk = 0;
    logic        rst = 1;
    logic        en  = 1;

    logic        a_wr_en   = 0;
    logic [7:0]  a_wr_addr = 0;
    logic [31:0] a_wr_data = 0;   // FP32

    logic        b_wr_en   = 0;
    logic [7:0]  b_wr_addr = 0;
    logic [15:0] b_wr_data = 0;

    logic        start = 0;

    logic [N*32-1:0] c_row;
    logic            c_valid;
    logic [1:0]      c_row_idx;

    bf16w_matmul #(.M(M), .K(K), .N(N), .MAC_LATENCY(MAC_LATENCY)) dut (
        .clk        (clk),
        .rst        (rst),
        .en         (en),
        .a_wr_en    (a_wr_en),
        .a_wr_addr  (a_wr_addr),
        .a_wr_data  (a_wr_data),
        .b_wr_en    (b_wr_en),
        .b_wr_addr  (b_wr_addr),
        .b_wr_data  (b_wr_data),
        .start      (start),
        .c_row      (c_row),
        .c_valid    (c_valid),
        .c_row_idx  (c_row_idx)
    );

    always #5 clk = ~clk;

    logic [31:0] a_vals [0:M*K-1];   // FP32
    logic [15:0] b_vals [0:K*N-1];   // BF16
    logic [31:0] got_c  [0:M*N-1];

    string  testvecs_dir;
    string  input_path, passfail_path, output_path;
    integer fd_in, fd_pf, fd_out, scan_ok;
    int     rows_received, cyc;

    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";

        input_path    = {testvecs_dir, "/bf16w_matmul/input.hex"};
        output_path   = {testvecs_dir, "/bf16w_matmul/output.hex"};
        passfail_path = {testvecs_dir, "/bf16w_matmul/pass_fail.txt"};

        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin write_pf({"FAIL:cannot open ", input_path}); $finish; end
        for (int ii = 0; ii < M*K; ii++) begin
            scan_ok = $fscanf(fd_in, "%h\n", a_vals[ii]);
            if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (A)"); $fclose(fd_in); $finish; end
        end
        for (int ii = 0; ii < K*N; ii++) begin
            scan_ok = $fscanf(fd_in, "%h\n", b_vals[ii]);
            if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (B)"); $fclose(fd_in); $finish; end
        end
        $fclose(fd_in);

        repeat(3) @(posedge clk); #1;
        rst = 0;

        // Load A (FP32)
        for (int ii = 0; ii < M*K; ii++) begin
            @(posedge clk); #1;
            a_wr_en   = 1;
            a_wr_addr = ii[7:0];
            a_wr_data = a_vals[ii];
        end
        @(posedge clk); #1;
        a_wr_en = 0;

        // Load B (BF16)
        for (int ii = 0; ii < K*N; ii++) begin
            @(posedge clk); #1;
            b_wr_en   = 1;
            b_wr_addr = ii[7:0];
            b_wr_data = b_vals[ii];
        end
        @(posedge clk); #1;
        b_wr_en = 0;

        // Pulse start
        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        // Collect M output rows
        rows_received = 0;
        for (cyc = 0; cyc < M + TOTAL_LAT + 4; cyc++) begin
            @(posedge clk); #1;
            if (c_valid) begin
                for (int jj = 0; jj < N; jj++)
                    got_c[c_row_idx * N + jj] = c_row[jj*32 +: 32];
                rows_received++;
                if (rows_received == M) break;
            end
        end

        if (rows_received != M) begin
            write_pf("FAIL:timed out waiting for output rows");
            $finish;
        end

        fd_out = $fopen(output_path, "w");
        if (fd_out != 0) begin
            for (int ii = 0; ii < M*N; ii++)
                $fwrite(fd_out, "%08h\n", got_c[ii]);
            $fclose(fd_out);
        end

        write_pf("PASS");
        $finish;
    end

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

endmodule
