// Testbench for bf16_matmul
//
// Reads test vectors from:
//   $NEURO_TESTVECS/bf16_matmul/input.hex    — M*K + K*N lines of BF16 hex (A then B)
//   $NEURO_TESTVECS/bf16_matmul/expected.hex — M*N lines of FP32 hex (C, row-major)
//
// Writes:
//   $NEURO_TESTVECS/bf16_matmul/pass_fail.txt
//
// Comparison: 16-ULP tolerance (matches exp_lut reasoning — XSim promotes
// shortreal to double internally; same algorithm as C# reference).
//
// Parameters match bf16_matmul defaults: M=4, K=4, N=4, MAC_LATENCY=3, ADD_STAGES=2.
// TOTAL_LAT = 5 cycles from start to first valid output row.

`timescale 1ns/1ps

module tb_bf16_matmul;

    localparam int M           = 4;
    localparam int K           = 4;
    localparam int N           = 4;
    localparam int MAC_LATENCY = 3;
    localparam int ADD_STAGES  = 2;
    localparam int TOTAL_LAT   = MAC_LATENCY + ADD_STAGES;

    // ------------------------------------------------------------------
    // DUT signals
    // ------------------------------------------------------------------
    logic        clk     = 0;
    logic        rst     = 1;
    logic        en      = 1;

    logic        a_wr_en   = 0;
    logic [7:0]  a_wr_addr = 0;
    logic [15:0] a_wr_data = 0;

    logic        b_wr_en   = 0;
    logic [7:0]  b_wr_addr = 0;
    logic [15:0] b_wr_data = 0;

    logic        start = 0;

    logic [N*32-1:0] c_row;
    logic            c_valid;
    logic [1:0]      c_row_idx;

    bf16_matmul #(.M(M), .K(K), .N(N), .MAC_LATENCY(MAC_LATENCY)) dut (
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

    // ------------------------------------------------------------------
    // Storage
    // ------------------------------------------------------------------
    logic [15:0] a_vals [0:M*K-1];
    logic [15:0] b_vals [0:K*N-1];
    logic [31:0] exp_c  [0:M*N-1];
    logic [31:0] got_c  [0:M*N-1];

    // ------------------------------------------------------------------
    // File / checker state (module scope — avoids 'automatic' in loops)
    // ------------------------------------------------------------------
    string  testvecs_dir;
    string  input_path, expected_path, passfail_path;
    integer fd_in, fd_exp, fd_pf, scan_ok;
    int     fail_count;
    string  fail_detail;
    int     rows_received;
    int     ulp_diff;
    int     cyc;

    // ------------------------------------------------------------------
    // Main
    // ------------------------------------------------------------------
    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";

        input_path    = {testvecs_dir, "/bf16_matmul/input.hex"};
        expected_path = {testvecs_dir, "/bf16_matmul/expected.hex"};
        passfail_path = {testvecs_dir, "/bf16_matmul/pass_fail.txt"};

        // Read A
        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin write_pf({"FAIL:cannot open ", input_path}); $finish; end
        for (int ii = 0; ii < M*K; ii++) begin
            scan_ok = $fscanf(fd_in, "%h\n", a_vals[ii]);
            if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (A)"); $fclose(fd_in); $finish; end
        end
        // Read B (same file, continues from where A ended)
        for (int ii = 0; ii < K*N; ii++) begin
            scan_ok = $fscanf(fd_in, "%h\n", b_vals[ii]);
            if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (B)"); $fclose(fd_in); $finish; end
        end
        $fclose(fd_in);

        // Read expected C
        fd_exp = $fopen(expected_path, "r");
        if (fd_exp == 0) begin write_pf({"FAIL:cannot open ", expected_path}); $finish; end
        for (int ii = 0; ii < M*N; ii++) begin
            scan_ok = $fscanf(fd_exp, "%h\n", exp_c[ii]);
            if (scan_ok != 1) begin write_pf("FAIL:expected.hex malformed"); $fclose(fd_exp); $finish; end
        end
        $fclose(fd_exp);

        // Reset
        repeat(3) @(posedge clk); #1;
        rst = 0;

        // Load A element by element
        for (int ii = 0; ii < M*K; ii++) begin
            @(posedge clk); #1;
            a_wr_en   = 1;
            a_wr_addr = ii[7:0];
            a_wr_data = a_vals[ii];
        end
        @(posedge clk); #1;
        a_wr_en = 0;

        // Load B element by element
        for (int ii = 0; ii < K*N; ii++) begin
            @(posedge clk); #1;
            b_wr_en   = 1;
            b_wr_addr = ii[7:0];
            b_wr_data = b_vals[ii];
        end
        @(posedge clk); #1;
        b_wr_en = 0;

        // Pulse start for one cycle
        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        // Collect M output rows (timeout after M + TOTAL_LAT + 4 cycles)
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

        // Check: 16 ULP tolerance (XSim shortreal→double promotion vs C# float32)
        fail_count  = 0;
        fail_detail = "";
        for (int ii = 0; ii < M*N; ii++) begin
            ulp_diff = int'(got_c[ii]) - int'(exp_c[ii]);
            if (ulp_diff < 0) ulp_diff = -ulp_diff;
            if (ulp_diff > 16) begin
                fail_count++;
                if (fail_count == 1)
                    $sformat(fail_detail, "C[%0d] got %08h exp %08h (diff %0d ULP)",
                        ii, got_c[ii], exp_c[ii], ulp_diff);
                $display("MISMATCH C[%0d] got %08h exp %08h (diff %0d ULP)",
                    ii, got_c[ii], exp_c[ii], ulp_diff);
            end
        end

        if (fail_count == 0) begin
            $display("PASS — all %0d C elements within 16 ULP", M*N);
            write_pf("PASS");
        end else begin
            $display("FAIL — %0d mismatches; first: %s", fail_count, fail_detail);
            write_pf({"FAIL:", fail_detail});
        end

        $finish;
    end

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

endmodule
