// Testbench for bf16_mac
//
// Reads test vectors from the file pointed to by NEURO_TESTVECS env var
// (set by C# FpgaXSimRunner before invoking xsim):
//
//   $NEURO_TESTVECS/bf16_mac/input.hex    — lines: <a_bf16_hex> <b_bf16_hex> <c_fp32_hex>
//   $NEURO_TESTVECS/bf16_mac/expected.hex — lines: <result_fp32_hex> <result_bf16_hex>
//
// Writes pass/fail result to:
//   $NEURO_TESTVECS/bf16_mac/pass_fail.txt  — single line: PASS or FAIL:<detail>
//
// The testbench drives valid_in for each vector, waits LATENCY cycles, then
// compares outputs.  Any mismatch is reported immediately and the file is
// written FAIL.  All vectors pass → PASS.

`timescale 1ns/1ps

module tb_bf16_mac;

    // ------------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------------
    localparam int LATENCY    = 3;
    localparam int N_VECTORS  = 16;   // C# writes exactly this many per run

    // ------------------------------------------------------------------
    // DUT signals
    // ------------------------------------------------------------------
    logic        clk      = 0;
    logic        rst      = 1;
    logic        en       = 1;
    logic [15:0] a_bf16;
    logic [15:0] b_bf16;
    logic [31:0] c_fp32;
    logic        valid_in = 0;
    logic [31:0] result_fp32;
    logic [15:0] result_bf16;
    logic        valid_out;

    bf16_mac #(.LATENCY(LATENCY)) dut (
        .clk        (clk),
        .rst        (rst),
        .en         (en),
        .a_bf16     (a_bf16),
        .b_bf16     (b_bf16),
        .c_fp32     (c_fp32),
        .valid_in   (valid_in),
        .result_fp32(result_fp32),
        .result_bf16(result_bf16),
        .valid_out  (valid_out)
    );

    // 10 ns clock
    always #5 clk = ~clk;

    // ------------------------------------------------------------------
    // Storage for vectors
    // ------------------------------------------------------------------
    logic [15:0] in_a   [0:N_VECTORS-1];
    logic [15:0] in_b   [0:N_VECTORS-1];
    logic [31:0] in_c   [0:N_VECTORS-1];
    logic [31:0] exp_fp32 [0:N_VECTORS-1];
    logic [15:0] exp_bf16 [0:N_VECTORS-1];
    logic [31:0] got_fp32 [0:N_VECTORS-1];

    // ------------------------------------------------------------------
    // File handles and paths
    // ------------------------------------------------------------------
    string testvecs_dir;
    string input_path;
    string expected_path;
    string passfail_path;
    string output_path;

    integer fd_in, fd_exp, fd_pf, fd_out;
    integer scan_ok;
    int     fail_count;
    string  fail_detail;

    // ------------------------------------------------------------------
    // Main test sequence
    // ------------------------------------------------------------------
    initial begin
        // Resolve paths from env var; fall back to relative path for manual runs
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";

        input_path    = {testvecs_dir, "/bf16_mac/input.hex"};
        expected_path = {testvecs_dir, "/bf16_mac/expected.hex"};
        passfail_path = {testvecs_dir, "/bf16_mac/pass_fail.txt"};
        output_path   = {testvecs_dir, "/bf16_mac/output.hex"};

        // Read input vectors
        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin
            $display("ERROR: cannot open %s", input_path);
            write_passfail({"FAIL:cannot open ", input_path});
            $finish;
        end
        for (int i = 0; i < N_VECTORS; i++) begin
            scan_ok = $fscanf(fd_in, "%h %h %h\n", in_a[i], in_b[i], in_c[i]);
            if (scan_ok != 3) begin
                $display("ERROR: input.hex line %0d malformed", i);
                write_passfail("FAIL:input.hex malformed");
                $fclose(fd_in);
                $finish;
            end
        end
        $fclose(fd_in);

        // Read expected outputs
        fd_exp = $fopen(expected_path, "r");
        if (fd_exp == 0) begin
            $display("ERROR: cannot open %s", expected_path);
            write_passfail({"FAIL:cannot open ", expected_path});
            $finish;
        end
        for (int i = 0; i < N_VECTORS; i++) begin
            scan_ok = $fscanf(fd_exp, "%h %h\n", exp_fp32[i], exp_bf16[i]);
            if (scan_ok != 2) begin
                $display("ERROR: expected.hex line %0d malformed", i);
                write_passfail("FAIL:expected.hex malformed");
                $fclose(fd_exp);
                $finish;
            end
        end
        $fclose(fd_exp);

        // Reset
        @(posedge clk); #1;
        rst = 1;
        repeat(3) @(posedge clk);
        #1; rst = 0;

        fail_count  = 0;
        fail_detail = "";

        // Drive inputs and collect outputs in one interleaved loop.
        // Pipeline LATENCY means valid_out appears LATENCY cycles after valid_in.
        // We run for N_VECTORS + LATENCY + 2 cycles to flush the pipeline.
        begin : main_loop
            automatic int in_idx  = 0;
            automatic int out_idx = 0;
            automatic int total_cycles = N_VECTORS + LATENCY + 2;

            for (int cyc = 0; cyc < total_cycles; cyc++) begin
                @(posedge clk); #1;

                // Capture output BEFORE driving new input (combinational output settled)
                if (valid_out && out_idx < N_VECTORS) begin
                    got_fp32[out_idx] = result_fp32;
                    if (result_fp32 !== exp_fp32[out_idx]) begin
                        fail_count++;
                        if (fail_count == 1)
                            $sformat(fail_detail,
                                "vec[%0d] fp32: got %08h exp %08h",
                                out_idx, result_fp32, exp_fp32[out_idx]);
                        $display("MISMATCH vec[%0d] fp32: got %08h exp %08h",
                            out_idx, result_fp32, exp_fp32[out_idx]);
                    end
                    if (result_bf16 !== exp_bf16[out_idx]) begin
                        fail_count++;
                        if (fail_count == 1)
                            $sformat(fail_detail,
                                "vec[%0d] bf16: got %04h exp %04h",
                                out_idx, result_bf16, exp_bf16[out_idx]);
                        $display("MISMATCH vec[%0d] bf16: got %04h exp %04h",
                            out_idx, result_bf16, exp_bf16[out_idx]);
                    end
                    out_idx++;
                end

                // Drive next input
                if (in_idx < N_VECTORS) begin
                    a_bf16   = in_a[in_idx];
                    b_bf16   = in_b[in_idx];
                    c_fp32   = in_c[in_idx];
                    valid_in = 1;
                    in_idx++;
                end else begin
                    valid_in = 0;
                end

                if (out_idx == N_VECTORS) break;
            end
        end

        if (fail_count == 0) begin
            $display("PASS — all %0d vectors matched", N_VECTORS);
            write_passfail("PASS");
        end else begin
            $display("FAIL — %0d mismatches; first: %s", fail_count, fail_detail);
            write_passfail({"FAIL:", fail_detail});
        end

        // Write output.hex for VsSoftware comparison
        fd_out = $fopen(output_path, "w");
        if (fd_out != 0) begin
            for (int i = 0; i < N_VECTORS; i++)
                $fwrite(fd_out, "%08h\n", got_fp32[i]);
            $fclose(fd_out);
        end

        $finish;
    end

    // ------------------------------------------------------------------
    // Helper: write pass/fail file
    // ------------------------------------------------------------------
    task automatic write_passfail(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin
            $fwrite(fd_pf, "%s\n", msg);
            $fclose(fd_pf);
        end else begin
            $display("WARNING: cannot write %s", passfail_path);
        end
    endtask

endmodule
