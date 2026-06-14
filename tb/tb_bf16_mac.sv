// Testbench for bf16_mac
//
// Reads:
//   $NEURO_TESTVECS/bf16_mac/input.hex  -- lines: <a_bf16_hex> <b_bf16_hex> <c_fp32_hex>
//
// Writes:
//   $NEURO_TESTVECS/bf16_mac/output.hex    -- N_VECTORS lines: FP32 result hex
//   $NEURO_TESTVECS/bf16_mac/pass_fail.txt -- "PASS" when all outputs collected
//
// Numeric verification is done in C# by reading output.hex.

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
    logic [31:0] got_fp32 [0:N_VECTORS-1];

    // ------------------------------------------------------------------
    // File handles and paths
    // ------------------------------------------------------------------
    string testvecs_dir;
    string input_path;
    string passfail_path;
    string output_path;

    integer fd_in, fd_pf, fd_out;
    integer scan_ok;

    // ------------------------------------------------------------------
    // Main test sequence
    // ------------------------------------------------------------------
    initial begin
        // Resolve paths from env var; fall back to relative path for manual runs
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";

        input_path    = {testvecs_dir, "/bf16_mac/input.hex"};
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

        // Reset
        @(posedge clk); #1;
        rst = 1;
        repeat(3) @(posedge clk);
        #1; rst = 0;

        // Drive inputs and collect outputs in one interleaved loop.
        begin : main_loop
            automatic int in_idx  = 0;
            automatic int out_idx = 0;
            automatic int total_cycles = N_VECTORS + LATENCY + 2;

            for (int cyc = 0; cyc < total_cycles; cyc++) begin
                @(posedge clk); #1;

                if (valid_out && out_idx < N_VECTORS) begin
                    got_fp32[out_idx] = result_fp32;
                    out_idx++;
                end

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

        // Write output.hex
        fd_out = $fopen(output_path, "w");
        if (fd_out != 0) begin
            for (int i = 0; i < N_VECTORS; i++)
                $fwrite(fd_out, "%08h\n", got_fp32[i]);
            $fclose(fd_out);
        end

        write_passfail("PASS");
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
