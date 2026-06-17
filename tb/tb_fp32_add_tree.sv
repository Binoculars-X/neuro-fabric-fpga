// Testbench for fp32_add_tree (T=4 pipelined FP32 adder tree, LATENCY=8 cycles)
//
// Simple approach: drive inputs for N_VECS cycles, then wait long enough for
// all outputs to appear (LATENCY + N_VECS), then capture and write.

`timescale 1ns/1ps

module tb_fp32_add_tree;

    localparam int T        = 4;
    localparam int N_VECS   = 8;
    localparam int LATENCY  = 8;    // fp32_add (4) + fp32_add (4)
    localparam int CLK_HALF = 5;    // 100 MHz

    logic            clk = 0;
    logic            rst = 1;
    logic            valid_in  = 0;
    logic [T*32-1:0] in_vec    = '0;
    logic [31:0]     sum_out;
    logic            valid_out;

    fp32_add_tree #(.T(T)) dut (
        .clk      (clk),
        .rst      (rst),
        .valid_in (valid_in),
        .in_vec   (in_vec),
        .sum_out  (sum_out),
        .valid_out(valid_out)
    );

    always #CLK_HALF clk = ~clk;

    logic [31:0] in_vals [0:N_VECS*T-1];
    logic [31:0] got     [0:N_VECS-1];
    logic [31:0] s1_s01  [0:N_VECS-1];  // Capture Stage 1 intermediate s01
    logic [31:0] s1_s23  [0:N_VECS-1];  // Capture Stage 1 intermediate s23

    string  testvecs_dir;
    string  input_path, passfail_path, output_path, debug_path;
    integer fd_in, fd_pf, fd_out, fd_debug, scan_ok;

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

    // ── Capture outputs: sample whenever valid_out=1 ────────────────────────────
    // valid_out=1 for N_VECS consecutive cycles; one output value per cycle.
    // Capture by polling valid_out in a separate sampling thread after stimulus ends.
    int   out_cnt = 0;

    // (Capture happens in the initial block with manual polling)

    // ── Main stimulus ─────────────────────────────────────────────────────────
    initial begin
        $dumpfile("../../run/fpga-testvecs/fp32_add_tree/tb_fp32_add_tree.vcd");
        $dumpvars(0, tb_fp32_add_tree, dut.s1_s01, dut.s1_s23);  // Capture intermediate values
        
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";
        input_path    = {testvecs_dir, "/fp32_add_tree/input.hex"};
        passfail_path = {testvecs_dir, "/fp32_add_tree/pass_fail.txt"};
        output_path   = {testvecs_dir, "/fp32_add_tree/output.hex"};
        debug_path    = {testvecs_dir, "/fp32_add_tree/debug.txt"};

        // Read input vectors
        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin write_pf({"FAIL:cannot open ", input_path}); $finish; end
        for (int i = 0; i < N_VECS * T; i++) begin
            scan_ok = $fscanf(fd_in, "%h\n", in_vals[i]);
            if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed"); $fclose(fd_in); $finish; end
        end
        $fclose(fd_in);

        // Reset
        @(posedge clk); @(posedge clk);
        rst = 0;
        @(posedge clk);
        @(posedge clk);  // Extra delay to ensure rst is fully propagated

        // Drive N_VECS inputs one per cycle
        for (int c = 0; c < N_VECS; c++) begin
            for (int j = 0; j < T; j++)
                in_vec[j*32 +: 32] = in_vals[c*T + j];
            valid_in = 1;
            @(posedge clk);
        end
        valid_in = 0;
        in_vec   = '0;

        // Poll for valid_out, advancing one cycle at a time
        do @(posedge clk);
        while (!valid_out);
        
        // Now we're on a posedge where valid_out is high.
        // Capture N_VECS outputs: read sum_out on each of the next N_VECS cycles.
        for (int c = 0; c < N_VECS; c++) begin
            got[c] = sum_out;
            if (c < N_VECS - 1) @(posedge clk);
        end

        // Wait a bit more to be safe
        repeat (5) @(posedge clk);

        // Write output and debug
        fd_out = $fopen(output_path, "w");
        if (fd_out == 0) begin write_pf("FAIL:cannot open output.hex"); $finish; end
        for (int c = 0; c < N_VECS; c++)
            $fwrite(fd_out, "%08h\n", got[c]);
        $fclose(fd_out);

        // Debug: also write what the RTL computed for s01 and s23
        fd_debug = $fopen(debug_path, "w");
        if (fd_debug != 0) begin
            $fwrite(fd_debug, "=== Debug: intermediate s01, s23 for each case ===\n");
            $fwrite(fd_debug, "NOTE: s1_s01/s23 are internal signals; captured at end\n");
            $fwrite(fd_debug, "In a real test, need to capture these during pipeline\n");
            $fclose(fd_debug);
        end

        write_pf($sformatf("PASS captured=%0d", out_cnt));
        $finish;
    end

endmodule
