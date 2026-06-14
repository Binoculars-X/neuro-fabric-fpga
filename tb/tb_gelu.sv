// Testbench for gelu
//
// Reads:
//   $NEURO_TESTVECS/gelu/input.hex  — N_VECTORS lines of FP32 hex (x inputs)
//
// Writes:
//   $NEURO_TESTVECS/gelu/output.hex    — N_VECTORS lines: actual RTL GeLU results
//   $NEURO_TESTVECS/gelu/pass_fail.txt — "PASS" when all outputs collected
//
// Numeric verification is done in C# (AttentionLayer.Gelu reference, ReferenceExactHardwareMode).
//
// gelu.sv latency = EXP_LAT + 2 = 6 cycles.

`timescale 1ns/1ps

module tb_gelu;

    localparam int LATENCY   = 6;   // EXP_LAT(4) + 2
    localparam int N_VECTORS = 32;
    localparam int LUT_SIZE  = 256;

    // ------------------------------------------------------------------
    // DUT signals
    // ------------------------------------------------------------------
    logic        clk = 0;
    logic        rst = 1;
    logic        en  = 1;
    logic [31:0] x_fp32     = 0;
    logic        valid_in   = 0;
    logic [31:0] result_fp32;
    logic        valid_out;

    gelu #(
        .EXP_LAT (4),
        .LUT_SIZE(LUT_SIZE),
        .LUT_FILE("exp_lut_init.hex")
    ) dut (
        .clk        (clk),
        .rst        (rst),
        .en         (en),
        .x_fp32     (x_fp32),
        .valid_in   (valid_in),
        .result_fp32(result_fp32),
        .valid_out  (valid_out)
    );

    always #5 clk = ~clk;

    // ------------------------------------------------------------------
    // Vector storage
    // ------------------------------------------------------------------
    logic [31:0] in_x   [0:N_VECTORS-1];
    logic [31:0] got_out[0:N_VECTORS-1];

    // ------------------------------------------------------------------
    // File / loop state
    // ------------------------------------------------------------------
    string  testvecs_dir;
    string  input_path, output_path, passfail_path;
    integer fd_in, fd_out, fd_pf, scan_ok;
    int     in_idx, out_idx, cyc, total_cycles;

    // ------------------------------------------------------------------
    // Main
    // ------------------------------------------------------------------
    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";

        input_path    = {testvecs_dir, "/gelu/input.hex"};
        output_path   = {testvecs_dir, "/gelu/output.hex"};
        passfail_path = {testvecs_dir, "/gelu/pass_fail.txt"};

        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin
            write_passfail({"FAIL:cannot open ", input_path});
            $finish;
        end
        for (int i = 0; i < N_VECTORS; i++) begin
            scan_ok = $fscanf(fd_in, "%h\n", in_x[i]);
            if (scan_ok != 1) begin
                write_passfail("FAIL:input.hex malformed");
                $fclose(fd_in); $finish;
            end
        end
        $fclose(fd_in);

        // Reset
        repeat(3) @(posedge clk);
        #1; rst = 0;

        in_idx       = 0;
        out_idx      = 0;
        total_cycles = N_VECTORS + LATENCY + 4;

        for (cyc = 0; cyc < total_cycles; cyc++) begin
            @(posedge clk); #1;

            if (valid_out && out_idx < N_VECTORS) begin
                got_out[out_idx] = result_fp32;
                out_idx++;
            end

            if (in_idx < N_VECTORS) begin
                x_fp32   = in_x[in_idx];
                valid_in = 1;
                in_idx++;
            end else begin
                valid_in = 0;
            end

            if (out_idx == N_VECTORS) break;
        end

        if (out_idx != N_VECTORS) begin
            write_passfail("FAIL:timed out before all outputs collected");
            $finish;
        end

        fd_out = $fopen(output_path, "w");
        if (fd_out != 0) begin
            for (int i = 0; i < N_VECTORS; i++)
                $fwrite(fd_out, "%08h\n", got_out[i]);
            $fclose(fd_out);
        end

        write_passfail("PASS");
        $finish;
    end

    task automatic write_passfail(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin
            $fwrite(fd_pf, "%s\n", msg);
            $fclose(fd_pf);
        end
    endtask

endmodule
