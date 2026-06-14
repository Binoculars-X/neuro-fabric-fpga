// Testbench for exp_lut
//
// Reads:
//   $NEURO_TESTVECS/exp_lut/input.hex  — N_VECTORS lines of FP32 hex (x inputs)
//
// Writes:
//   $NEURO_TESTVECS/exp_lut/output.hex    — N_VECTORS lines: actual RTL results
//   $NEURO_TESTVECS/exp_lut/pass_fail.txt — "PASS" when all outputs collected
//
// Numeric verification is done in C# by reading output.hex.

`timescale 1ns/1ps

module tb_exp_lut;

    localparam int LATENCY   = 4;
    localparam int N_VECTORS = 33;
    localparam int LUT_SIZE  = 256;

    // ------------------------------------------------------------------
    // DUT signals
    // ------------------------------------------------------------------
    logic        clk = 0;
    logic        rst = 1;
    logic        en  = 1;
    logic [31:0] x_fp32   = 0;
    logic        valid_in = 0;
    logic [31:0] result_fp32;
    logic        valid_out;

    // LUT file path — relative to xsim working dir (xsim_work/ inside testvecs/exp_lut/)
    // C# runner copies exp_lut_init.hex there before launching xsim.
    exp_lut #(
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
    logic [31:0] in_x    [0:N_VECTORS-1];
    logic [31:0] got_out  [0:N_VECTORS-1];

    // ------------------------------------------------------------------
    // File / loop state
    // ------------------------------------------------------------------
    string  testvecs_dir;
    string  input_path, passfail_path, output_path;
    integer fd_in, fd_pf, fd_out, scan_ok;
    int     in_idx, out_idx, total_cycles, cyc;

    // ------------------------------------------------------------------
    // Main
    // ------------------------------------------------------------------
    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";

        input_path    = {testvecs_dir, "/exp_lut/input.hex"};
        output_path   = {testvecs_dir, "/exp_lut/output.hex"};
        passfail_path = {testvecs_dir, "/exp_lut/pass_fail.txt"};

        // Read inputs
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

            // Drive input
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
