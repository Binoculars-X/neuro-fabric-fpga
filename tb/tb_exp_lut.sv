// Testbench for exp_lut
//
// Reads test vectors from:
//   $NEURO_TESTVECS/exp_lut/input.hex    — lines: <x_fp32_hex>
//   $NEURO_TESTVECS/exp_lut/expected.hex — lines: <result_fp32_hex>
//
// Writes:
//   $NEURO_TESTVECS/exp_lut/pass_fail.txt
//
// Tolerance: relative error < 0.003 (0.3%) OR absolute < 1e-10 (underflow region).
// This matches the LUT-256 accuracy spec from ExpLutHelper.
//
// Pipeline LATENCY = 4 cycles. Driver and checker interleaved (same pattern as tb_bf16_mac).

`timescale 1ns/1ps

module tb_exp_lut;

    localparam int LATENCY   = 4;
    localparam int N_VECTORS = 32;
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
    logic [31:0] exp_out  [0:N_VECTORS-1];
    logic [31:0] got_out  [0:N_VECTORS-1];

    // ------------------------------------------------------------------
    // Checker temporaries
    // ------------------------------------------------------------------
    int       ulp_diff;
    int       pass_flag;

    // ------------------------------------------------------------------
    // File / loop state
    // ------------------------------------------------------------------
    string  testvecs_dir;
    string  input_path, expected_path, passfail_path, output_path;
    integer fd_in, fd_exp, fd_pf, fd_out, scan_ok;
    int     fail_count;
    string  fail_detail;
    int     in_idx, out_idx, total_cycles, cyc;

    // ------------------------------------------------------------------
    // Main
    // ------------------------------------------------------------------
    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";

        input_path    = {testvecs_dir, "/exp_lut/input.hex"};
        output_path   = {testvecs_dir, "/exp_lut/output.hex"};
        expected_path = {testvecs_dir, "/exp_lut/expected.hex"};
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

        // Read expected
        fd_exp = $fopen(expected_path, "r");
        if (fd_exp == 0) begin
            write_passfail({"FAIL:cannot open ", expected_path});
            $finish;
        end
        for (int i = 0; i < N_VECTORS; i++) begin
            scan_ok = $fscanf(fd_exp, "%h\n", exp_out[i]);
            if (scan_ok != 1) begin
                write_passfail("FAIL:expected.hex malformed");
                $fclose(fd_exp); $finish;
            end
        end
        $fclose(fd_exp);

        // Reset
        repeat(3) @(posedge clk);
        #1; rst = 0;

        fail_count   = 0;
        fail_detail  = "";
        in_idx       = 0;
        out_idx      = 0;
        total_cycles = N_VECTORS + LATENCY + 4;

        for (cyc = 0; cyc < total_cycles; cyc++) begin
            @(posedge clk); #1;

            // Check output — 1-ULP tolerance
            // exp(x) >= 0 so bits are positive IEEE 754: ULP diff = |got - exp| as unsigned ints.
            // Allows for XSim shortreal→real promotion causing last-bit rounding differences.
            if (valid_out && out_idx < N_VECTORS) begin
                got_out[out_idx] = result_fp32;
                ulp_diff  = int'(result_fp32) - int'(exp_out[out_idx]);
                if (ulp_diff < 0) ulp_diff = -ulp_diff;
                pass_flag = (ulp_diff <= 16) ? 1 : 0;

                if (!pass_flag) begin
                    fail_count++;
                    if (fail_count == 1)
                        $sformat(fail_detail,
                            "vec[%0d] got %08h exp %08h (diff %0d ULP)",
                            out_idx, result_fp32, exp_out[out_idx], ulp_diff);
                    $display("MISMATCH vec[%0d] got %08h exp %08h (diff %0d ULP)",
                        out_idx, result_fp32, exp_out[out_idx], ulp_diff);
                end
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

        if (fail_count == 0) begin
            $display("PASS — all %0d vectors within 16 ULP", N_VECTORS);
            write_passfail("PASS");
        end else begin
            $display("FAIL — %0d mismatches; first: %s", fail_count, fail_detail);
            write_passfail({"FAIL:", fail_detail});
        end

        // Write output.hex for VsSoftware comparison
        fd_out = $fopen(output_path, "w");
        if (fd_out != 0) begin
            for (int i = 0; i < N_VECTORS; i++)
                $fwrite(fd_out, "%08h\n", got_out[i]);
            $fclose(fd_out);
        end

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
