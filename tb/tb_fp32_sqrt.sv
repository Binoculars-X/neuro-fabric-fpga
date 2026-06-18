// Testbench for fp32_sqrt — synthesizable FSM-based 1/sqrt(x)
//
// Protocol: FSM-based DUT — one transaction at a time, wait for valid_out per case.
//
// Reads:
//   $NEURO_TESTVECS/fp32_sqrt/input.hex     — N_VECS lines: x_fp32 hex per case
// Writes:
//   $NEURO_TESTVECS/fp32_sqrt/output.hex    — N_VECS lines: RTL result (FP32 hex)
//   $NEURO_TESTVECS/fp32_sqrt/pass_fail.txt — "PASS" if all outputs received
//
// Numeric verification (VsSoftwareRelTol vs C# ExpLutHelper.RecipSqrt reference)
// is done in C# by reading output.hex.
//
// tb/ files MAY use shortreal for reference display diagnostics only.
//
// LATENCY: fp32_sqrt FSM is ~29 cycles per transaction.
// MAX_WAIT: 200 cycles per transaction (hard timeout guard).

`timescale 1ns/1ps

module tb_fp32_sqrt;

    localparam int N_VECS    = 32;
    localparam int CLK_HALF  = 5;    // 100 MHz
    localparam int MAX_WAIT  = 200;  // cycles per transaction before timeout

    // ── Clock ────────────────────────────────────────────────────────────────
    logic clk = 1'b0;
    always #CLK_HALF clk = ~clk;

    // ── DUT ports ────────────────────────────────────────────────────────────
    logic        rst         = 1'b1;
    logic        valid_in    = 1'b0;
    logic [31:0] x_fp32      = 32'h0;
    logic [31:0] result_fp32;
    logic        valid_out;

    fp32_sqrt dut (
        .clk        (clk),
        .rst        (rst),
        .valid_in   (valid_in),
        .x_fp32     (x_fp32),
        .result_fp32(result_fp32),
        .valid_out  (valid_out)
    );

    // ── Test vector storage ───────────────────────────────────────────────────
    logic [31:0] in_x [0:N_VECS-1];
    logic [31:0] got  [0:N_VECS-1];

    // ── Reference display (shortreal allowed in tb/ only) ─────────────────────
    shortreal ref_val;

    // ── File paths ────────────────────────────────────────────────────────────
    string  testvecs_dir;
    string  input_path, output_path, passfail_path;
    integer fd_in, fd_out, fd_pf, scan_ok;

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

    // ── Main sequence ─────────────────────────────────────────────────────────
    int wait_cnt;

    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, tb_fp32_sqrt);
    end

    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";
        input_path    = {testvecs_dir, "/fp32_sqrt/input.hex"};
        output_path   = {testvecs_dir, "/fp32_sqrt/output.hex"};
        passfail_path = {testvecs_dir, "/fp32_sqrt/pass_fail.txt"};

        // Read input vectors
        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin write_pf({"FAIL:cannot open ", input_path}); $finish; end
        for (int i = 0; i < N_VECS; i++) begin
            scan_ok = $fscanf(fd_in, "%h\n", in_x[i]);
            if (scan_ok != 1) begin
                write_pf("FAIL:input.hex malformed");
                $fclose(fd_in);
                $finish;
            end
        end
        $fclose(fd_in);

        // Reset
        repeat (4) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        // Drive one transaction at a time; wait for valid_out before next
        for (int c = 0; c < N_VECS; c++) begin
            // Assert valid_in for one cycle
            x_fp32   <= in_x[c];
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;

            // Wait for valid_out (up to MAX_WAIT cycles)
            wait_cnt = 0;
            while (!valid_out && wait_cnt < MAX_WAIT) begin
                @(posedge clk);
                wait_cnt++;
            end

            if (!valid_out) begin
                write_pf($sformatf("FAIL:timeout waiting for valid_out on case %0d", c));
                $finish;
            end

            got[c] = result_fp32;
            ref_val = 1.0 / $sqrt($bitstoshortreal(in_x[c]));
            $display("[%0d] x=%h RTL=%h ref=%h",
                c, in_x[c], result_fp32, $shortrealtobits(ref_val));

            @(posedge clk); // one idle cycle between transactions
        end

        // Write output.hex
        fd_out = $fopen(output_path, "w");
        if (fd_out == 0) begin write_pf({"FAIL:cannot write ", output_path}); $finish; end
        for (int i = 0; i < N_VECS; i++)
            $fwrite(fd_out, "%08h\n", got[i]);
        $fclose(fd_out);

        write_pf("PASS");
        $finish;
    end

endmodule
