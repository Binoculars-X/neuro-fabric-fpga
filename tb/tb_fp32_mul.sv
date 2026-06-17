// Testbench for fp32_mul — synthesizable IEEE 754 FP32 multiplier
//
// Protocol:
//   Reads  $NEURO_TESTVECS/fp32_mul/input.hex   — N_VECS*2 lines: a, b (FP32 hex)
//   Writes $NEURO_TESTVECS/fp32_mul/output.hex  — N_VECS lines: RTL result (FP32 hex)
//   Writes $NEURO_TESTVECS/fp32_mul/pass_fail.txt
//
// Numeric verification (≤1 ULP vs C# (float)((double)a*(double)b)) is in C#.
// tb/ files MAY use shortreal for reference display only — never in the DUT.
//
// LATENCY: fp32_mul has 3-cycle pipeline. Testbench uses a shift-register
// to align input index with output window.

`timescale 1ns/1ps

module tb_fp32_mul;

    // Must match C# Fp32MulVecGen.NVecs
    localparam int N_VECS   = 64;
    localparam int LATENCY  = 3;
    localparam int CLK_HALF = 5;   // 100 MHz

    // ── Clock ────────────────────────────────────────────────────────────────
    logic clk = 1'b0;
    always #CLK_HALF clk = ~clk;

    // ── DUT ports ────────────────────────────────────────────────────────────
    logic        rst      = 1'b1;
    logic        valid_in = 1'b0;
    logic [31:0] a_fp32   = 32'h0;
    logic [31:0] b_fp32   = 32'h0;
    logic [31:0] result;
    logic        valid_out;

    fp32_mul dut (
        .clk      (clk),
        .rst      (rst),
        .valid_in (valid_in),
        .a        (a_fp32),
        .b        (b_fp32),
        .result   (result),
        .valid_out(valid_out)
    );

    // ── Test vector storage ───────────────────────────────────────────────────
    logic [31:0] in_a [0:N_VECS-1];
    logic [31:0] in_b [0:N_VECS-1];
    logic [31:0] got  [0:N_VECS-1];

    // ── Reference comparison (shortreal allowed in tb/ only) ─────────────────
    // Used only for $display diagnostics, not for pass/fail decision.
    shortreal    ref_val;

    // ── File paths ────────────────────────────────────────────────────────────
    string testvecs_dir;
    string input_path, output_path, passfail_path;
    integer fd_in, fd_out, fd_pf, scan_ok;

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

    // ── Main sequence ─────────────────────────────────────────────────────────
    int out_idx;

    initial begin
        // Resolve test-vector directory
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";
        input_path    = {testvecs_dir, "/fp32_mul/input.hex"};
        output_path   = {testvecs_dir, "/fp32_mul/output.hex"};
        passfail_path = {testvecs_dir, "/fp32_mul/pass_fail.txt"};

        // Read input vectors
        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin
            write_pf({"FAIL:cannot open ", input_path});
            $finish;
        end
        for (int i = 0; i < N_VECS; i++) begin
            scan_ok = $fscanf(fd_in, "%h\n", in_a[i]);
            if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (a)"); $fclose(fd_in); $finish; end
            scan_ok = $fscanf(fd_in, "%h\n", in_b[i]);
            if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (b)"); $fclose(fd_in); $finish; end
        end
        $fclose(fd_in);

        // Reset
        repeat (4) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        // Drive inputs: N_VECS cycles with valid_in=1, then LATENCY drain cycles
        out_idx = 0;
        for (int i = 0; i < N_VECS + LATENCY; i++) begin
            if (i < N_VECS) begin
                a_fp32   <= in_a[i];
                b_fp32   <= in_b[i];
                valid_in <= 1'b1;
            end else begin
                valid_in <= 1'b0;
            end
            @(posedge clk);
            // Capture outputs as they emerge (latency-aligned)
            if (valid_out && out_idx < N_VECS) begin
                got[out_idx] = result;
                // Diagnostic: compare against shortreal reference
                ref_val = $bitstoshortreal(in_a[out_idx]) * $bitstoshortreal(in_b[out_idx]);
                $display("[%0d] a=%h b=%h RTL=%h ref=%h",
                    out_idx, in_a[out_idx], in_b[out_idx], result, $shortrealtobits(ref_val));
                out_idx++;
            end
        end

        if (out_idx != N_VECS) begin
            write_pf($sformatf("FAIL:only %0d of %0d outputs received", out_idx, N_VECS));
            $finish;
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
