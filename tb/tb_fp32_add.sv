// Testbench for fp32_add — synthesizable IEEE 754 FP32 adder
//
// Protocol:
//   Reads  $NEURO_TESTVECS/fp32_add/input.hex   — N_VECS*2 lines: a, b (FP32 hex)
//   Writes $NEURO_TESTVECS/fp32_add/output.hex  — N_VECS lines: RTL result (FP32 hex)
//   Writes $NEURO_TESTVECS/fp32_add/pass_fail.txt
//
// LATENCY: fp32_add has 4-cycle pipeline.
// shortreal permitted in tb/ for $display diagnostics only — never in DUT.

`timescale 1ns/1ps

module tb_fp32_add;

    localparam int N_VECS   = 64;
    localparam int LATENCY  = 4;
    localparam int CLK_HALF = 5;

    logic clk = 1'b0;
    always #CLK_HALF clk = ~clk;

    logic        rst      = 1'b1;
    logic        valid_in = 1'b0;
    logic [31:0] a_fp32   = 32'h0;
    logic [31:0] b_fp32   = 32'h0;
    logic [31:0] result;
    logic        valid_out;

    fp32_add dut (
        .clk      (clk),
        .rst      (rst),
        .valid_in (valid_in),
        .a        (a_fp32),
        .b        (b_fp32),
        .result   (result),
        .valid_out(valid_out)
    );

    logic [31:0] in_a [0:N_VECS-1];
    logic [31:0] in_b [0:N_VECS-1];
    logic [31:0] got  [0:N_VECS-1];

    shortreal ref_val;   // diagnostic only — tb/ is allowed to use shortreal

    string testvecs_dir;
    string input_path, output_path, passfail_path;
    integer fd_in, fd_out, fd_pf, scan_ok;

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

    int out_idx;

    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";
        input_path    = {testvecs_dir, "/fp32_add/input.hex"};
        output_path   = {testvecs_dir, "/fp32_add/output.hex"};
        passfail_path = {testvecs_dir, "/fp32_add/pass_fail.txt"};

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

        repeat (4) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

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
            if (valid_out && out_idx < N_VECS) begin
                got[out_idx] = result;
                ref_val = $bitstoshortreal(in_a[out_idx]) + $bitstoshortreal(in_b[out_idx]);
                $display("[%0d] a=%h b=%h RTL=%h ref=%h",
                    out_idx, in_a[out_idx], in_b[out_idx], result, $shortrealtobits(ref_val));
                out_idx++;
            end
        end

        if (out_idx != N_VECS) begin
            write_pf($sformatf("FAIL:only %0d of %0d outputs received", out_idx, N_VECS));
            $finish;
        end

        fd_out = $fopen(output_path, "w");
        if (fd_out == 0) begin write_pf({"FAIL:cannot write ", output_path}); $finish; end
        for (int i = 0; i < N_VECS; i++)
            $fwrite(fd_out, "%08h\n", got[i]);
        $fclose(fd_out);

        write_pf("PASS");
        $finish;
    end

endmodule
