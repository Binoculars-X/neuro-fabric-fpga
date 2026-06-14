// Testbench for fp32_div (combinatorial FP32 scalar divide)
//
// Reads:
//   $NEURO_TESTVECS/fp32_div/input.hex  -- N_VECS*2 lines: a_fp32, b_fp32 per case
// Writes:
//   $NEURO_TESTVECS/fp32_div/output.hex -- N_VECS lines: actual RTL results
//   $NEURO_TESTVECS/fp32_div/pass_fail.txt -- always "PASS" if simulation completes
//
// Numeric verification (1 ULP vs C# a/b reference) is done in C# by reading output.hex.

`timescale 1ns/1ps

module tb_fp32_div;

    localparam int N_VECS = 8;

    logic [31:0] a_fp32;
    logic [31:0] b_fp32;
    logic [31:0] result_fp32;

    fp32_div dut (
        .a_fp32     (a_fp32),
        .b_fp32     (b_fp32),
        .result_fp32(result_fp32)
    );

    logic [31:0] in_a[0:N_VECS-1];
    logic [31:0] in_b[0:N_VECS-1];
    logic [31:0] got [0:N_VECS-1];

    string  testvecs_dir;
    string  input_path, passfail_path, output_path;
    integer fd_in, fd_pf, fd_out, scan_ok;

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";
        input_path    = {testvecs_dir, "/fp32_div/input.hex"};
        passfail_path = {testvecs_dir, "/fp32_div/pass_fail.txt"};
        output_path   = {testvecs_dir, "/fp32_div/output.hex"};

        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin write_pf({"FAIL:cannot open ", input_path}); $finish; end
        for (int i = 0; i < N_VECS; i++) begin
            scan_ok = $fscanf(fd_in, "%h\n", in_a[i]);
            if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (a)"); $fclose(fd_in); $finish; end
            scan_ok = $fscanf(fd_in, "%h\n", in_b[i]);
            if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (b)"); $fclose(fd_in); $finish; end
        end
        $fclose(fd_in);

        for (int c = 0; c < N_VECS; c++) begin
            a_fp32 = in_a[c];
            b_fp32 = in_b[c];
            #10;
            got[c] = result_fp32;
        end

        fd_out = $fopen(output_path, "w");
        if (fd_out != 0) begin
            for (int c = 0; c < N_VECS; c++)
                $fwrite(fd_out, "%08h\n", got[c]);
            $fclose(fd_out);
        end

        write_pf("PASS");
        $finish;
    end

endmodule
