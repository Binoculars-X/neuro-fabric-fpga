// Testbench for fp32_sqrt (combinatorial 1/sqrt(x))
//
// Reads:
//   $NEURO_TESTVECS/fp32_sqrt/input.hex   -- N_VECS lines: x_fp32 per case
// Writes:
//   $NEURO_TESTVECS/fp32_sqrt/output.hex  -- N_VECS lines: actual RTL results
//   $NEURO_TESTVECS/fp32_sqrt/pass_fail.txt -- "PASS" if simulation completes
//
// Numeric verification (VsSoftwareRelTol vs C# ExpLutHelper.RecipSqrt reference)
// is done in C# by reading output.hex.

`timescale 1ns/1ps

module tb_fp32_sqrt;

    localparam int N_VECS = 32;

    logic [31:0] x_fp32;
    logic [31:0] result_fp32;

    fp32_sqrt dut (
        .x_fp32     (x_fp32),
        .result_fp32(result_fp32)
    );

    logic [31:0] in_x [0:N_VECS-1];
    logic [31:0] got  [0:N_VECS-1];

    string  testvecs_dir;
    string  input_path, output_path, passfail_path;
    integer fd_in, fd_out, fd_pf, scan_ok;

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";
        input_path    = {testvecs_dir, "/fp32_sqrt/input.hex"};
        output_path   = {testvecs_dir, "/fp32_sqrt/output.hex"};
        passfail_path = {testvecs_dir, "/fp32_sqrt/pass_fail.txt"};

        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin write_pf({"FAIL:cannot open ", input_path}); $finish; end
        for (int i = 0; i < N_VECS; i++) begin
            scan_ok = $fscanf(fd_in, "%h\n", in_x[i]);
            if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed"); $fclose(fd_in); $finish; end
        end
        $fclose(fd_in);

        for (int c = 0; c < N_VECS; c++) begin
            x_fp32 = in_x[c];
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
