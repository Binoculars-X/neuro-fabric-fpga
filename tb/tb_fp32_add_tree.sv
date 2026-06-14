// Testbench for fp32_add_tree (T=4 combinatorial FP32 adder tree)
//
// Reads:
//   $NEURO_TESTVECS/fp32_add_tree/input.hex  -- N_VECS*T lines of FP32 hex (4 inputs per case)
// Writes:
//   $NEURO_TESTVECS/fp32_add_tree/output.hex -- N_VECS lines: actual RTL sums
//   $NEURO_TESTVECS/fp32_add_tree/pass_fail.txt -- always "PASS" if simulation completes
//
// Numeric verification (1 ULP vs tree-order C# reference) is done in C# by reading output.hex.

`timescale 1ns/1ps

module tb_fp32_add_tree;

    localparam int T      = 4;
    localparam int N_VECS = 8;

    logic [T*32-1:0] in_vec;
    logic [31:0]     sum_out;

    fp32_add_tree #(.T(T)) dut (
        .in_vec  (in_vec),
        .sum_out (sum_out)
    );

    logic [31:0] in_vals[0:N_VECS*T-1];
    logic [31:0] got    [0:N_VECS-1];

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
        input_path    = {testvecs_dir, "/fp32_add_tree/input.hex"};
        passfail_path = {testvecs_dir, "/fp32_add_tree/pass_fail.txt"};
        output_path   = {testvecs_dir, "/fp32_add_tree/output.hex"};

        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin write_pf({"FAIL:cannot open ", input_path}); $finish; end
        for (int i = 0; i < N_VECS * T; i++) begin
            scan_ok = $fscanf(fd_in, "%h\n", in_vals[i]);
            if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed"); $fclose(fd_in); $finish; end
        end
        $fclose(fd_in);

        for (int c = 0; c < N_VECS; c++) begin
            for (int j = 0; j < T; j++)
                in_vec[j*32 +: 32] = in_vals[c*T + j];
            #10;
            got[c] = sum_out;
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
