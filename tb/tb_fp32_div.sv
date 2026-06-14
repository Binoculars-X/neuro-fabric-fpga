// Testbench for fp32_div (combinatorial FP32 scalar divide)
//
// Self-contained: test vectors are hardcoded.
// Tolerance: 1 ULP — reference uses (float)((double)a / (double)b).
//
// Writes: $NEURO_TESTVECS/fp32_div/pass_fail.txt

`timescale 1ns/1ps

module tb_fp32_div;

    localparam int N_CASES = 8;

    logic [31:0] a_fp32;
    logic [31:0] b_fp32;
    logic [31:0] result_fp32;

    fp32_div dut (
        .a_fp32     (a_fp32),
        .b_fp32     (b_fp32),
        .result_fp32(result_fp32)
    );

    logic [31:0] tv_a  [0:N_CASES-1];
    logic [31:0] tv_b  [0:N_CASES-1];
    logic [31:0] tv_exp[0:N_CASES-1];

    string  testvecs_dir, passfail_path;
    integer fd_pf;
    int     fail_count, ulp_diff;
    string  fail_detail;

    function automatic int ulp_distance(input logic [31:0] a, input logic [31:0] b);
        int ia, ib;
        ia = int'(a);
        ib = int'(b);
        if (ia < 0) ia = 32'h80000000 - ia;
        if (ib < 0) ib = 32'h80000000 - ib;
        return (ia > ib) ? (ia - ib) : (ib - ia);
    endfunction

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        $fwrite(fd_pf, "%s", msg);
        $fclose(fd_pf);
    endtask

    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";
        passfail_path = {testvecs_dir, "/fp32_div/pass_fail.txt"};

        // ---- Test cases (FP32 hex) ----
        // Case 0: 1.0 / 1.0 = 1.0
        tv_a[0] = 32'h3F800000; tv_b[0] = 32'h3F800000; tv_exp[0] = 32'h3F800000;

        // Case 1: 6.0 / 3.0 = 2.0
        tv_a[1] = 32'h40C00000; tv_b[1] = 32'h40400000; tv_exp[1] = 32'h40000000;

        // Case 2: 1.0 / 4.0 = 0.25
        tv_a[2] = 32'h3F800000; tv_b[2] = 32'h40800000; tv_exp[2] = 32'h3E800000;

        // Case 3: -3.0 / 2.0 = -1.5
        tv_a[3] = 32'hC0400000; tv_b[3] = 32'h40000000; tv_exp[3] = 32'hBFC00000;

        // Case 4: softmax normalisation: exp(0)/sum — 1.0 / 2.10975 ≈ 0.47404
        // (float)((double)1.0 / (double)2.10975) = 0.47404... = 0x3EF2F6AB (approx)
        tv_a[4] = 32'h3F800000; tv_b[4] = 32'h40073C73; tv_exp[4] = 32'h3EF2F6AB;

        // Case 5: 0.0 / 2.0 = 0.0 (masked position after exp → 0/sum = 0)
        tv_a[5] = 32'h00000000; tv_b[5] = 32'h40000000; tv_exp[5] = 32'h00000000;

        // Case 6: 10.0 / 4.0 = 2.5
        tv_a[6] = 32'h41200000; tv_b[6] = 32'h40800000; tv_exp[6] = 32'h40200000;

        // Case 7: 100.0 / 1000.0 = 0.1
        tv_a[7] = 32'h42C80000; tv_b[7] = 32'h447A0000; tv_exp[7] = 32'h3DCCCCCD;

        fail_count  = 0;
        fail_detail = "";

        for (int c = 0; c < N_CASES; c++) begin
            a_fp32 = tv_a[c];
            b_fp32 = tv_b[c];
            #10;

            ulp_diff = ulp_distance(result_fp32, tv_exp[c]);
            if (ulp_diff > 1) begin
                fail_count++;
                $sformat(fail_detail,
                    "FAIL case%0d: got %h exp %h ulp=%0d",
                    c, result_fp32, tv_exp[c], ulp_diff);
                $display("%s", fail_detail);
            end
        end

        if (fail_count == 0)
            write_pf("PASS");
        else
            write_pf(fail_detail);

        $finish;
    end

endmodule
