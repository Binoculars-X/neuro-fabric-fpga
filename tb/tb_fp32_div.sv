// tb_fp32_div.sv -- XSim testbench for fp32_div (clocked FSM, valid_in/valid_out)
//
// Reads:
//   $NEURO_TESTVECS/fp32_div/input.hex  -- N_VECS*2 lines: a_fp32, b_fp32 per case
// Writes:
//   $NEURO_TESTVECS/fp32_div/output.hex -- N_VECS lines: actual RTL results
//
// Numeric verification (vs C# reference) is done in C# by reading output.hex.
// shortreal used only in $display diagnostics (allowed in tb/).

`timescale 1ns/1ps

module tb_fp32_div;

    localparam int N_VECS    = 32;
    localparam int MAX_WAIT  = 200;
    localparam int CLK_HALF  = 5;

    logic        clk = 0;
    logic        rst = 1;
    logic        valid_in;
    logic [31:0] a_fp32, b_fp32;
    logic [31:0] result_fp32;
    logic        valid_out;

    always #CLK_HALF clk = ~clk;

    fp32_div dut (
        .clk        (clk),
        .rst        (rst),
        .valid_in   (valid_in),
        .a_fp32     (a_fp32),
        .b_fp32     (b_fp32),
        .result_fp32(result_fp32),
        .valid_out  (valid_out)
    );

    logic [31:0] in_a[0:N_VECS-1];
    logic [31:0] in_b[0:N_VECS-1];
    logic [31:0] got [0:N_VECS-1];

    string  testvecs_dir;
    string  input_path, output_path, passfail_path;
    integer fd_in, fd_out, fd_pf, scan_ok;

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

    // Wait for valid_out up to MAX_WAIT cycles; returns 1 on timeout
    task automatic wait_valid(output logic timed_out);
        integer cnt;
        cnt = 0;
        timed_out = 0;
        while (!valid_out) begin
            @(posedge clk);
            cnt = cnt + 1;
            if (cnt >= MAX_WAIT) begin timed_out = 1; disable wait_valid; end
        end
    endtask

    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";
        input_path    = {testvecs_dir, "/fp32_div/input.hex"};
        output_path   = {testvecs_dir, "/fp32_div/output.hex"};
        passfail_path = {testvecs_dir, "/fp32_div/pass_fail.txt"};

        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin write_pf({"FAIL:cannot open ", input_path}); $finish; end
        for (int i = 0; i < N_VECS; i++) begin
            scan_ok = $fscanf(fd_in, "%h\n", in_a[i]);
            if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (a)"); $fclose(fd_in); $finish; end
            scan_ok = $fscanf(fd_in, "%h\n", in_b[i]);
            if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (b)"); $fclose(fd_in); $finish; end
        end
        $fclose(fd_in);

        valid_in = 0; a_fp32 = 0; b_fp32 = 0;
        repeat(4) @(posedge clk);
        rst = 0;
        @(posedge clk);

        for (int c = 0; c < N_VECS; c++) begin
            logic timed_out;
            @(posedge clk);
            a_fp32   = in_a[c];
            b_fp32   = in_b[c];
            valid_in = 1;
            @(posedge clk);
            valid_in = 0;
            wait_valid(timed_out);
            if (timed_out) begin
                write_pf($sformatf("FAIL:case[%0d] timed out after %0d cycles", c, MAX_WAIT));
                $finish;
            end
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