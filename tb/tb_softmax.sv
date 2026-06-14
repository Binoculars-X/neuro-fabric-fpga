// Testbench for softmax (T=4, causal mask, LUT-256 exp)
//
// Reads:
//   $NEURO_TESTVECS/softmax/input.hex  -- N_ROWS*(T+1) lines per row:
//                                         T FP32 score lines, then 1 byte row_idx
//   exp_lut_init.hex must be present in xsim_work/
// Writes:
//   $NEURO_TESTVECS/softmax/output.hex -- N_ROWS*T lines: actual RTL probs (row-major)
//   $NEURO_TESTVECS/softmax/pass_fail.txt -- "PASS" if all rows produced valid_out,
//                                            "FAIL:rowN timed out" otherwise
//
// Numeric verification is done in C# by reading output.hex.

`timescale 1ns/1ps

module tb_softmax;

    localparam int T       = 4;
    localparam int EXP_LAT = 4;
    localparam int N_ROWS  = 4;

    logic clk = 0;
    logic rst = 1;
    logic en  = 1;

    logic [T*32-1:0]      scores_in;
    logic [$clog2(T)-1:0] row_idx;
    logic                  start;
    logic [T*32-1:0]      probs_out;
    logic                  valid_out;

    softmax #(
        .T       (T),
        .EXP_LAT (EXP_LAT),
        .LUT_SIZE(256),
        .LUT_FILE("exp_lut_init.hex")
    ) dut (
        .clk       (clk),
        .rst       (rst),
        .en        (en),
        .scores_in (scores_in),
        .row_idx   (row_idx),
        .start     (start),
        .probs_out (probs_out),
        .valid_out (valid_out)
    );

    always #5 clk = ~clk;

    logic [31:0] in_scores[0:N_ROWS-1][0:T-1];
    logic [1:0]  in_row   [0:N_ROWS-1];
    logic [31:0] got_probs[0:N_ROWS-1][0:T-1];

    string  testvecs_dir;
    string  input_path, passfail_path, output_path;
    integer fd_in, fd_pf, fd_out, scan_ok;
    int     cyc, rows_done;
    logic [7:0] tmp_byte;

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";
        input_path    = {testvecs_dir, "/softmax/input.hex"};
        passfail_path = {testvecs_dir, "/softmax/pass_fail.txt"};
        output_path   = {testvecs_dir, "/softmax/output.hex"};

        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin write_pf({"FAIL:cannot open ", input_path}); $finish; end
        for (int r = 0; r < N_ROWS; r++) begin
            for (int j = 0; j < T; j++) begin
                scan_ok = $fscanf(fd_in, "%h\n", in_scores[r][j]);
                if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (scores)"); $fclose(fd_in); $finish; end
            end
            scan_ok = $fscanf(fd_in, "%h\n", tmp_byte);
            if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (row_idx)"); $fclose(fd_in); $finish; end
            in_row[r] = tmp_byte[1:0];
        end
        $fclose(fd_in);

        repeat(3) @(posedge clk); #1;
        rst = 0;

        rows_done = 0;
        for (int r = 0; r < N_ROWS; r++) begin
            @(posedge clk); #1;
            for (int j = 0; j < T; j++)
                scores_in[j*32 +: 32] = in_scores[r][j];
            row_idx = in_row[r];
            start   = 1;
            @(posedge clk); #1;
            start = 0;

            for (cyc = 0; cyc < 30; cyc++) begin
                @(posedge clk); #1;
                if (valid_out) begin
                    for (int j = 0; j < T; j++)
                        got_probs[r][j] = probs_out[j*32 +: 32];
                    rows_done++;
                    break;
                end
            end
            if (rows_done != r + 1) begin
                write_pf($sformatf("FAIL:row%0d timed out", r));
                $finish;
            end
        end

        fd_out = $fopen(output_path, "w");
        if (fd_out != 0) begin
            for (int r = 0; r < N_ROWS; r++)
                for (int j = 0; j < T; j++)
                    $fwrite(fd_out, "%08h\n", got_probs[r][j]);
            $fclose(fd_out);
        end

        write_pf("PASS");
        $finish;
    end

endmodule
