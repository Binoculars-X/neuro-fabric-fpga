// Testbench for layernorm (D=4, T=4 token rows)
//
// Reads:
//   $NEURO_TESTVECS/layernorm/input.hex  -- T*(3*D) lines per token:
//                                           D FP32 x lines, D FP32 gamma lines, D FP32 beta lines
// Writes:
//   $NEURO_TESTVECS/layernorm/output.hex -- T*D FP32 lines (row-major)
//   $NEURO_TESTVECS/layernorm/pass_fail.txt -- "PASS" if all T rows produced out_valid
//
// recipsqrt_rom.hex must be present in xsim_work/ before simulation.
// Numeric verification is done in C# by reading output.hex.

`timescale 1ns/1ps

module tb_layernorm;

    localparam int D      = 4;
    localparam int T      = 4;
    localparam int TIMEOUT = 50; // cycles per token row before FAIL

    logic clk = 0;
    logic rst = 1;
    logic en  = 1;

    logic [D*32-1:0] x_in;
    logic [D*32-1:0] gamma;
    logic [D*32-1:0] beta;
    logic             start;
    logic [D*32-1:0] y_out;
    logic             out_valid;

    layernorm #(.D(D)) dut (
        .clk      (clk),
        .rst      (rst),
        .en       (en),
        .x_in     (x_in),
        .gamma    (gamma),
        .beta     (beta),
        .start    (start),
        .y_out    (y_out),
        .out_valid(out_valid)
    );

    always #5 clk = ~clk;

    logic [31:0] in_x    [0:T-1][0:D-1];
    logic [31:0] in_gamma[0:T-1][0:D-1];
    logic [31:0] in_beta [0:T-1][0:D-1];
    logic [31:0] got_y   [0:T-1][0:D-1];

    string  testvecs_dir;
    string  input_path, output_path, passfail_path;
    integer fd_in, fd_out, fd_pf, scan_ok;
    int     cyc, rows_done;

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";
        input_path    = {testvecs_dir, "/layernorm/input.hex"};
        output_path   = {testvecs_dir, "/layernorm/output.hex"};
        passfail_path = {testvecs_dir, "/layernorm/pass_fail.txt"};

        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin write_pf({"FAIL:cannot open ", input_path}); $finish; end
        for (int t = 0; t < T; t++) begin
            for (int d = 0; d < D; d++) begin
                scan_ok = $fscanf(fd_in, "%h\n", in_x    [t][d]);
                if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (x)");     $fclose(fd_in); $finish; end
            end
            for (int d = 0; d < D; d++) begin
                scan_ok = $fscanf(fd_in, "%h\n", in_gamma[t][d]);
                if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (gamma)"); $fclose(fd_in); $finish; end
            end
            for (int d = 0; d < D; d++) begin
                scan_ok = $fscanf(fd_in, "%h\n", in_beta [t][d]);
                if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (beta)");  $fclose(fd_in); $finish; end
            end
        end
        $fclose(fd_in);

        repeat(3) @(posedge clk); #1;
        rst = 0;

        rows_done = 0;
        for (int t = 0; t < T; t++) begin
            @(posedge clk); #1;
            for (int d = 0; d < D; d++) begin
                x_in  [d*32 +: 32] = in_x    [t][d];
                gamma [d*32 +: 32] = in_gamma [t][d];
                beta  [d*32 +: 32] = in_beta  [t][d];
            end
            start = 1;
            @(posedge clk); #1;
            start = 0;

            for (cyc = 0; cyc < TIMEOUT; cyc++) begin
                @(posedge clk); #1;
                if (out_valid) begin
                    for (int d = 0; d < D; d++)
                        got_y[t][d] = y_out[d*32 +: 32];
                    rows_done++;
                    break;
                end
            end
            if (rows_done != t + 1) begin
                write_pf($sformatf("FAIL:row%0d timed out", t));
                $finish;
            end
        end

        fd_out = $fopen(output_path, "w");
        if (fd_out != 0) begin
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    $fwrite(fd_out, "%08h\n", got_y[t][d]);
            $fclose(fd_out);
        end

        write_pf("PASS");
        $finish;
    end

endmodule
