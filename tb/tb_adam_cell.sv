// Testbench for adam_cell (STEPS=10, one scalar parameter)
//
// Reads:
//   $NEURO_TESTVECS/adam_cell/input.hex  -- per step (7 lines each):
//     g_fp32, w_bf16 (16-bit, zero-padded to 32), m_fp32, v_fp32,
//     lr_fp32, bc1_fp32, bc2_fp32
//
// Writes:
//   $NEURO_TESTVECS/adam_cell/output.hex -- STEPS*3 lines per test case:
//     w_bf16_out (16-bit, zero-padded to 32), m_fp32_out, v_fp32_out
//   $NEURO_TESTVECS/adam_cell/pass_fail.txt -- "PASS" on success
//
// recipsqrt_rom.hex must be present in xsim_work/ before simulation.
// Numeric verification is done in C# by reading output.hex.

`timescale 1ns/1ps

module tb_adam_cell;

    localparam int STEPS   = 10;
    localparam int TIMEOUT = 5;   // cycles to wait for output after en pulse

    logic clk = 0;
    logic rst = 1;
    logic en  = 0;

    logic [31:0] g_fp32;
    logic [15:0] w_bf16;
    logic [31:0] m_fp32;
    logic [31:0] v_fp32;
    logic [31:0] lr_fp32;
    logic [31:0] bc1_fp32;
    logic [31:0] bc2_fp32;

    logic [15:0] w_bf16_out;
    logic [31:0] m_fp32_out;
    logic [31:0] v_fp32_out;

    adam_cell dut (
        .clk        (clk),
        .rst        (rst),
        .en         (en),
        .g_fp32     (g_fp32),
        .w_bf16     (w_bf16),
        .m_fp32     (m_fp32),
        .v_fp32     (v_fp32),
        .lr_fp32    (lr_fp32),
        .bc1_fp32   (bc1_fp32),
        .bc2_fp32   (bc2_fp32),
        .w_bf16_out (w_bf16_out),
        .m_fp32_out (m_fp32_out),
        .v_fp32_out (v_fp32_out)
    );

    always #5 clk = ~clk;

    logic [31:0] in_g   [0:STEPS-1];
    logic [15:0] in_w   [0:STEPS-1];
    logic [31:0] in_m   [0:STEPS-1];
    logic [31:0] in_v   [0:STEPS-1];
    logic [31:0] in_lr  [0:STEPS-1];
    logic [31:0] in_bc1 [0:STEPS-1];
    logic [31:0] in_bc2 [0:STEPS-1];

    logic [15:0] got_w  [0:STEPS-1];
    logic [31:0] got_m  [0:STEPS-1];
    logic [31:0] got_v  [0:STEPS-1];

    string  testvecs_dir;
    string  input_path, output_path, passfail_path;
    integer fd_in, fd_out, fd_pf, scan_ok;
    logic [31:0] tmp32;

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";
        input_path    = {testvecs_dir, "/adam_cell/input.hex"};
        output_path   = {testvecs_dir, "/adam_cell/output.hex"};
        passfail_path = {testvecs_dir, "/adam_cell/pass_fail.txt"};

        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin write_pf({"FAIL:cannot open ", input_path}); $finish; end

        for (int s = 0; s < STEPS; s++) begin
            scan_ok = $fscanf(fd_in, "%h\n", in_g  [s]); if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (g)");   $fclose(fd_in); $finish; end
            scan_ok = $fscanf(fd_in, "%h\n", tmp32);      if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (w)");   $fclose(fd_in); $finish; end
            in_w[s] = tmp32[15:0];
            scan_ok = $fscanf(fd_in, "%h\n", in_m  [s]); if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (m)");   $fclose(fd_in); $finish; end
            scan_ok = $fscanf(fd_in, "%h\n", in_v  [s]); if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (v)");   $fclose(fd_in); $finish; end
            scan_ok = $fscanf(fd_in, "%h\n", in_lr [s]); if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (lr)");  $fclose(fd_in); $finish; end
            scan_ok = $fscanf(fd_in, "%h\n", in_bc1[s]); if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (bc1)"); $fclose(fd_in); $finish; end
            scan_ok = $fscanf(fd_in, "%h\n", in_bc2[s]); if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (bc2)"); $fclose(fd_in); $finish; end
        end
        $fclose(fd_in);

        repeat(3) @(posedge clk); #1;
        rst = 0;
        @(posedge clk); #1;

        for (int s = 0; s < STEPS; s++) begin
            // Present inputs, assert en for one cycle
            g_fp32   = in_g  [s];
            w_bf16   = in_w  [s];
            m_fp32   = in_m  [s];
            v_fp32   = in_v  [s];
            lr_fp32  = in_lr [s];
            bc1_fp32 = in_bc1[s];
            bc2_fp32 = in_bc2[s];
            en = 1;
            @(posedge clk); #1;
            en = 0;
            // Outputs are registered -- available one cycle after the en edge
            got_w[s] = w_bf16_out;
            got_m[s] = m_fp32_out;
            got_v[s] = v_fp32_out;
        end

        fd_out = $fopen(output_path, "w");
        if (fd_out == 0) begin write_pf({"FAIL:cannot open ", output_path}); $finish; end
        for (int s = 0; s < STEPS; s++) begin
            $fwrite(fd_out, "%08h\n", {16'h0000, got_w[s]});  // zero-pad to 32 bits
            $fwrite(fd_out, "%08h\n", got_m[s]);
            $fwrite(fd_out, "%08h\n", got_v[s]);
        end
        $fclose(fd_out);

        write_pf("PASS");
        $finish;
    end

endmodule
