// Testbench for adam_core (R=4, C=4, STEPS=100)
//
// Reads:
//   $NEURO_TESTVECS/adam_core/input.hex  -- per step (4*N + 3 lines):
//     N g_fp32 values (FP32, one per line)
//     N w_bf16 values (zero-padded to 32-bit)
//     N m_fp32 values
//     N v_fp32 values
//     lr_fp32
//     bc1_fp32
//     bc2_fp32
//
// Writes:
//   $NEURO_TESTVECS/adam_core/output.hex -- STEPS * 3*N lines:
//     N w_bf16_out (zero-padded to 32-bit)
//     N m_out (FP32)
//     N v_out (FP32)
//   $NEURO_TESTVECS/adam_core/pass_fail.txt -- "PASS" on success
//
// recipsqrt_rom.hex must be present in xsim_work/ before simulation.
// Numeric verification is done in C# by reading output.hex.

`timescale 1ns/1ps

module tb_adam_core;

    localparam int R     = 4;
    localparam int C     = 4;
    localparam int N     = R * C;      // 16
    localparam int STEPS = 100;
    localparam int TIMEOUT = N + 10;   // cycles to wait per step

    logic clk  = 0;
    logic rst  = 1;
    logic start = 0;

    logic [N*32-1:0] grad;
    logic [N*16-1:0] w_bf16_in;
    logic [N*32-1:0] m_in;
    logic [N*32-1:0] v_in;
    logic [31:0]     lr_fp32;
    logic [31:0]     bc1_fp32;
    logic [31:0]     bc2_fp32;

    logic [N*16-1:0] w_bf16_out;
    logic [N*32-1:0] m_out;
    logic [N*32-1:0] v_out;
    logic            done;

    adam_core #(.R(R), .C(C)) dut (
        .clk        (clk),
        .rst        (rst),
        .grad       (grad),
        .w_bf16_in  (w_bf16_in),
        .m_in       (m_in),
        .v_in       (v_in),
        .lr_fp32    (lr_fp32),
        .bc1_fp32   (bc1_fp32),
        .bc2_fp32   (bc2_fp32),
        .start      (start),
        .w_bf16_out (w_bf16_out),
        .m_out      (m_out),
        .v_out      (v_out),
        .done       (done)
    );

    always #5 clk = ~clk;

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
        input_path    = {testvecs_dir, "/adam_core/input.hex"};
        output_path   = {testvecs_dir, "/adam_core/output.hex"};
        passfail_path = {testvecs_dir, "/adam_core/pass_fail.txt"};

        // Release reset
        repeat(3) @(posedge clk); #1;
        rst = 0;
        @(posedge clk); #1;

        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin write_pf({"FAIL:cannot open ", input_path}); $finish; end

        fd_out = $fopen(output_path, "w");
        if (fd_out == 0) begin write_pf({"FAIL:cannot open ", output_path}); $fclose(fd_in); $finish; end

        for (int s = 0; s < STEPS; s++) begin

            // --- Read N grad values ---
            for (int i = 0; i < N; i++) begin
                scan_ok = $fscanf(fd_in, "%h\n", tmp32);
                if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (grad)"); $fclose(fd_in); $fclose(fd_out); $finish; end
                grad[i*32 +: 32] = tmp32;
            end

            // --- Read N w_bf16 values (zero-padded to 32) ---
            for (int i = 0; i < N; i++) begin
                scan_ok = $fscanf(fd_in, "%h\n", tmp32);
                if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (w)"); $fclose(fd_in); $fclose(fd_out); $finish; end
                w_bf16_in[i*16 +: 16] = tmp32[15:0];
            end

            // --- Read N m values ---
            for (int i = 0; i < N; i++) begin
                scan_ok = $fscanf(fd_in, "%h\n", tmp32);
                if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (m)"); $fclose(fd_in); $fclose(fd_out); $finish; end
                m_in[i*32 +: 32] = tmp32;
            end

            // --- Read N v values ---
            for (int i = 0; i < N; i++) begin
                scan_ok = $fscanf(fd_in, "%h\n", tmp32);
                if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (v)"); $fclose(fd_in); $fclose(fd_out); $finish; end
                v_in[i*32 +: 32] = tmp32;
            end

            // --- Read scalars ---
            scan_ok = $fscanf(fd_in, "%h\n", lr_fp32);  if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (lr)");  $fclose(fd_in); $fclose(fd_out); $finish; end
            scan_ok = $fscanf(fd_in, "%h\n", bc1_fp32); if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (bc1)"); $fclose(fd_in); $fclose(fd_out); $finish; end
            scan_ok = $fscanf(fd_in, "%h\n", bc2_fp32); if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (bc2)"); $fclose(fd_in); $fclose(fd_out); $finish; end

            // --- Pulse start ---
            start = 1;
            @(posedge clk); #1;
            start = 0;

            // --- Wait for done ---
            begin : wait_done
                integer wait_cnt;
                wait_cnt = 0;
                while (!done) begin
                    @(posedge clk); #1;
                    wait_cnt = wait_cnt + 1;
                    if (wait_cnt > TIMEOUT) begin
                        write_pf("FAIL:timeout waiting for done");
                        $fclose(fd_in); $fclose(fd_out);
                        $finish;
                    end
                end
            end

            // --- Write outputs (done=1 this cycle) ---
            for (int i = 0; i < N; i++)
                $fwrite(fd_out, "%08h\n", {16'h0000, w_bf16_out[i*16 +: 16]});
            for (int i = 0; i < N; i++)
                $fwrite(fd_out, "%08h\n", m_out[i*32 +: 32]);
            for (int i = 0; i < N; i++)
                $fwrite(fd_out, "%08h\n", v_out[i*32 +: 32]);

        end // for STEPS

        $fclose(fd_in);
        $fclose(fd_out);

        write_pf("PASS");
        $finish;
    end

endmodule
