// Testbench for ce_grad (T=4, D=4, V=16)
//
// Reads:
//   $NEURO_TESTVECS/ce_grad/input.hex  (148 entries):
//     [0..63]    logits[T][V] FP32   -- t=0..3, v=0..15 (row-major, t outer)
//     [64..67]   targets[T]   int32  -- token index 0..V-1
//     [68..131]  emb[V][D]    FP32   -- v=0..15, d=0..3 (row-major, v outer)
//     [132..147] x[T][D]      FP32   -- t=0..3, d=0..3
//
// Writes:
//   $NEURO_TESTVECS/ce_grad/output.hex  (80 entries):
//     [0..15]  dX_init[T][D] FP32  -- t=0..3, d=0..3
//     [16..79] dWout  [V][D] FP32  -- v=0..15, d=0..3
//   $NEURO_TESTVECS/ce_grad/pass_fail.txt  -- "PASS" (numeric check done in C#)
//
// Numeric verification is done in C# (CeGradTests.cs) by comparing output.hex
// to expected.hex within VsSoftwareRelTol.

`timescale 1ns/1ps

module tb_ce_grad;

    localparam int T = 4;
    localparam int D = 4;
    localparam int V = 16;

    localparam int N_LOGITS  = T * V;   // 64
    localparam int N_TARGETS = T;       // 4
    localparam int N_EMB     = V * D;   // 64
    localparam int N_X       = T * D;   // 16
    localparam int N_TOTAL   = N_LOGITS + N_TARGETS + N_EMB + N_X; // 148

    localparam int N_DX_INIT = T * D;  // 16
    localparam int N_DWOUT   = V * D;  // 64
    localparam int N_OUT     = N_DX_INIT + N_DWOUT; // 80

    localparam int TIMEOUT = 5000;

    logic clk   = 0;
    logic rst   = 1;
    logic wr_en  = 0;
    logic [8:0]  wr_addr = '0;
    logic [31:0] wr_data = '0;
    logic start  = 0;
    logic done;

    logic [T*D*32-1:0] dX_init_out;
    logic [V*D*32-1:0] dWout_out;

    ce_grad #(.T(T), .D(D), .V(V)) dut (
        .clk         (clk),
        .rst         (rst),
        .wr_en       (wr_en),
        .wr_addr     (wr_addr),
        .wr_data     (wr_data),
        .start       (start),
        .done        (done),
        .dX_init_out (dX_init_out),
        .dWout_out   (dWout_out)
    );

    always #5 clk = ~clk;

    logic [31:0] input_data [0:N_TOTAL-1];

    string  testvecs_dir, input_path, output_path, passfail_path;
    integer fd_in, fd_out, fd_pf, scan_ok;
    int     timeout_cnt;

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

    task automatic write_reg(input logic [8:0] addr, input logic [31:0] data);
        @(posedge clk); #1;
        wr_en <= 1; wr_addr <= addr; wr_data <= data;
        @(posedge clk); #1;
        wr_en <= 0;
    endtask

    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";

        input_path    = {testvecs_dir, "/ce_grad/input.hex"};
        output_path   = {testvecs_dir, "/ce_grad/output.hex"};
        passfail_path = {testvecs_dir, "/ce_grad/pass_fail.txt"};

        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin
            $display("ERROR: cannot open %s", input_path);
            write_pf({"FAIL:cannot open ", input_path});
            $finish;
        end
        for (int i = 0; i < N_TOTAL; i++)
            scan_ok = $fscanf(fd_in, "%h\n", input_data[i]);
        $fclose(fd_in);

        // Reset
        repeat(4) @(posedge clk); #1;
        rst = 0;
        repeat(2) @(posedge clk); #1;

        // Load logits[T][V]: addr = t*V + v (0x000..0x03F)
        for (int t = 0; t < T; t++)
            for (int v = 0; v < V; v++)
                write_reg(9'(t*V + v), input_data[t*V + v]);

        // Load targets[T]: addr 0x040..0x043
        for (int t = 0; t < T; t++)
            write_reg(9'(9'h040 + t), input_data[N_LOGITS + t]);

        // Load emb[V][D]: addr 0x044 + v*D + d
        for (int v = 0; v < V; v++)
            for (int d = 0; d < D; d++)
                write_reg(9'(9'h044 + v*D + d), input_data[N_LOGITS + N_TARGETS + v*D + d]);

        // Load x[T][D]: addr 0x084 + t*D + d
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++)
                write_reg(9'(9'h084 + t*D + d), input_data[N_LOGITS + N_TARGETS + N_EMB + t*D + d]);

        repeat(2) @(posedge clk); #1;

        // Start
        @(posedge clk); #1; start <= 1;
        @(posedge clk); #1; start <= 0;

        timeout_cnt = 0;
        while (timeout_cnt < TIMEOUT) begin
            @(posedge clk); #1;
            if (done) begin timeout_cnt = -1; break; end
            timeout_cnt++;
        end

        if (timeout_cnt != -1) begin
            $display("ERROR: timeout waiting for done");
            write_pf("FAIL:timeout");
            $finish;
        end

        // Wait one extra cycle
        @(posedge clk); #1;

        // Write output.hex
        fd_out = $fopen(output_path, "w");
        if (fd_out == 0) begin
            write_pf({"FAIL:cannot open output ", output_path});
            $finish;
        end

        // dX_init[T][D]
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++)
                $fwrite(fd_out, "%08h\n", dX_init_out[(t*D+d)*32 +: 32]);

        // dWout[V][D]
        for (int v = 0; v < V; v++)
            for (int d = 0; d < D; d++)
                $fwrite(fd_out, "%08h\n", dWout_out[(v*D+d)*32 +: 32]);

        $fclose(fd_out);
        write_pf("PASS");
        $display("ce_grad: PASS");
        $finish;
    end

endmodule
