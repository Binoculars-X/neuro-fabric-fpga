// Testbench for transformer (T=4, D=4, DH=4, FF=4, L=2, V=16)
//
// Reads:
//   $NEURO_TESTVECS/transformer/input.hex  (304 lines, one 32-bit hex per line)
//
// input.hex layout:
//   [  0.. 63]  emb_reg[v][d]     FP32  (addr 0x000..0x03F)
//   [ 64.. 79]  x_init[t][d]      FP32  (addr 0x040..0x04F)
//   Layer l=0  (112 entries, addr 0x050..0x0BF):
//     [  0..  3]  ln1_gamma[0][0..3] FP32
//     [  4..  7]  ln1_beta [0][0..3] FP32
//     [  8.. 11]  ln2_gamma[0][0..3] FP32
//     [ 12.. 15]  ln2_beta [0][0..3] FP32
//     [ 16.. 31]  Wq[0]  BF16 (16 entries, wr_data[15:0])
//     [ 32.. 47]  Wk[0]  BF16
//     [ 48.. 63]  Wv[0]  BF16
//     [ 64.. 79]  Wo[0]  BF16
//     [ 80.. 95]  Wff1[0] BF16
//     [ 96..111]  Wff2[0] BF16
//   Layer l=1  (112 entries, addr 0x0C0..0x12F): same structure
//   Total: 64+16+2*112 = 304 entries
//
// Writes:
//   $NEURO_TESTVECS/transformer/output.hex  -- T*V lines: logits[t][v] FP32
//   $NEURO_TESTVECS/transformer/pass_fail.txt
//
// ROMs required in xsim_work/: exp_lut_init.hex, recipsqrt_rom.hex

`timescale 1ns/1ps

module tb_transformer;

    localparam int T   = 4;
    localparam int D   = 4;
    localparam int DH  = 4;
    localparam int FF  = 4;
    localparam int L   = 2;
    localparam int V   = 16;

    localparam int N_EMB   = V * D;              // 64
    localparam int N_XINIT = T * D;              // 16
    localparam int N_LAYER = 4*D + 6*(D*DH);     // 16 + 96 = 112 per layer
    localparam int N_TOTAL = N_EMB + N_XINIT + L * N_LAYER;  // 304

    logic clk = 0;
    logic rst = 1;
    logic en  = 1;

    logic        wr_en   = 0;
    logic [8:0]  wr_addr = 0;
    logic [31:0] wr_data = 0;
    logic        start   = 0;

    logic [V*32-1:0] out_row;
    logic            out_valid;
    logic [1:0]      out_row_idx;
    logic            done;

    transformer #(
        .T(T), .D(D), .DH(DH), .FF(FF), .L(L), .V(V),
        .MAC_LAT(3), .MUL_LAT(3), .EXP_LAT(4),
        .LUT_SIZE(256), .LUT_FILE("exp_lut_init.hex")
    ) dut (
        .clk(clk), .rst(rst), .en(en),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data),
        .start(start),
        .out_row(out_row), .out_valid(out_valid), .out_row_idx(out_row_idx),
        .done(done)
    );

    always #5 clk = ~clk;

    logic [31:0] input_data [0:N_TOTAL-1];
    logic [31:0] got_out    [0:T*V-1];

    string  testvecs_dir, input_path, output_path, passfail_path;
    integer fd_in, fd_out, fd_pf, scan_ok;
    int     timeout_cnt;

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";

        input_path    = {testvecs_dir, "/transformer/input.hex"};
        output_path   = {testvecs_dir, "/transformer/output.hex"};
        passfail_path = {testvecs_dir, "/transformer/pass_fail.txt"};

        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin
            $display("ERROR: cannot open %s", input_path);
            write_pf({"FAIL:cannot open ", input_path});
            $finish;
        end
        for (int i = 0; i < N_TOTAL; i++)
            scan_ok = $fscanf(fd_in, "%h\n", input_data[i]);
        $fclose(fd_in);

        repeat(4) @(posedge clk); #1;
        rst = 0;
        repeat(2) @(posedge clk); #1;

        // Load emb_reg (addr 0x000..0x03F)
        for (int i = 0; i < N_EMB; i++) begin
            @(posedge clk); #1;
            wr_en <= 1; wr_addr <= 9'(i); wr_data <= input_data[i];
        end
        @(posedge clk); #1; wr_en <= 0;

        // Load x_init (addr 0x040..0x04F)
        for (int i = 0; i < N_XINIT; i++) begin
            @(posedge clk); #1;
            wr_en <= 1; wr_addr <= 9'(9'h040 + i); wr_data <= input_data[N_EMB + i];
        end
        @(posedge clk); #1; wr_en <= 0;

        // Load layer 0 (addr 0x050..0x0BF, 112 entries)
        for (int i = 0; i < N_LAYER; i++) begin
            @(posedge clk); #1;
            wr_en <= 1; wr_addr <= 9'(9'h050 + i); wr_data <= input_data[N_EMB + N_XINIT + i];
        end
        @(posedge clk); #1; wr_en <= 0;

        // Load layer 1 (addr 0x0C0..0x12F, 112 entries)
        for (int i = 0; i < N_LAYER; i++) begin
            @(posedge clk); #1;
            wr_en <= 1; wr_addr <= 9'(9'h0C0 + i); wr_data <= input_data[N_EMB + N_XINIT + N_LAYER + i];
        end
        @(posedge clk); #1; wr_en <= 0;

        repeat(2) @(posedge clk); #1;

        // Pulse start
        @(posedge clk); #1; start <= 1;
        @(posedge clk); #1; start <= 0;

        // Collect output rows
        timeout_cnt = 0;
        while (timeout_cnt < 20000) begin
            @(posedge clk); #1;
            if (out_valid)
                for (int v = 0; v < V; v++)
                    got_out[out_row_idx * V + v] = out_row[v*32+:32];
            if (done) begin
                timeout_cnt = -1; // signal success
                break;
            end
            timeout_cnt = timeout_cnt + 1;
        end

        if (timeout_cnt != -1) begin
            $display("ERROR: timeout waiting for done after %0d cycles", timeout_cnt);
            write_pf("FAIL:timeout");
            $finish;
        end

        fd_out = $fopen(output_path, "w");
        if (fd_out == 0) begin write_pf({"FAIL:cannot open output ", output_path}); $finish; end
        for (int i = 0; i < T*V; i++)
            $fwrite(fd_out, "%08x\n", got_out[i]);
        $fclose(fd_out);

        write_pf("PASS");
        $display("tb_transformer: PASS (output written to %s)", output_path);
        $finish;
    end

endmodule
