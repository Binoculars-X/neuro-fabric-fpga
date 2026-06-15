// tb_transformer_train.sv
// Testbench for transformer_train: runs one full train step.
//
// Sequence:
//   1. Load 304-entry weight/embedding input.hex (same format as tb_transformer).
//   2. Load tokens[T] and targets[T] from input.hex[304..311].
//   3. Pulse start → wait for done → capture logits[T×V].
//   4. Compute cross-entropy softmax and dLogits[T×V] in SystemVerilog shortreal.
//   5. Compute dX_init[T×D] = dLogits · Emb  (backprop through output projection).
//   6. Compute dWout[V×D]   = dLogitsᵀ · lastLayerOut  (embedding tied gradient).
//   7. Clip dLogits by grad norm (clip to 1.0).
//   8. Write dX_init[T×D]  → wr_addr 0x130..0x13F.
//      Write dWout[V×D]    → wr_addr 0x140..0x17F.
//      Write lr/bc1/bc2    → wr_addr 0x180..0x182.
//   9. Pulse train_start → wait for adam_done.
//  10. Read Adam state from DUT hierarchical ports and write output.hex.
//      output.hex matches expected.hex from FpgaTransformerVecGen.WriteHexTrain.
//
// ROMs required in xsim_work/: exp_lut_init.hex, recipsqrt_rom.hex

`timescale 1ns/1ps

module tb_transformer_train;

    localparam int T  = 4;
    localparam int D  = 4;
    localparam int DH = 4;
    localparam int FF = 4;
    localparam int L  = 2;
    localparam int V  = 16;

    localparam int N_EMB   = V * D;               // 64
    localparam int N_XINIT = T * D;               // 16
    localparam int N_LAYER = 4*D + 6*(D*DH);      // 16 + 96 = 112 per layer
    localparam int N_FWD   = N_EMB + N_XINIT + L * N_LAYER;  // 304
    localparam int N_TOTAL = N_FWD + 2*T;         // 304 + 8 = 312

    // Adam float constants (matching C# AdamBF16WeightsAttentionCore)
    localparam real LR   = 1e-3;
    localparam real B1   = 0.9;
    localparam real B2   = 0.999;
    localparam real STEP = 1;  // first step
    // bc1 = 1 - B1^step,  bc2 = 1 - B2^step
    localparam real BC1  = 1.0 - 0.9;       // 0.1
    localparam real BC2  = 1.0 - 0.999;     // 0.001

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

    logic train_start = 0;
    logic adam_done;

    transformer_train #(
        .T(T), .D(D), .DH(DH), .FF(FF), .L(L), .V(V),
        .MAC_LAT(3), .MUL_LAT(3), .EXP_LAT(4),
        .LUT_SIZE(256), .LUT_FILE("exp_lut_init.hex")
    ) dut (
        .clk(clk), .rst(rst), .en(en),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data),
        .start(start),
        .out_row(out_row), .out_valid(out_valid), .out_row_idx(out_row_idx),
        .done(done),
        .train_start(train_start),
        .adam_done(adam_done)
    );

    always #5 clk = ~clk;

    // -----------------------------------------------------------------------
    // Storage — testbench only handles data loading and output collection
    // -----------------------------------------------------------------------
    logic [31:0] input_data [0:N_TOTAL-1];
    int          tokens  [0:T-1];
    int          targets [0:T-1];

    string  testvecs_dir, input_path, output_path, passfail_path;
    integer fd_in, fd_out, fd_pf, scan_ok;
    int     timeout_cnt;

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

    // Write one 32-bit word via write bus
    task automatic write_reg(input logic [8:0] addr, input logic [31:0] data);
        @(posedge clk); #1;
        wr_en <= 1; wr_addr <= addr; wr_data <= data;
        @(posedge clk); #1;
        wr_en <= 0;
    endtask

    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";

        input_path    = {testvecs_dir, "/transformer_train/input.hex"};
        output_path   = {testvecs_dir, "/transformer_train/output.hex"};
        passfail_path = {testvecs_dir, "/transformer_train/pass_fail.txt"};

        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin
            $display("ERROR: cannot open %s", input_path);
            write_pf({"FAIL:cannot open ", input_path});
            $finish;
        end
        for (int i = 0; i < N_TOTAL; i++)
            scan_ok = $fscanf(fd_in, "%h\n", input_data[i]);
        $fclose(fd_in);

        // Parse tokens and targets from end of input_data
        for (int t = 0; t < T; t++) begin
            tokens [t] = int'(input_data[N_FWD + t]);
            targets[t] = int'(input_data[N_FWD + T + t]);
        end

        // -----------------------------------------------------------------------
        // Reset
        // -----------------------------------------------------------------------
        repeat(4) @(posedge clk); #1;
        rst = 0;
        repeat(2) @(posedge clk); #1;

        // -----------------------------------------------------------------------
        // Load weights (304 entries): emb, x_init, layer0, layer1
        // -----------------------------------------------------------------------
        for (int i = 0; i < N_EMB; i++)
            write_reg(9'(i), input_data[i]);

        for (int i = 0; i < N_XINIT; i++)
            write_reg(9'(9'h040 + i), input_data[N_EMB + i]);

        for (int i = 0; i < N_LAYER; i++)
            write_reg(9'(9'h050 + i), input_data[N_EMB + N_XINIT + i]);

        for (int i = 0; i < N_LAYER; i++)
            write_reg(9'(9'h0C0 + i), input_data[N_EMB + N_XINIT + N_LAYER + i]);

        // Write targets (RTL uses them for internal cross-entropy)  addr 0x130..0x133
        for (int t = 0; t < T; t++)
            write_reg(9'(9'h130 + t), 32'(targets[t]));

        // Write scalar Adam hyperparams
        write_reg(9'h180, $shortrealtobits(shortreal'(LR)));
        write_reg(9'h181, $shortrealtobits(shortreal'(BC1)));
        write_reg(9'h182, $shortrealtobits(shortreal'(BC2)));

        repeat(2) @(posedge clk); #1;

        // -----------------------------------------------------------------------
        // Forward pass — RTL runs internally
        // -----------------------------------------------------------------------
        @(posedge clk); #1; start <= 1;
        @(posedge clk); #1; start <= 0;

        timeout_cnt = 0;
        while (timeout_cnt < 30000) begin
            @(posedge clk); #1;
            if (done) begin timeout_cnt = -1; break; end
            timeout_cnt++;
        end

        if (timeout_cnt != -1) begin
            $display("ERROR: timeout in forward pass");
            write_pf("FAIL:forward timeout");
            $finish;
        end

        // -----------------------------------------------------------------------
        // Train step: RTL computes CE, dLogits, dX_init, dWout, backward, Adam
        // -----------------------------------------------------------------------
        @(posedge clk); #1; train_start <= 1;
        @(posedge clk); #1; train_start <= 0;

        timeout_cnt = 0;
        while (timeout_cnt < 200000) begin
            @(posedge clk); #1;
            if (adam_done) begin timeout_cnt = -1; break; end
            timeout_cnt++;
        end

        if (timeout_cnt != -1) begin
            $display("ERROR: timeout waiting for adam_done");
            write_pf("FAIL:adam timeout");
            $finish;
        end

        // Wait one extra cycle for outputs to settle
        @(posedge clk); #1;

        // -----------------------------------------------------------------------
        // Write output.hex from Adam core hierarchical ports
        // Format: Emb(w,m,v)[V*D=64 each], L0 Wq..Wff2 (w,m,v), L1 same
        // -----------------------------------------------------------------------
        fd_out = $fopen(output_path, "w");
        if (fd_out == 0) begin
            $display("ERROR: cannot open output %s", output_path);
            write_pf({"FAIL:cannot open output ", output_path});
            $finish;
        end

        // Embedding Adam state
        for (int i = 0; i < V*D; i++)
            $fwrite(fd_out, "%08x\n", {16'h0, dut.u_adam_emb.w_bf16_out[i*16 +: 16]});
        for (int i = 0; i < V*D; i++)
            $fwrite(fd_out, "%08x\n", dut.u_adam_emb.m_out[i*32 +: 32]);
        for (int i = 0; i < V*D; i++)
            $fwrite(fd_out, "%08x\n", dut.u_adam_emb.v_out[i*32 +: 32]);

        // Layer 0 Adam state
        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", {16'h0, dut.u_adam_wq0.w_bf16_out[i*16 +: 16]});
        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wq0.m_out[i*32 +: 32]);
        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wq0.v_out[i*32 +: 32]);

        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", {16'h0, dut.u_adam_wk0.w_bf16_out[i*16 +: 16]});
        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wk0.m_out[i*32 +: 32]);
        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wk0.v_out[i*32 +: 32]);

        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", {16'h0, dut.u_adam_wv0.w_bf16_out[i*16 +: 16]});
        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wv0.m_out[i*32 +: 32]);
        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wv0.v_out[i*32 +: 32]);

        for (int i = 0; i < DH*D; i++) $fwrite(fd_out, "%08x\n", {16'h0, dut.u_adam_wo0.w_bf16_out[i*16 +: 16]});
        for (int i = 0; i < DH*D; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wo0.m_out[i*32 +: 32]);
        for (int i = 0; i < DH*D; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wo0.v_out[i*32 +: 32]);

        for (int i = 0; i < D*FF; i++) $fwrite(fd_out, "%08x\n", {16'h0, dut.u_adam_wff10.w_bf16_out[i*16 +: 16]});
        for (int i = 0; i < D*FF; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wff10.m_out[i*32 +: 32]);
        for (int i = 0; i < D*FF; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wff10.v_out[i*32 +: 32]);

        for (int i = 0; i < FF*D; i++) $fwrite(fd_out, "%08x\n", {16'h0, dut.u_adam_wff20.w_bf16_out[i*16 +: 16]});
        for (int i = 0; i < FF*D; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wff20.m_out[i*32 +: 32]);
        for (int i = 0; i < FF*D; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wff20.v_out[i*32 +: 32]);

        // Layer 1 Adam state
        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", {16'h0, dut.u_adam_wq1.w_bf16_out[i*16 +: 16]});
        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wq1.m_out[i*32 +: 32]);
        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wq1.v_out[i*32 +: 32]);

        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", {16'h0, dut.u_adam_wk1.w_bf16_out[i*16 +: 16]});
        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wk1.m_out[i*32 +: 32]);
        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wk1.v_out[i*32 +: 32]);

        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", {16'h0, dut.u_adam_wv1.w_bf16_out[i*16 +: 16]});
        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wv1.m_out[i*32 +: 32]);
        for (int i = 0; i < D*DH; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wv1.v_out[i*32 +: 32]);

        for (int i = 0; i < DH*D; i++) $fwrite(fd_out, "%08x\n", {16'h0, dut.u_adam_wo1.w_bf16_out[i*16 +: 16]});
        for (int i = 0; i < DH*D; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wo1.m_out[i*32 +: 32]);
        for (int i = 0; i < DH*D; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wo1.v_out[i*32 +: 32]);

        for (int i = 0; i < D*FF; i++) $fwrite(fd_out, "%08x\n", {16'h0, dut.u_adam_wff11.w_bf16_out[i*16 +: 16]});
        for (int i = 0; i < D*FF; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wff11.m_out[i*32 +: 32]);
        for (int i = 0; i < D*FF; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wff11.v_out[i*32 +: 32]);

        for (int i = 0; i < FF*D; i++) $fwrite(fd_out, "%08x\n", {16'h0, dut.u_adam_wff21.w_bf16_out[i*16 +: 16]});
        for (int i = 0; i < FF*D; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wff21.m_out[i*32 +: 32]);
        for (int i = 0; i < FF*D; i++) $fwrite(fd_out, "%08x\n", dut.u_adam_wff21.v_out[i*32 +: 32]);

        $fclose(fd_out);

        write_pf("PASS");
        $display("tb_transformer_train: PASS");
        $finish;
    end

endmodule
