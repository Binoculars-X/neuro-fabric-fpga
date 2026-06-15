// Testbench for attention_core backward pass (T=4, D=4, DH=4, BF16 weights)
//
// Reads:
//   $NEURO_TESTVECS/attention_core_backward/input.hex
//       T*D    lines: X[T×D]     FP32 row-major
//       D*DH   lines: Wq[D×DH]   BF16 row-major
//       D*DH   lines: Wk[D×DH]   BF16 row-major
//       D*DH   lines: Wv[D×DH]   BF16 row-major
//       DH*D   lines: Wo[DH×D]   BF16 row-major
//       T*D    lines: dY[T×D]    FP32 row-major (upstream gradient)
//
//   exp_lut_init.hex must be present in xsim_work/
//
// Writes:
//   $NEURO_TESTVECS/attention_core_backward/output.hex
//       T*D    lines: dX[T×D]    FP32 row-major
//       D*DH   lines: dWq[D×DH]  FP32 row-major
//       D*DH   lines: dWk[D×DH]  FP32 row-major
//       D*DH   lines: dWv[D×DH]  FP32 row-major
//       DH*D   lines: dWo[DH×D]  FP32 row-major

`timescale 1ns/1ps

module tb_attention_core_backward;

    localparam int T  = 4;
    localparam int D  = 4;
    localparam int DH = 4;

    logic clk = 0;
    logic rst = 1;
    logic en  = 1;

    logic        x_wr_en   = 0; logic [7:0]  x_wr_addr  = 0; logic [31:0] x_wr_data  = 0;
    logic        wq_wr_en  = 0; logic [7:0]  wq_wr_addr = 0; logic [15:0] wq_wr_data = 0;
    logic        wk_wr_en  = 0; logic [7:0]  wk_wr_addr = 0; logic [15:0] wk_wr_data = 0;
    logic        wv_wr_en  = 0; logic [7:0]  wv_wr_addr = 0; logic [15:0] wv_wr_data = 0;
    logic        wo_wr_en  = 0; logic [7:0]  wo_wr_addr = 0; logic [15:0] wo_wr_data = 0;
    logic        dy_wr_en  = 0; logic [7:0]  dy_wr_addr = 0; logic [31:0] dy_wr_data = 0;

    logic start     = 0;
    logic bwd_start = 0;

    logic [D*32-1:0]      out_row;
    logic                  out_valid;
    logic [$clog2(T)-1:0] out_row_idx;

    logic [D*32-1:0]      dx_row;
    logic                  dx_valid;
    logic [$clog2(T)-1:0] dx_row_idx;
    logic [D*DH*32-1:0]   dWq_flat;
    logic [D*DH*32-1:0]   dWk_flat;
    logic [D*DH*32-1:0]   dWv_flat;
    logic [DH*D*32-1:0]   dWo_flat;

    attention_core #(
        .T(T), .D(D), .DH(DH), .MAC_LAT(3), .MUL_LAT(3),
        .EXP_LAT(4), .LUT_SIZE(256), .LUT_FILE("exp_lut_init.hex")
    ) dut (
        .clk(clk), .rst(rst), .en(en),
        .x_wr_en(x_wr_en),   .x_wr_addr(x_wr_addr),   .x_wr_data(x_wr_data),
        .wq_wr_en(wq_wr_en), .wq_wr_addr(wq_wr_addr), .wq_wr_data(wq_wr_data),
        .wk_wr_en(wk_wr_en), .wk_wr_addr(wk_wr_addr), .wk_wr_data(wk_wr_data),
        .wv_wr_en(wv_wr_en), .wv_wr_addr(wv_wr_addr), .wv_wr_data(wv_wr_data),
        .wo_wr_en(wo_wr_en), .wo_wr_addr(wo_wr_addr), .wo_wr_data(wo_wr_data),
        .start(start),
        .out_row(out_row), .out_valid(out_valid), .out_row_idx(out_row_idx),
        .dy_wr_en(dy_wr_en), .dy_wr_addr(dy_wr_addr), .dy_wr_data(dy_wr_data),
        .bwd_start(bwd_start),
        .dx_row(dx_row), .dx_valid(dx_valid), .dx_row_idx(dx_row_idx),
        .dWq_flat(dWq_flat), .dWk_flat(dWk_flat), .dWv_flat(dWv_flat), .dWo_flat(dWo_flat)
    );

    always #5 clk = ~clk;

    // -----------------------------------------------------------------------
    // Storage
    // -----------------------------------------------------------------------
    logic [31:0] in_X    [0:T*D-1];
    logic [15:0] in_Wq   [0:D*DH-1];
    logic [15:0] in_Wk   [0:D*DH-1];
    logic [15:0] in_Wv   [0:D*DH-1];
    logic [15:0] in_Wo   [0:DH*D-1];
    logic [31:0] in_dY   [0:T*D-1];
    logic [31:0] got_dx  [0:T*D-1];

    string  testvecs_dir;
    string  input_path, passfail_path, output_path;
    integer fd_in, fd_pf, fd_out, scan_ok;
    int     rows_done, cyc;

    task automatic write_pf(input string msg);
        fd_pf = $fopen(passfail_path, "w");
        if (fd_pf != 0) begin $fwrite(fd_pf, "%s\n", msg); $fclose(fd_pf); end
    endtask

    initial begin
        if (!$value$plusargs("NEURO_TESTVECS=%s", testvecs_dir))
            testvecs_dir = "../../run/fpga-testvecs";

        input_path    = {testvecs_dir, "/attention_core_backward/input.hex"};
        passfail_path = {testvecs_dir, "/attention_core_backward/pass_fail.txt"};
        output_path   = {testvecs_dir, "/attention_core_backward/output.hex"};

        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin
            write_pf({"FAIL:cannot open ", input_path});
            $finish;
        end
        for (int i = 0; i < T*D;  i++) scan_ok = $fscanf(fd_in, "%h\n", in_X[i]);
        for (int i = 0; i < D*DH; i++) scan_ok = $fscanf(fd_in, "%h\n", in_Wq[i]);
        for (int i = 0; i < D*DH; i++) scan_ok = $fscanf(fd_in, "%h\n", in_Wk[i]);
        for (int i = 0; i < D*DH; i++) scan_ok = $fscanf(fd_in, "%h\n", in_Wv[i]);
        for (int i = 0; i < DH*D; i++) scan_ok = $fscanf(fd_in, "%h\n", in_Wo[i]);
        for (int i = 0; i < T*D;  i++) scan_ok = $fscanf(fd_in, "%h\n", in_dY[i]);
        $fclose(fd_in);

        repeat(3) @(posedge clk); #1; rst = 0;

        // Load X
        for (int i = 0; i < T*D; i++) begin
            @(posedge clk); #1; x_wr_en = 1; x_wr_addr = i[7:0]; x_wr_data = in_X[i];
        end
        @(posedge clk); #1; x_wr_en = 0;

        // Load Wq
        for (int i = 0; i < D*DH; i++) begin
            @(posedge clk); #1; wq_wr_en = 1; wq_wr_addr = i[7:0]; wq_wr_data = in_Wq[i];
        end
        @(posedge clk); #1; wq_wr_en = 0;

        // Load Wk
        for (int i = 0; i < D*DH; i++) begin
            @(posedge clk); #1; wk_wr_en = 1; wk_wr_addr = i[7:0]; wk_wr_data = in_Wk[i];
        end
        @(posedge clk); #1; wk_wr_en = 0;

        // Load Wv
        for (int i = 0; i < D*DH; i++) begin
            @(posedge clk); #1; wv_wr_en = 1; wv_wr_addr = i[7:0]; wv_wr_data = in_Wv[i];
        end
        @(posedge clk); #1; wv_wr_en = 0;

        // Load Wo
        for (int i = 0; i < DH*D; i++) begin
            @(posedge clk); #1; wo_wr_en = 1; wo_wr_addr = i[7:0]; wo_wr_data = in_Wo[i];
        end
        @(posedge clk); #1; wo_wr_en = 0;

        // Load dY
        for (int i = 0; i < T*D; i++) begin
            @(posedge clk); #1; dy_wr_en = 1; dy_wr_addr = i[7:0]; dy_wr_data = in_dY[i];
        end
        @(posedge clk); #1; dy_wr_en = 0;

        // Run forward pass (needed to populate Q/K/V/scores/attn/attnOut registers)
        @(posedge clk); #1; start = 1;
        @(posedge clk); #1; start = 0;

        rows_done = 0;
        for (cyc = 0; cyc < 3000; cyc++) begin
            @(posedge clk); #1;
            if (out_valid) begin
                rows_done++;
                if (rows_done == T) break;
            end
        end
        if (rows_done != T) begin write_pf("FAIL:forward timed out"); $finish; end

        // One idle cycle then backward
        @(posedge clk); #1;
        bwd_start = 1;
        @(posedge clk); #1;
        bwd_start = 0;

        // Collect T dx rows
        rows_done = 0;
        for (cyc = 0; cyc < 8000; cyc++) begin
            @(posedge clk); #1;
            if (dx_valid) begin
                for (int j = 0; j < D; j++)
                    got_dx[int'(dx_row_idx) * D + j] = dx_row[j*32 +: 32];
                rows_done++;
                if (rows_done == T) break;
            end
        end
        if (rows_done != T) begin write_pf("FAIL:backward dx timed out"); $finish; end

        // Write output: dX, dWq, dWk, dWv, dWo
        fd_out = $fopen(output_path, "w");
        if (fd_out != 0) begin
            for (int i = 0; i < T*D; i++)
                $fwrite(fd_out, "%08h\n", got_dx[i]);
            for (int i = 0; i < D;  i++) for (int j = 0; j < DH; j++)
                $fwrite(fd_out, "%08h\n", dWq_flat[(i*DH+j)*32 +: 32]);
            for (int i = 0; i < D;  i++) for (int j = 0; j < DH; j++)
                $fwrite(fd_out, "%08h\n", dWk_flat[(i*DH+j)*32 +: 32]);
            for (int i = 0; i < D;  i++) for (int j = 0; j < DH; j++)
                $fwrite(fd_out, "%08h\n", dWv_flat[(i*DH+j)*32 +: 32]);
            for (int i = 0; i < DH; i++) for (int j = 0; j < D; j++)
                $fwrite(fd_out, "%08h\n", dWo_flat[(i*D+j)*32 +: 32]);
            $fclose(fd_out);
        end

        write_pf("PASS");
        $finish;
    end

endmodule
