// Testbench for mlp_core backward pass (T=4, D=4, FF=4, BF16 weights)
//
// Reads:
//   $NEURO_TESTVECS/mlp_core_backward/input.hex
//       T*D  lines: X[T×D]     FP32 row-major
//       D*FF lines: Wff1[D×FF] BF16 row-major
//       FF*D lines: Wff2[FF×D] BF16 row-major
//       T*D  lines: dY[T×D]    FP32 row-major (upstream gradient)
//
//   exp_lut_init.hex must be present in xsim_work/
//
// Writes:
//   $NEURO_TESTVECS/mlp_core_backward/output.hex
//       T*D   lines: dX[T×D]    FP32 row-major
//       D*FF  lines: dWff1[D×FF] FP32 row-major
//       FF*D  lines: dWff2[FF×D] FP32 row-major
//
// Numeric verification is done in C# by reading output.hex.

`timescale 1ns/1ps

module tb_mlp_core_backward;

    localparam int T  = 4;
    localparam int D  = 4;
    localparam int FF = 4;

    logic clk = 0;
    logic rst = 1;
    logic en  = 1;

    logic        x_wr_en   = 0; logic [7:0]  x_wr_addr   = 0; logic [31:0] x_wr_data   = 0;
    logic        wff1_wr_en = 0; logic [7:0] wff1_wr_addr = 0; logic [15:0] wff1_wr_data = 0;
    logic        wff2_wr_en = 0; logic [7:0] wff2_wr_addr = 0; logic [15:0] wff2_wr_data = 0;
    logic        dy_wr_en   = 0; logic [7:0]  dy_wr_addr  = 0; logic [31:0] dy_wr_data   = 0;

    logic start     = 0;
    logic bwd_start = 0;

    logic [D*32-1:0]      out_row;
    logic                  out_valid;
    logic [$clog2(T)-1:0] out_row_idx;

    logic [D*32-1:0]      dx_row;
    logic                  dx_valid;
    logic [$clog2(T)-1:0] dx_row_idx;
    logic [D*FF*32-1:0]   dWff1_flat;
    logic [FF*D*32-1:0]   dWff2_flat;

    mlp_core #(
        .T       (T),
        .D       (D),
        .FF      (FF),
        .MAC_LAT (3),
        .EXP_LAT (4),
        .LUT_SIZE(256),
        .LUT_FILE("exp_lut_init.hex")
    ) dut (
        .clk          (clk),
        .rst          (rst),
        .en           (en),
        .x_wr_en      (x_wr_en),    .x_wr_addr    (x_wr_addr),    .x_wr_data    (x_wr_data),
        .wff1_wr_en   (wff1_wr_en), .wff1_wr_addr (wff1_wr_addr), .wff1_wr_data (wff1_wr_data),
        .wff2_wr_en   (wff2_wr_en), .wff2_wr_addr (wff2_wr_addr), .wff2_wr_data (wff2_wr_data),
        .start        (start),
        .out_row      (out_row),
        .out_valid    (out_valid),
        .out_row_idx  (out_row_idx),
        .dy_wr_en     (dy_wr_en),   .dy_wr_addr   (dy_wr_addr),   .dy_wr_data   (dy_wr_data),
        .bwd_start    (bwd_start),
        .dx_row       (dx_row),
        .dx_valid     (dx_valid),
        .dx_row_idx   (dx_row_idx),
        .dWff1_flat   (dWff1_flat),
        .dWff2_flat   (dWff2_flat)
    );

    always #5 clk = ~clk;

    // -----------------------------------------------------------------------
    // Input / output storage
    // -----------------------------------------------------------------------
    logic [31:0] in_X    [0:T*D-1];
    logic [15:0] in_Wff1 [0:D*FF-1];
    logic [15:0] in_Wff2 [0:FF*D-1];
    logic [31:0] in_dY   [0:T*D-1];
    logic [31:0] got_dx   [0:T*D-1];

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

        input_path    = {testvecs_dir, "/mlp_core_backward/input.hex"};
        passfail_path = {testvecs_dir, "/mlp_core_backward/pass_fail.txt"};
        output_path   = {testvecs_dir, "/mlp_core_backward/output.hex"};

        fd_in = $fopen(input_path, "r");
        if (fd_in == 0) begin
            write_pf({"FAIL:cannot open ", input_path});
            $finish;
        end
        for (int i = 0; i < T*D;  i++) begin scan_ok = $fscanf(fd_in, "%h\n", in_X[i]);    end
        for (int i = 0; i < D*FF; i++) begin scan_ok = $fscanf(fd_in, "%h\n", in_Wff1[i]); end
        for (int i = 0; i < FF*D; i++) begin scan_ok = $fscanf(fd_in, "%h\n", in_Wff2[i]); end
        for (int i = 0; i < T*D;  i++) begin scan_ok = $fscanf(fd_in, "%h\n", in_dY[i]);   end
        $fclose(fd_in);

        // Reset
        repeat(3) @(posedge clk);
        #1; rst = 0;

        // Load X
        for (int i = 0; i < T*D; i++) begin
            @(posedge clk); #1;
            x_wr_en = 1; x_wr_addr = i[7:0]; x_wr_data = in_X[i];
        end
        @(posedge clk); #1; x_wr_en = 0;

        // Load Wff1
        for (int i = 0; i < D*FF; i++) begin
            @(posedge clk); #1;
            wff1_wr_en = 1; wff1_wr_addr = i[7:0]; wff1_wr_data = in_Wff1[i];
        end
        @(posedge clk); #1; wff1_wr_en = 0;

        // Load Wff2
        for (int i = 0; i < FF*D; i++) begin
            @(posedge clk); #1;
            wff2_wr_en = 1; wff2_wr_addr = i[7:0]; wff2_wr_data = in_Wff2[i];
        end
        @(posedge clk); #1; wff2_wr_en = 0;

        // Load dY (upstream gradient)
        for (int i = 0; i < T*D; i++) begin
            @(posedge clk); #1;
            dy_wr_en = 1; dy_wr_addr = i[7:0]; dy_wr_data = in_dY[i];
        end
        @(posedge clk); #1; dy_wr_en = 0;

        // Pulse start — run forward pass to compute H1, G and retain them
        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        // Wait for forward to complete (all T rows out_valid)
        rows_done = 0;
        for (cyc = 0; cyc < 2000; cyc++) begin
            @(posedge clk); #1;
            if (out_valid) begin
                rows_done++;
                if (rows_done == T) break;
            end
        end

        if (rows_done != T) begin
            write_pf("FAIL:forward timed out");
            $finish;
        end

        // Give one idle cycle, then pulse bwd_start
        @(posedge clk); #1;
        bwd_start = 1;
        @(posedge clk); #1;
        bwd_start = 0;

        // Collect T dx_row outputs
        rows_done = 0;
        for (cyc = 0; cyc < 4000; cyc++) begin
            @(posedge clk); #1;
            if (dx_valid) begin
                for (int j = 0; j < D; j++)
                    got_dx[int'(dx_row_idx) * D + j] = dx_row[j*32 +: 32];
                rows_done++;
                if (rows_done == T) break;
            end
        end

        if (rows_done != T) begin
            write_pf("FAIL:backward dx timed out");
            $finish;
        end

        // Write output: dX then dWff1 then dWff2
        fd_out = $fopen(output_path, "w");
        if (fd_out != 0) begin
            // dX [T×D]
            for (int i = 0; i < T*D; i++)
                $fwrite(fd_out, "%08h\n", got_dx[i]);
            // dWff1 [D×FF] — available combinationally on dWff1_flat after backward
            for (int i = 0; i < D; i++)
                for (int j = 0; j < FF; j++)
                    $fwrite(fd_out, "%08h\n", dWff1_flat[(i*FF+j)*32 +: 32]);
            // dWff2 [FF×D]
            for (int i = 0; i < FF; i++)
                for (int j = 0; j < D; j++)
                    $fwrite(fd_out, "%08h\n", dWff2_flat[(i*D+j)*32 +: 32]);
            $fclose(fd_out);
        end

        write_pf("PASS");
        $finish;
    end

endmodule
