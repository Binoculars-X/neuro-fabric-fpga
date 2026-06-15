// Testbench for layernorm backward pass (D=4, T=4 token rows)
//
// Protocol:
//   1. Reset, load recipsqrt_rom.hex into fp32_sqrt's ROM.
//   2. Write T rows of (X, gamma, beta) via x_in/gamma/beta + start pulse
//      (same as forward testbench — drives the forward FSM to latch mean/var/x).
//   3. Write T*D values of dY via dy_wr_en/dy_wr_addr/dy_wr_data.
//   4. Assert bwd_start for 1 cycle.
//   5. Collect T dx_row outputs when dx_valid fires; also read dGamma_flat/dBeta_flat.
//
// Reads:
//   $NEURO_TESTVECS/layernorm_backward/input.hex:
//     T*(3*D) lines (forward: D FP32 X + D FP32 gamma + D FP32 beta per row)
//     T*D     lines (dY FP32, row-major)
//
// Writes:
//   $NEURO_TESTVECS/layernorm_backward/output.hex:
//     T*D FP32 lines — dX (row-major)
//     D   FP32 lines — dGamma
//     D   FP32 lines — dBeta
//   $NEURO_TESTVECS/layernorm_backward/pass_fail.txt

`timescale 1ns/1ps

module tb_layernorm_backward;

    localparam int D       = 4;
    localparam int T       = 4;
    localparam int TIMEOUT = 200;

    logic clk = 0;
    logic rst = 1;
    logic en  = 1;

    // Forward ports
    logic [D*32-1:0] x_in;
    logic [D*32-1:0] gamma_in;
    logic [D*32-1:0] beta_in;
    logic             start;
    logic [D*32-1:0] y_out;
    logic             out_valid;

    // Backward ports
    logic             dy_wr_en;
    logic [7:0]       dy_wr_addr;
    logic [31:0]      dy_wr_data;
    logic             bwd_start;
    logic [D*32-1:0]  dx_row;
    logic             dx_valid;
    logic [$clog2(T)-1:0] dx_row_idx;
    logic [D*32-1:0]  dGamma_flat;
    logic [D*32-1:0]  dBeta_flat;

    layernorm #(.D(D), .T(T)) dut (
        .clk        (clk),
        .rst        (rst),
        .en         (en),
        .x_in       (x_in),
        .gamma      (gamma_in),
        .beta       (beta_in),
        .start      (start),
        .y_out      (y_out),
        .out_valid  (out_valid),
        .dy_wr_en   (dy_wr_en),
        .dy_wr_addr (dy_wr_addr),
        .dy_wr_data (dy_wr_data),
        .bwd_start  (bwd_start),
        .dx_row     (dx_row),
        .dx_valid   (dx_valid),
        .dx_row_idx (dx_row_idx),
        .dGamma_flat(dGamma_flat),
        .dBeta_flat (dBeta_flat)
    );

    always #5 clk = ~clk;

    logic [31:0] in_x    [0:T-1][0:D-1];
    logic [31:0] in_gamma[0:T-1][0:D-1];
    logic [31:0] in_beta [0:T-1][0:D-1];
    logic [31:0] in_dy   [0:T-1][0:D-1];
    logic [31:0] got_dx  [0:T-1][0:D-1];

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
        input_path    = {testvecs_dir, "/layernorm_backward/input.hex"};
        output_path   = {testvecs_dir, "/layernorm_backward/output.hex"};
        passfail_path = {testvecs_dir, "/layernorm_backward/pass_fail.txt"};

        // Read input.hex
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
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++) begin
                scan_ok = $fscanf(fd_in, "%h\n", in_dy[t][d]);
                if (scan_ok != 1) begin write_pf("FAIL:input.hex malformed (dy)"); $fclose(fd_in); $finish; end
            end
        $fclose(fd_in);

        // Defaults
        start      = 1'b0;
        dy_wr_en   = 1'b0;
        dy_wr_addr = 8'h0;
        dy_wr_data = 32'h0;
        bwd_start  = 1'b0;
        x_in       = '0;
        gamma_in   = '0;
        beta_in    = '0;

        // Release reset
        @(posedge clk); #1;
        @(posedge clk); #1;
        rst = 0;
        @(posedge clk); #1;

        // ── Phase 1: run T forward rows to latch mean/var/x ──────────────
        for (int t = 0; t < T; t++) begin
            for (int d = 0; d < D; d++) begin
                x_in    [d*32 +: 32] = in_x    [t][d];
                gamma_in[d*32 +: 32] = in_gamma[t][d];
                beta_in [d*32 +: 32] = in_beta [t][d];
            end
            start = 1'b1;
            @(posedge clk); #1;
            start = 1'b0;

            // Wait for out_valid
            cyc = 0;
            while (!out_valid) begin
                @(posedge clk); #1;
                cyc = cyc + 1;
                if (cyc > TIMEOUT) begin
                    write_pf("FAIL:forward timeout");
                    $fclose(fd_pf);
                    $finish;
                end
            end
            @(posedge clk); #1;  // deassert out_valid
        end

        // ── Phase 2: write dY via dy_wr_* port ───────────────────────────
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++) begin
                dy_wr_en   = 1'b1;
                dy_wr_addr = t * D + d;
                dy_wr_data = in_dy[t][d];
                @(posedge clk); #1;
            end
        dy_wr_en = 1'b0;

        // ── Phase 3: assert bwd_start ─────────────────────────────────────
        bwd_start = 1'b1;
        @(posedge clk); #1;
        bwd_start = 1'b0;

        // ── Phase 4: collect T dx_row outputs ────────────────────────────
        rows_done = 0;
        cyc       = 0;
        while (rows_done < T) begin
            @(posedge clk); #1;
            cyc = cyc + 1;
            if (cyc > TIMEOUT) begin
                write_pf("FAIL:backward timeout");
                $fclose(fd_pf);
                $finish;
            end
            if (dx_valid) begin
                for (int d = 0; d < D; d++)
                    got_dx[int'(dx_row_idx)][d] = dx_row[d*32 +: 32];
                rows_done = rows_done + 1;
            end
        end
        @(posedge clk); #1;

        // ── Phase 5: write output.hex ─────────────────────────────────────
        fd_out = $fopen(output_path, "w");
        if (fd_out == 0) begin write_pf({"FAIL:cannot open output ", output_path}); $finish; end

        // dX [T×D]
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++)
                $fwrite(fd_out, "%08X\n", got_dx[t][d]);

        // dGamma [D] — stable after backward completes
        for (int d = 0; d < D; d++)
            $fwrite(fd_out, "%08X\n", dGamma_flat[d*32 +: 32]);

        // dBeta [D]
        for (int d = 0; d < D; d++)
            $fwrite(fd_out, "%08X\n", dBeta_flat[d*32 +: 32]);

        $fclose(fd_out);

        write_pf("PASS");
        $finish;
    end

endmodule
