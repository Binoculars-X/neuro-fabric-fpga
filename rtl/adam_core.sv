// Adam matrix update core — BF16-weights variant
//
// Applies adam_cell to an entire R×C weight matrix, one parameter per clock cycle.
// Matches AdamBF16WeightsAttentionCore.ApplyUpdate() for all R*C parameters.
//
// Interface:
//   grad      [R*C*32-1:0] — FP32 gradients, flat row-major  (param i at [i*32+:32])
//   w_bf16_in [R*C*16-1:0] — BF16 weights in                 (param i at [i*16+:16])
//   m_in      [R*C*32-1:0] — FP32 first moments in
//   v_in      [R*C*32-1:0] — FP32 second moments in
//   lr_fp32, bc1_fp32, bc2_fp32 — scalar per-step inputs
//   start  — pulse high for one cycle to begin a step
//   done   — pulses high for one cycle when all R*C updates are complete
//   w_bf16_out, m_out, v_out — updated values (valid when done=1, held until next start)
//
// FSM: IDLE → UPDATE (R*C+1 cycles, one adam_cell instance) → DONE → IDLE
//
// Timing: from start to done = R*C + 3 cycles.
//   IDLE  (1): start captured, enter UPDATE, idx=0
//   UPDATE (R*C+1): idx 0..R*C; cell driven at idx<R*C; output[i] captured at idx=i+1
//   DONE  (1): done=1 for one cycle
//
// Dependencies: adam_cell.sv (which requires fp32_div.sv, fp32_sqrt.sv)
// C# reference: AdamBF16WeightsAttentionCore.ApplyUpdate()
// XSim tolerance: VsSoftwareRelTol = 0.01%

`timescale 1ns/1ps

module adam_core #(
    parameter int R = 4,
    parameter int C = 4
)(
    input  logic clk,
    input  logic rst,

    input  logic [R*C*32-1:0] grad,        // FP32 gradients, flat
    input  logic [R*C*16-1:0] w_bf16_in,   // BF16 weights in
    input  logic [R*C*32-1:0] m_in,        // FP32 first moments in
    input  logic [R*C*32-1:0] v_in,        // FP32 second moments in
    input  logic [31:0]        lr_fp32,
    input  logic [31:0]        bc1_fp32,   // 1 - Beta1^t  (caller computes)
    input  logic [31:0]        bc2_fp32,   // 1 - Beta2^t

    input  logic start,

    output logic [R*C*16-1:0] w_bf16_out,
    output logic [R*C*32-1:0] m_out,
    output logic [R*C*32-1:0] v_out,
    output logic              done
);

    localparam int N        = R * C;
    localparam int IDX_BITS = $clog2(N + 2);

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    typedef enum logic [1:0] { IDLE, UPDATE, DONE_ST } state_t;
    state_t state;

    logic [IDX_BITS-1:0] idx;

    // -----------------------------------------------------------------------
    // adam_cell connections
    // -----------------------------------------------------------------------
    logic        cell_en;
    logic [31:0] cell_g, cell_m, cell_v;
    logic [15:0] cell_w;
    logic [15:0] cell_w_out;
    logic [31:0] cell_m_out, cell_v_out;

    adam_cell u_cell (
        .clk        (clk),
        .rst        (rst),
        .en         (cell_en),
        .g_fp32     (cell_g),
        .w_bf16     (cell_w),
        .m_fp32     (cell_m),
        .v_fp32     (cell_v),
        .lr_fp32    (lr_fp32),
        .bc1_fp32   (bc1_fp32),
        .bc2_fp32   (bc2_fp32),
        .w_bf16_out (cell_w_out),
        .m_fp32_out (cell_m_out),
        .v_fp32_out (cell_v_out)
    );

    // Mux cell inputs from packed buses based on current idx
    always_comb begin
        if (idx < N) begin
            cell_g = grad     [idx*32 +: 32];
            cell_w = w_bf16_in[idx*16 +: 16];
            cell_m = m_in     [idx*32 +: 32];
            cell_v = v_in     [idx*32 +: 32];
        end else begin
            cell_g = '0;
            cell_w = '0;
            cell_m = '0;
            cell_v = '0;
        end
        cell_en = (state == UPDATE) && (idx < N);
    end

    // -----------------------------------------------------------------------
    // Output registers
    // -----------------------------------------------------------------------
    logic [15:0] w_reg [0:N-1];
    logic [31:0] m_reg [0:N-1];
    logic [31:0] v_reg [0:N-1];

    // Pack output registers onto output ports
    generate
        for (genvar i = 0; i < N; i++) begin : pack_out
            assign w_bf16_out[i*16 +: 16] = w_reg[i];
            assign m_out     [i*32 +: 32] = m_reg[i];
            assign v_out     [i*32 +: 32] = v_reg[i];
        end
    endgenerate

    // -----------------------------------------------------------------------
    // FSM + output capture
    // -----------------------------------------------------------------------
    // Capture timing:
    //   adam_cell is fully registered: outputs available one cycle after en=1.
    //   When idx=i+1 in UPDATE, outputs from param[i] are ready on cell_*_out.
    //   When idx=N, capture output from param[N-1] then transition to DONE.
    // -----------------------------------------------------------------------
    integer capture_idx;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            idx   <= '0;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)

                IDLE: begin
                    if (start) begin
                        state <= UPDATE;
                        idx   <= '0;
                    end
                end

                UPDATE: begin
                    // Capture output from param[idx-1] (available this cycle)
                    if (idx >= 1) begin
                        capture_idx = idx - 1;
                        w_reg[capture_idx] <= cell_w_out;
                        m_reg[capture_idx] <= cell_m_out;
                        v_reg[capture_idx] <= cell_v_out;
                    end
                    if (idx == N) begin
                        // Last param captured; go to DONE
                        state <= DONE_ST;
                    end else begin
                        idx <= idx + 1'b1;
                    end
                end

                DONE_ST: begin
                    done  <= 1'b1;
                    state <= IDLE;
                    idx   <= '0;
                end

                default: state <= IDLE;

            endcase
        end
    end

endmodule
