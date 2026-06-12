# sim/scripts/sim_bf16_mac.tcl
# Run bf16_mac simulation from a Vivado TCL shell.
# Usage: vivado -mode batch -source sim/scripts/sim_bf16_mac.tcl
#
# Env vars expected (set before invoking Vivado):
#   NEURO_FPGA_SRC   — absolute path to rtl/
#   NEURO_TESTVECS   — absolute path to run/fpga-testvecs/

set fpga_src  $::env(NEURO_FPGA_SRC)
set testvecs  $::env(NEURO_TESTVECS)

# Compile
exec xvlog --sv \
    $fpga_src/bf16_mac.sv \
    [file join [file dirname [file dirname [info script]]] ../../tb/tb_bf16_mac.sv] \
    >@ stdout 2>@ stderr

# Elaborate
exec xelab tb_bf16_mac -s tb_bf16_mac_sim \
    >@ stdout 2>@ stderr

# Simulate — pass testvecs path as plusarg
exec xsim tb_bf16_mac_sim -runall \
    -testplusarg "NEURO_TESTVECS=$testvecs" \
    >@ stdout 2>@ stderr
