# neuro-fabric-fpga

Xilinx FPGA RTL implementation of the NeuronFabric Transformer training pipeline.

**Paper:** [NeuronFabric — arXiv:2606.16440](https://arxiv.org/abs/2606.16440)

## Folder structure

```
rtl/          Synthesisable SystemVerilog source
tb/           XSim testbenches (not synthesisable)
sim/scripts/  Vivado TCL simulation scripts
scripts/      Vivado project-creation and build TCL scripts
constraints/  XDC timing/pin constraint files
ip/           Vivado IP (.xci) cores
docs/         Hardware design notes
```

## Environment variables (set on dev machine)

| Variable          | Points to                                    |
|-------------------|----------------------------------------------|
| `NEURO_FPGA_SRC`  | `<this repo>/rtl/`                           |
| `NEURO_TESTVECS`  | `<neuro-fabric repo>/run/fpga-testvecs/`     |

## Simulation quick-start (manual)

```powershell
xvlog --sv rtl/bf16_mac.sv tb/tb_bf16_mac.sv
xelab tb_bf16_mac -s tb_bf16_mac_sim
xsim tb_bf16_mac_sim -runall
```

Or from C#: `dotnet test Neuro.Attention.XSim.LocalTests` (runs the above automatically).
