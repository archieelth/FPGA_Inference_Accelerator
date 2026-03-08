//Code developed solely by Archie Eltherington December 2025

module layer #(
    parameter int NUMWEIGHT       = 784,
    parameter int DATAWIDTH       = 16,
    parameter int NEURONLAYER     = 0,
    parameter int NEURONNODES     = 10
)(
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    infer,
    input  logic [DATAWIDTH-1:0]    my_input,
    output logic [NEURONNODES-1:0]  infer_done,
    output logic [NEURONNODES-1:0][DATAWIDTH-1:0] layer_out_reg
);
    genvar n;
    generate
        for (n = 0; n < NEURONNODES; n++) begin : gen_neurons
            neuron #(
                .NUMWEIGHT(NUMWEIGHT),
                .DATAWIDTH(DATAWIDTH),
                .NEURONLAYER(NEURONLAYER),
                .NEURONNODE(n)
            ) neuron_i (
                .clk(clk),
                .rst(rst),
                .infer(infer),
                .my_input(my_input),
                .INFER_DONE(infer_done[n]),
                .neuron_out_reg(layer_out_reg[n])
            );
        end
    endgenerate
endmodule