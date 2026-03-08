module ReLU #(
    parameter int DATAWIDTH     = 64
)(
    input logic [DATAWIDTH-1: 0] x,
    output logic [DATAWIDTH-1:0] y
);

    always_comb begin
        if (x[DATAWIDTH-1] == 1)
            y = '0;
        else
            y = x;
    end
endmodule
