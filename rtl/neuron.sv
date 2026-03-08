//Code developed solely by Archie Eltherington December-January 2025-26

module neuron #(
    parameter int NUMWEIGHT       = 784,
    parameter int DATAWIDTH       = 16,
    parameter int NEURONLAYER     = 0,
    parameter int NEURONNODE      = 0
)(
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    infer,
    input  logic signed [DATAWIDTH-1:0] my_input,  
    output logic                    INFER_DONE,
    output logic signed [DATAWIDTH-1:0] neuron_out_reg
);


    logic [31:0] count;

    logic signed [DATAWIDTH-1:0] weights [0:NUMWEIGHT-1];
    logic signed [DATAWIDTH-1:0] bias;
    logic signed [DATAWIDTH-1:0] bias_mem [0:0];


    logic signed [31:0] mult_reg;

    logic signed [47:0] sum_reg;


    logic signed [31:0] shifted_sum;

    logic signed [DATAWIDTH-1:0] out_reg;


    initial begin
        string weights_dir, weight_file, bias_file;
        if (!$value$plusargs("WEIGHTSDIR=%s", weights_dir))
            weights_dir = "models/hidden10";
        weight_file = $sformatf("%s/weights_L%0d_N%0d.hex", weights_dir, NEURONLAYER, NEURONNODE);
        bias_file   = $sformatf("%s/bias_L%0d_N%0d.hex",    weights_dir, NEURONLAYER, NEURONNODE);
        $readmemh(weight_file, weights);
        $readmemh(bias_file,   bias_mem);
    end

    assign bias = bias_mem[0];

    typedef enum logic [1:0] {IDLE, INFER, OUTPUT} neuron_state;
    neuron_state current_state, next_state;

    always_ff @(posedge clk) begin
        if (rst)
            current_state <= IDLE;
        else
            current_state <= next_state;
    end

    always_comb begin
        next_state = current_state;
        case (current_state)
            IDLE: begin  
                if (infer) next_state = INFER;
            end
            INFER:  begin
                if (count == NUMWEIGHT) next_state = OUTPUT;
            end
            OUTPUT: begin
                next_state = IDLE;
            end
            default: next_state = IDLE;   
        endcase
    end


    always_ff @(posedge clk) begin
        if (rst) begin
            count   <= 0;
            sum_reg <= '0;
        end
        else if (current_state == INFER) begin
            mult_reg <= $signed(weights[count]) * $signed(my_input);
            sum_reg <= sum_reg + mult_reg;

            count <= count + 1;
        end
        else begin
            count   <= 0;
            sum_reg <= '0;
        end
    end


    // Align bias to Q4.26 before addition
    assign shifted_sum =
        (sum_reg + ($signed(bias) <<< 13)) >>> 13;


    always_comb begin
        if (shifted_sum < 0)
            out_reg = '0;
        else if (shifted_sum > 16'sh7FFF)
            out_reg = 16'sh7FFF;
        else
            out_reg = shifted_sum[15:0];
    end


    always_comb begin
        INFER_DONE = (current_state == OUTPUT);
    end

    always_ff @(posedge clk) begin
        if (rst)
            neuron_out_reg <= '0;
        else if (current_state == OUTPUT)
            neuron_out_reg <= out_reg;
    end

endmodule
