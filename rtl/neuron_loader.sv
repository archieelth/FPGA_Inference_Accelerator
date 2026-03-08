module neuron_loader#(
    parameter int DATAWIDTH       = 16,
    parameter int NEURONNODES     = 10
)(
    input  logic clk,
    input  logic rst,
    input  logic START_FLAG,
    input  logic [NEURONNODES-1:0][DATAWIDTH-1:0] layer_out_reg,
    output logic [DATAWIDTH-1:0] neuron_val
);

    logic [$clog2(NEURONNODES+1):0] count;

    typedef enum {IDLE, STREAM, DONE} stream_state;
    stream_state current_state, next_state;


    always_comb begin 
        next_state = current_state;
        case (current_state)
            IDLE: begin
                if(START_FLAG) 
                    next_state = STREAM;
            end
            STREAM: begin
                if (count == NEURONNODES)
                    next_state = DONE;
            end
            DONE: begin
                next_state = IDLE;
            end
        endcase
    end

    //register update
    always_ff @(posedge clk) begin
        if (rst)
            neuron_val <= '0;
        else if (current_state == STREAM)
            neuron_val <= layer_out_reg[count];
    end

    //reset count
    always_ff @(posedge clk) begin
        if(rst || current_state == IDLE)
            count <= '0;
        else if (current_state == STREAM)
            count <= count + 1;
    end

    always_ff @(posedge clk) begin
        if (rst)
            current_state <= IDLE;
        else
            current_state <= next_state;
    end
endmodule
