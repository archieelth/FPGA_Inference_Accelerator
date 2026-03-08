module result_finder#(
    parameter int DATAWIDTH       = 16,
    parameter int NEURONNODES     = 10
)(  
    input logic clk,
    input logic rst,
    input logic start_search,
    input logic [NEURONNODES-1:0][DATAWIDTH-1:0] layer_out_reg,
    output logic found,
    output logic [3:0] result

);
    logic [$clog2(NEURONNODES+1):0] count;
    logic [DATAWIDTH-1:0] tmp;
    logic [$clog2(NEURONNODES)-1:0] result_holder;

    typedef enum {IDLE, SEARCH, DONE} stream_state;
    stream_state current_state, next_state;

    //state transistions
    always_comb begin 
        next_state = current_state;
        found = 0;
        case (current_state)
            IDLE: begin
                if(start_search) 
                    next_state = SEARCH;
            end
            SEARCH: begin
                if (count == NEURONNODES)
                    next_state = DONE;
            end
            DONE: begin
                found = 1;
                next_state = IDLE;
            end
        endcase
    end
    
    //compare module
    always_ff @(posedge clk) begin
        if (current_state == SEARCH) begin
            if (count == 0 || tmp < layer_out_reg[count]) begin
                tmp <= layer_out_reg[count];
                result_holder <= count;
            end
        end
        else if (current_state == IDLE) begin
            tmp <= '0;
            result_holder <= '0;
        end
    end

    //register update
    always_ff @(posedge clk) begin
        if (rst)
            result <= '0;
        else if (current_state == DONE)
            result <= {4'b0, result_holder};
    end

    //count
    always_ff @(posedge clk) begin
        if(rst || current_state == IDLE)
            count <= '0;
        else if (current_state == SEARCH)
            count <= count + 1;
    end

    //next state
    always_ff @(posedge clk) begin
        if (rst)
            current_state <= IDLE;
        else
            current_state <= next_state;
    end
endmodule
