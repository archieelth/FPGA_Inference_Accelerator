module pixel_stream #(
    parameter int NUMWEIGHT       = 784,
    parameter int DATAWIDTH       = 16,
    parameter string IMAGEFILE    = "input.hex"
)(
    input  logic clk,
    input  logic rst,
    input  logic START_FLAG,
    input  logic [15:0]image[0:NUMWEIGHT-1],
    output logic [DATAWIDTH-1:0] pixel_data
);

    // Store pixels as 16-bit values (Q2.13 format from Python)
    logic [9:0] count;  // Only need 10 bits for 0-783

    typedef enum {IDLE, STREAM, DONE} stream_state;
    stream_state current_state, next_state;

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
                if(START_FLAG) 
                    next_state = STREAM;
            end
            STREAM: begin
                if (count == NUMWEIGHT)
                    next_state = DONE;
            end
            DONE: begin
                next_state = IDLE;
            end
        endcase
    end

    //reset count
    always_ff @(posedge clk) begin
        if(rst || current_state == IDLE)
            count <= '0;
        else if (current_state == STREAM)
            count <= count + 1;
    end

    //register update - directly read from memory
    always_ff @(posedge clk) begin
        if (rst)
            pixel_data <= '0;
        else if (current_state == STREAM)
            pixel_data <= image[count];
    end

endmodule