// ============================================================
// TOP-LEVEL NETWORK CONFIGURATION
// To change the architecture, edit these parameters only:
//
//   HIDDEN_NODES  — number of neurons in the hidden layer
//                   e.g. 10 (current), 32, 64, 128
//                   ⚠ must also retrain weights in fixedtest.py
//
//   NUMWEIGHTL1   — number of inputs (pixels). 784 for 28x28 MNIST.
//                   Only change if using a different dataset.
//
//   DATAWIDTH     — bit width of all data/weights. 16 for Q2.13.
//                   Only change if altering fixed-point format.
//
//   NEURONNODES   — number of output classes. 10 for digits 0-9.
//                   Only change if classifying more/fewer classes.
// ============================================================
module network#(
    parameter int HIDDEN_NODES    = 64,  // ← change this to grow the hidden layer
    parameter int NUMWEIGHTL1     = 784,
    parameter int DATAWIDTH       = 16,
    parameter int NEURONNODES     = 10   // output classes — matches number of digits
)(
    input logic clk,
    input logic rst,
    input logic test,
  
    output logic [3:0] inference,
    output logic complete
);
    localparam int LAYER1_ID = 1;
    localparam int LAYER2_ID = 2;

    logic [15:0] image [0:NUMWEIGHTL1-1];

    initial begin
        string imagefile;
        if (!$value$plusargs("IMAGEFILE=%s", imagefile))
            imagefile = "test_images/image_2.hex";
        $readmemh(imagefile, image);
    end

    logic infer_l1;
    logic infer_l2;
    logic [HIDDEN_NODES-1:0]  l1_done;
    logic [NEURONNODES-1:0]   l2_done;
    logic START_FLAG1;
    logic START_FLAG2;
    logic start_search;
    logic found;

    logic [DATAWIDTH-1:0] pixel_data;
    logic [DATAWIDTH-1:0] neuron_val;
    logic [HIDDEN_NODES-1:0][DATAWIDTH-1:0]  layer1_out_reg;
    logic [NEURONNODES-1:0][DATAWIDTH-1:0]   layer2_out_reg;

    //layer 1 — HIDDEN_NODES neurons, each with NUMWEIGHTL1 inputs
    layer  #(
        .NUMWEIGHT  (NUMWEIGHTL1),
        .DATAWIDTH  (DATAWIDTH),
        .NEURONLAYER(LAYER1_ID),
        .NEURONNODES(HIDDEN_NODES)
    ) layer1 (
        .clk(clk),
        .rst(rst),
        .infer(infer_l1),
        .my_input(pixel_data),
        .infer_done(l1_done),
        .layer_out_reg(layer1_out_reg)
    );

    // layer 2 — NEURONNODES output neurons, each with HIDDEN_NODES inputs
    layer #(
        .NUMWEIGHT  (HIDDEN_NODES),
        .DATAWIDTH  (DATAWIDTH),
        .NEURONLAYER(LAYER2_ID),
        .NEURONNODES(NEURONNODES)
    ) layer2 (
        .clk(clk),
        .rst(rst),
        .infer(infer_l2),
        .my_input(neuron_val),
        .infer_done(l2_done),
        .layer_out_reg(layer2_out_reg)
    );

    pixel_stream  #(
        .NUMWEIGHT(NUMWEIGHTL1),
        .DATAWIDTH(DATAWIDTH),
        .IMAGEFILE("input.hex")
    ) pixels_module (
        .clk(clk),
        .rst(rst),
        .START_FLAG(START_FLAG1),
        .image(image),
        .pixel_data(pixel_data)
    );

    // streams layer1 outputs one-by-one into layer2 — uses HIDDEN_NODES
    neuron_loader  #(
        .DATAWIDTH  (DATAWIDTH),
        .NEURONNODES(HIDDEN_NODES)
    ) neurons_module (
        .clk(clk),
        .rst(rst),
        .START_FLAG(START_FLAG2),
        .layer_out_reg(layer1_out_reg),
        .neuron_val(neuron_val)
    );

    result_finder #(
        .DATAWIDTH(DATAWIDTH),
        .NEURONNODES(NEURONNODES)
    )findingg(
        .clk(clk),
        .rst(rst),
        .start_search(start_search),
        .layer_out_reg(layer2_out_reg),
        .found(found),
        .result(inference)
    );

    //FSM to load image into layer1 retrieve the 10 results and then prepare streaming into next
    typedef enum {IDLE, START_STREAM1 ,COMPUTE_LAYER1, START_STREAM2, COMPUTE_LAYER2, RESULT, RESULTpp} network_state;
    network_state current_state, next_state;


    always_ff @(posedge clk)
        if(rst)
            current_state <= IDLE;
        else
            current_state <= next_state;

    //state transitions
    always_comb begin
        next_state = current_state;
        case (current_state)
            IDLE: begin
                if (test)         
                    next_state = START_STREAM1;
            end
            START_STREAM1: begin
                next_state = COMPUTE_LAYER1;
            end
            COMPUTE_LAYER1: begin
                if (&l1_done)  
                    next_state = START_STREAM2;
            end
            START_STREAM2: begin
                next_state = COMPUTE_LAYER2;
            end
            COMPUTE_LAYER2: begin
                if (&l2_done)  
                    next_state = RESULT;
            end
            RESULT: begin
                if (found)
                    next_state = RESULTpp;       
            end
            RESULTpp: begin
                next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end

    //control  logic
    always_comb begin
        START_FLAG1 =  0;
        START_FLAG2 =  0;
        infer_l1     = 0;
        infer_l2     = 0;
        start_search = 0;
        complete     = 0;

        case (current_state)
            START_STREAM1:      START_FLAG1  = 1;
            START_STREAM2:      START_FLAG2  = 1;
            COMPUTE_LAYER1:     infer_l1     = 1;
            COMPUTE_LAYER2:     infer_l2     = 1;
            RESULT:             start_search = 1;
            RESULTpp:           complete     = 1;
        endcase
    end
endmodule
