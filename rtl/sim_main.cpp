#include "Vnetwork.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <cstdio>

int main(int argc, char** argv) {
    VerilatedContext* contextp = new VerilatedContext;
    contextp->commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    Vnetwork* top = new Vnetwork(contextp);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    top->trace(tfp, 99);
    tfp->open("wave.vcd");

    // Initialise signals
    top->clk = 0;
    top->rst = 1;
    top->test = 0;
    top->eval();
    tfp->dump(0);

    // Release reset after 1 cycle
    top->rst = 0;

    const int sim_cycles = 5000;

    for (int cycle = 0; cycle < sim_cycles; cycle++) {
        // Assert test on cycle 0
        top->clk = 0;
        if (cycle == 0) top->test = 1;
        else            top->test = 0;
        top->eval();
        tfp->dump(cycle * 2);

        top->clk = 1;
        top->eval();
        tfp->dump(cycle * 2 + 1);

        // Check for completion on rising edge
        if (top->complete) {
            printf("HW_RESULT: %d\n", (int)top->inference);
            fflush(stdout);
            break;
        }
    }

    tfp->close();
    delete top;
    delete contextp;
    return 0;
}
