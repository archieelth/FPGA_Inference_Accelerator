SV_SRC = rtl/pixel_stream.sv rtl/neuron_loader.sv rtl/layer.sv \
         rtl/neuron.sv rtl/result_finder.sv rtl/network.sv

build:
	verilator -Wno-width --cc $(SV_SRC) \
	  --top-module network --trace \
	  --exe rtl/sim_main.cpp --build -o Vnetwork

run:
	./obj_dir/Vnetwork

clean:
	rm -rf obj_dir/ wave.vcd

.PHONY: build run clean
