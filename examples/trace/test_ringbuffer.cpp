#include "trace_recorder.h"
#include <cstdio>

int main(){
    TraceBuffer buf;
    for(int i = 0; i<5; i++){
        LayerEvent ev;
        ev.name = "layer_"+std::to_string(i);
        ev.latency_ms = i*1.5;
        ev.mean = i*0.1f;
        buf.push(ev);
    }
    printf("after pushing 5 events:\n");
    for(auto& ev:buf.snapshot()){
        printf(" %s latency=%.2fms mean=%.2f\n", ev.name.c_str(), ev.latency_ms, ev.mean);
    }
    return 0;
}