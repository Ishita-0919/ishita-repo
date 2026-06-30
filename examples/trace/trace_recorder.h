#pragma once
#include <array>
#include <vector>
#include <mutex>
#include <chrono>
#include <string>
#include <cstdint>

struct LayerEvent{
    std::string name;
    std::vector<int64_t> shape;
    float mean = 0, max_val = 0, sparsity = 0;
    double latency_ms = 0;
    std::chrono::system_clock::time_point timestamp;
};
struct AttentionSnap{
    std::string layer_name;
    int kv_len = 0;
    int n_tokens = 0;
    int n_heads = 0;
    std::vector<float> weights;
    std::chrono::system_clock::time_point timestamp;
};
template<typename T, size_t N>
class RingBuffer{
    std::array<T, N> buf;
    size_t head = 0, count = 0;
    std::mutex mtx;
    public: 
    void push(T item){
        std::lock_guard<std::mutex> lock(mtx);
        buf[head] = std::move(item);
        head = (head+1)%N;
        if(count<N) count++;
    }
    std::vector<T> snapshot(){
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<T> out;
        for(size_t i = 0; i<count; i++){
            out.push_back(buf[(head+N-count+i)%N]);
        }
        return out;
    }
};
using TraceBuffer = RingBuffer<LayerEvent, 512>;
using AttnBuffer = RingBuffer<AttentionSnap, 10>;