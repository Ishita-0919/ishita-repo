#pragma once 
#include <atomic>
#include <vector>
#include <string>
#include "trace_recorder.h"
#include "ftxui/dom/elements.hpp"

ftxui::Element RenderUI(
    TraceBuffer& trace_buffer,
    AttnBuffer& attn_buffer,
    const std::vector<std::string>& tokens,
    const std::atomic<bool>& generation_done,
    const std::atomic<int>& frame_count
);