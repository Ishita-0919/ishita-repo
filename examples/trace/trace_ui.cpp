#include <atomic>
#include <algorithm>
#include <string>
#include "trace_recorder.h"
#include <ftxui/dom/elements.hpp>

using namespace ftxui;
static int active_layer = 0;
std::string ProgressBar(float value, int width = 20) {
    value = std::max(0.f, std::min(1.f, value));

    int filled = int(value * width);

    std::string bar;

    for (int i = 0; i < filled; i++)
        bar += u8"\u2588";

    for (int i = filled; i < width; i++)
        bar += u8"\u2591";

    return bar;
}
Element RenderHeatmap(
    AttnBuffer& attn_buffer,
    const std::vector<std::string>& tokens) {

    auto snaps = attn_buffer.snapshot();

    if (snaps.empty())
        return text("No attention captured yet.");

    auto& snap = snaps.back();

    int kv_len = snap.kv_len;
    int n_tokens = snap.n_tokens;
    int n_heads = snap.n_heads;

    std::vector<float> avg(kv_len * n_tokens, 0.0f);

    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < n_tokens; t++) {
            for (int k = 0; k < kv_len; k++) {

                avg[t * kv_len + k] +=
                    snap.weights[
                        h * n_tokens * kv_len +
                        t * kv_len +
                        k
                    ];
            }
        }
    }

    for (auto& v : avg)
        v /= n_heads;

    const std::string blocks[] = {
        " ",
        u8"\u2591",
        u8"\u2592",
        u8"\u2593",
        u8"\u2588"
    };

    Elements rows;

    int show = std::min(kv_len, 20);
    for (int t = 0; t<n_tokens; t++) {
        float row_max = 1e-6f;
        for(int k = kv_len-show; k<kv_len; k++){
            row_max = std::max(row_max, avg[t*kv_len+k]);
        }
        std::string line;
        for (int k = kv_len - show; k < kv_len; k++) {
            float w = avg[t*kv_len+k]/row_max;
            int idx = std::min(4, (int)(w*5.0f));
            line += blocks[idx];
        }

        std::string label =
            (t < (int)tokens.size())
            ? tokens[t]
            : "tok" + std::to_string(t);

        std::string debug_line;
        for(char c:line){
            debug_line += std::to_string((int)(unsigned char)c) + ",";
        }    
        rows.push_back(
            hbox({
                text(label)
                    | size(WIDTH, EQUAL, 12),
                text(debug_line)
            })
        );
    }

    return window(
        text(" Attention Heatmap "),
        vbox(rows) | frame
    );
}
Element RenderUI(
    TraceBuffer& trace_buffer,
    AttnBuffer& attn_buffer,
    const std::vector<std::string>& tokens,
    const std::atomic<bool>& generation_done,
    const std::atomic<int>& frame_count) {

    auto events = trace_buffer.snapshot();

    if (!events.empty()) {
      std::string name = events.back().name;

      for (int i = 0; i < 32; i++) {
          std::string s = std::to_string(i);
          if (name.find("." + s + ".") != std::string::npos) {
              active_layer = i;
              break;
          }
      }
    }
    Elements topology;

    topology.push_back(text("llama"));
    topology.push_back(text("├── embed_tokens"));
    topology.push_back(text("├── layers"));

    for (int i = 0; i < 16; i++) {
      bool active = (i == active_layer);
      std::string prefix = active ? "│   ► " : "│     ";
      topology.push_back(
          text(prefix + std::string("layer.") + std::to_string(i))
          | (active ? bold | color(Color::Green)
                    : color(Color::GrayLight))
      );

    if (active) {
        topology.push_back(
            text("│      ├── Attention")
            | color(Color::Yellow));

        topology.push_back(
            text("│      ├── Feed Forward")
            | color(Color::Yellow));

        topology.push_back(
            text("│      └── Residual")
            | color(Color::Yellow));
    }
    }
    topology.push_back(text("└── lm_head"));

    auto topology_panel =
    window(
        text(" Model Topology "),
        vbox(topology) | frame
    );
    Elements left;

    int start = std::max(0, (int)events.size() - 20);
    for (int i = start; i<(int)events.size(); i++) {
        auto &e = events[i];
        Element latency;
        std::string latency_str = std::to_string(e.latency_ms).substr(0, 5)+" ms";
        if (e.latency_ms < 0.05) {
          latency = text(latency_str) | color(Color::Green);
        }
        else if (e.latency_ms < 0.5) {
          latency = text(latency_str) | color(Color::Yellow);
        }
        else {
          latency = text(latency_str) | color(Color::Red);
        }
        Element tensor = text(e.name);

        if (e.name.find("attn") != std::string::npos)
          tensor |= color(Color::Cyan);
        else if (e.name.find("ffn") != std::string::npos)
          tensor |= color(Color::Yellow);
        else if (e.name.find("norm") != std::string::npos)
          tensor |= color(Color::Green);
        else if (e.name.find("Q") != std::string::npos ||
         e.name.find("K") != std::string::npos ||
         e.name.find("V") != std::string::npos)
          tensor |= color(Color::Magenta);
        left.push_back(
            hbox({
                tensor | size(WIDTH, EQUAL, 25),
                separator(),
                text(
                    "[" +
                    std::to_string(e.shape[0]) + "," +
                    std::to_string(e.shape[1]) + "," +
                    std::to_string(e.shape[2]) + "," +
                    std::to_string(e.shape[3]) + "]"
                ) | size(WIDTH,EQUAL,19),
                separator(),
                latency
            })
        );
    }

    auto left_panel =
        window(
            text(" Layer Execution "),
            vbox(left) | frame
        );

    Elements stats;

    if(!events.empty()){
        auto &e = events.back();
        auto bar = [](float x){
            int n = std::min(20,(int)(x*20));
            std::string s;
            for(int i=0;i<n;i++)
                s += u8"\u2588";
            return s;
        };
        stats.push_back(text("Tensor"));
        stats.push_back(text(e.name));
        stats.push_back(separator());
        stats.push_back(text("Mean : "+std::to_string(e.mean)));
        stats.push_back(text("Max  : "+std::to_string(e.max_val)));
        stats.push_back(separator());
        stats.push_back(text("Sparsity"));
        stats.push_back(text(ProgressBar(e.sparsity)+std::to_string((int)(e.sparsity*100))
            +" %"));
        // stats.push_back(text(
        //     std::to_string((int)(e.sparsity*100))
        //     +" %"));
        stats.push_back(separator());
        stats.push_back(text(
            generation_done.load()
            ? "Generation Finished"
            :
            "Running..."
        ));

        stats.push_back(text(
            "Frame : "
            +std::to_string(frame_count.load())
        ));
    }

    auto right_panel =
        window(
            text(" Runtime "),
            vbox(stats)
        );
    
    auto heatmap_panel = RenderHeatmap(attn_buffer, tokens);    
    return vbox(Elements{
        text("LLM Telemetry Dashboard") | bold | center,
        separator(),
        hbox({
            topology_panel | size(WIDTH, EQUAL, 25),
            separator(),
            left_panel | flex,
            separator(),
            right_panel | size(WIDTH,EQUAL,25)
        }),
        separator(),
        heatmap_panel
    }) | border;
}