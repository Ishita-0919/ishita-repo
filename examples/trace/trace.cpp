#include "llama.h"
#include "ggml.h"
#include "trace_recorder.h"
#include <chrono>
#include <cstdio>
#include <cmath>
#include <thread>
#include <atomic>

static std::atomic<bool> g_generation_done{false};
static TraceBuffer g_trace_buffer;
static AttnBuffer g_attn_buffer;
static std::chrono::steady_clock::time_point g_last_ts;
static std::vector<std::string> g_token_strs;

static void compute_stats(const float* data, size_t n, float& mean, float& max_val, float& sparsity) {
    double sum = 0; max_val = -INFINITY; size_t zeros = 0;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
        if (data[i] > max_val) max_val = data[i];
        if (std::fabs(data[i]) < 1e-6f) zeros++;
    }
    mean = n ? (float)(sum / n) : 0.f;
    sparsity = n ? (float)zeros / n : 0.f;
}

static bool eval_callback(struct ggml_tensor* t, bool ask, void* /*user_data*/) {
    if (ask) {
        return true;
    }
    auto now = std::chrono::steady_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(now - g_last_ts).count();
    g_last_ts = now;

    LayerEvent ev;
    ev.name = t->name;
    for (int i = 0; i < GGML_MAX_DIMS; i++) ev.shape.push_back(t->ne[i]);
    ev.latency_ms = latency_ms;
    ev.timestamp = std::chrono::system_clock::now();

    size_t nelem = ggml_nelements(t);
    std::vector<float> host_buf;
    const float* ptr = nullptr;

    if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
        ptr = (const float*) t->data;
    } else if (t->buffer) {
        host_buf.resize(nelem);
        ggml_backend_tensor_get(t, host_buf.data(), 0, nelem * sizeof(float));
        ptr = host_buf.data();
    }

    if (ptr) compute_stats(ptr, nelem, ev.mean, ev.max_val, ev.sparsity);
    g_trace_buffer.push(ev);

    if (ptr && std::string(t->name).find("kq_soft_max-") != std::string::npos) {
        AttentionSnap snap;
        snap.layer_name = t->name;
        snap.kv_len = t->ne[0];
        snap.n_tokens = t->ne[1];
        snap.n_heads = t->ne[2];
        snap.weights.assign(ptr, ptr + nelem);
        g_attn_buffer.push(snap);
    }
    return true;
}

static void run_generation(llama_context* ctx, const llama_vocab* vocab,
                            std::vector<llama_token> tokens) {
    int n_predict = 20;

    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());

    for (int i = 0; i < n_predict; i++) {
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at step %d\n", i);
            break;
        }

        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            printf("\n[EOG reached at step %d]\n", i);
            break;
        }

        char piece_buf[64];
        int piece_len = llama_token_to_piece(vocab, new_token, piece_buf, sizeof(piece_buf), 0, true);
        std::string piece(piece_buf, piece_len > 0 ? piece_len : 0);
        g_token_strs.push_back(piece);
        printf("%s", piece.c_str());
        fflush(stdout);

        batch = llama_batch_get_one(&new_token, 1);
    }
    printf("\n");

    llama_sampler_free(smpl);
    g_generation_done = true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.cb_eval = eval_callback;
    cparams.cb_eval_user_data = nullptr;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    g_last_ts = std::chrono::steady_clock::now();

    std::string prompt = "The quick brown fox";
    std::vector<llama_token> tokens(prompt.size() + 8);
    int n = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), tokens.data(), (int)tokens.size(), true, false);
    tokens.resize(n);

    g_token_strs.clear();
    for (auto tok : tokens) {
        char buf[64];
        int len = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (len < 0) len = 0;
        g_token_strs.emplace_back(buf, len);
    }
    printf("\nToken strings (%zu):\n", g_token_strs.size());
    for (size_t i = 0; i < g_token_strs.size(); i++) {
        printf(" [%zu] '%s'\n", i, g_token_strs[i].c_str());
    }

    g_last_ts = std::chrono::steady_clock::now();

    std::thread worker(run_generation, ctx, vocab, tokens);
    worker.join(); // for now: wait for it to finish, same behavior as before (no live TUI yet)

    auto events = g_trace_buffer.snapshot();
    printf("Captured %zu trace events:\n", events.size());
    for (auto& ev : events) {
        printf("%-30s shape=[%lld,%lld,%lld,%lld] latency=%.3fms mean=%.4f max=%.4f sparsity=%.2f\n",
            ev.name.c_str(),
            (long long)ev.shape[0], (long long)ev.shape[1], (long long)ev.shape[2], (long long)ev.shape[3],
            ev.latency_ms, ev.mean, ev.max_val, ev.sparsity);
    }
    auto attn_snaps = g_attn_buffer.snapshot();
    printf("\ncaptured %zu attention snaps:\n", attn_snaps.size());
    for (auto& s : attn_snaps) {
        printf("%-20s kv_len=%d n_tokens=%d n_heads=%d\n", s.layer_name.c_str(), s.kv_len, s.n_tokens, s.n_heads);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}