include(FetchContent)

FetchContent_Declare(
  ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
  GIT_TAG v5.0.0
)  

FetchContent_MakeAvailable(ftxui)

add_executable(trace trace.cpp)
target_link_libraries(trace PRIVATE llama ggml
  ftxui::screen
  ftxui::dom
  ftxui::component
)
