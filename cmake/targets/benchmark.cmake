set(BENCHMARK_SOURCES main.cpp)

list(TRANSFORM BENCHMARK_SOURCES PREPEND "${PROJECT_NOR_BENCHMARK_SRC_DIR}/")

add_executable(${nor_benchmark} ${BENCHMARK_SOURCES})

target_link_libraries(${nor_benchmark} PRIVATE ${nor_lib}_envs benchmark::benchmark)
