cmake_minimum_required(VERSION 3.10)
project(native_worker_mock)

set(CMAKE_CXX_STANDARD 17)

add_executable(native_worker main.cpp)

# nlohmann/json is header-only. Expect it to be available in the system or vendor as a submodule.
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
if(NLOHMANN_JSON_INCLUDE_DIR)
    target_include_directories(native_worker PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
else()
    message(WARNING "nlohmann/json.hpp not found; please install or vendor the header to build native_worker")
endif()
