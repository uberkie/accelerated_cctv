#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: native_worker <socket_path>\n";
        return 1;
    }
    const char* socket_path = argv[1];

    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path)-1);
    unlink(socket_path);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        perror("bind");
        return 1;
    }
    if (listen(server_fd, 1) == -1) {
        perror("listen");
        return 1;
    }
    std::cout << "native_worker listening on " << socket_path << std::endl;

    int client = accept(server_fd, NULL, NULL);
    if (client == -1) {
        perror("accept");
        return 1;
    }

    // Read a JSON message (length-prefixed or newline-terminated). We'll read until EOF.
    std::string buf;
    char tmp[1024];
    ssize_t n;
    while ((n = read(client, tmp, sizeof(tmp))) > 0) {
        buf.append(tmp, n);
    }
    try {
        auto j = json::parse(buf);
        std::cout << "received json: " << j.dump(2) << std::endl;
        // Mock processing: reply with status
        json resp = { {"status", "ok"}, {"processed_by", "native_worker_mock"} };
        std::string out = resp.dump();
        write(client, out.c_str(), out.size());
    } catch (std::exception &e) {
        std::cerr << "error parsing json: " << e.what() << std::endl;
    }

    close(client);
    close(server_fd);
    return 0;
}
