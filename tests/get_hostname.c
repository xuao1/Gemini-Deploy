#include <iostream>
#include <unistd.h>

int main() {
    char hostName[256]; // You can use a buffer of appropriate size
    if (gethostname(hostName, sizeof(hostName)) == 0) {
        std::cout << "Hostname: " << hostName << std::endl;
    } else {
        std::cerr << "Failed to get hostname." << std::endl;
    }
    return 0;
}
