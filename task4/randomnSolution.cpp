#include <iostream>

using namespace std;

int main() {
    for(int i = 0; i < 59543; ++i) {
        cout << rand() % 2 << endl;
    }
    cout << rand() % 2;
}