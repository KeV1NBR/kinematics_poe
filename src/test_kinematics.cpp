#include "kinematics.h"

using namespace std;

int main() {
    Kinematics km;
    vector<float> pos = {M_PI_2, M_PI_2, M_PI_2, M_PI_2, M_PI_2, M_PI_2};

    for (int i = 0; i < 10; i++) km.forward(pos);

    clock_t now = clock();
    for (int i = 0; i < 10; i++) km.forward(pos);
    clock_t end = clock();
    cout << double(end - now) / CLOCKS_PER_SEC << endl;

    return 0;
}
