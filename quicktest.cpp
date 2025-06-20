#include <bits/stdc++.h>

using namespace std;

int main(){

    int time = 3599;

    int seconds = time % 60;
    int minutes = (time%3600)/60;

    cout << minutes << ", " << seconds << endl;

    // for (int i = 0; i < 9; i++){
    //     cout << i % 4 << endl;
    // }

    // vector<int> mytestvec (2,5);
    // for (int item : mytestvec){
    //     cout << item << endl;
    // }

    int index = 145;
    vector<int> dims {2,12, 25};
    int curdim = 1;
    int prevdim = curdim;
    for (int mydim : dims){
        curdim *= mydim;
        cout << (index % curdim)/prevdim << ", ";
        prevdim = curdim;
        // This might be an off by one error...
        // Need to remove the first dimensions value when looking at the second one.
    }
    cout << endl;
}