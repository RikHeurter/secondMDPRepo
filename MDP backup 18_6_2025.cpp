#include <stdio.h>
#include <vector>
#include <string>
#include <tuple>
#include <utility>
#include <iostream>
#include <assert.h>
#include <fstream>

// TODO:
//      1. Add binary search indexing of partitions;            DONE
//      2. Apply uniformisation;                                DONE
//      3. Calculate SA;                                        DONE (For now)
//      8. Check which of the two versions of line 447 I want   DONE
//      9. Fix this epsilon only growing bug using arbitrary precision floats. TODO
//      4. Add some fancy way to output vals;
//      5. Create plots in Matplotlib.
//      6. Add a way to input percentages.
//      7. Some more clean-up
//      10. After I am done with the BSc. thesis. Look at how
//          there is such a big speed-up by the compiler -O2 setting.

//      Check that I do vector initialisation correctly.
//      Check how I want to get the correct partitions.
using namespace std;

/*
@brief
Simply calculates Amdahlslaw, and then applies the actual speed on top of
it.

In:
    - @param Speed: the actual speed of the core running a single job;
    - @param p: the percentage parallelisability of a job;
    - @param n: the number of cores.

Out:
    - @return the new speed.
*/
double calcAmdahlsLaw(double speed, double p, double n){
    return 1/(1-p + p/n)*speed;
}

/*
@brief
Calculates all partitions of size n given a set of the previous partitions
of size (n-1);

In:
    - @param prevpartitions: The set of previous partitions of size n-1;
    - @param n: The new size of the partition n;

Out:
    - @return partitions, all the partitions of size n.
*/
vector<vector<int>> calc_partition(vector<vector<vector<int>>> prevpartitions, int n){
    vector<vector<int>> partitions = {{n}};
    // Get all possible starting numbers for the partitions.
    for (int i = n-1; i > 0; i--){
        vector<vector<int>> curpartition = prevpartitions[n-i-1];
        for (vector<int> parti : curpartition){
            // Always have the sum in sorted order to prevent duplicates.
            if (parti[0] > i){
                continue;
            }
            vector<int> newparti = {i};
            for (int item : parti){
                newparti.push_back(item);
            }
            partitions.push_back(newparti);
        }
    }
    return partitions;
}

/*
@brief
Calculates all the partitions up to size n.

In:
    - @param n: the paritition number;

Out:
    - @return All possible partitions up to size n.
*/
vector<vector<vector<int>>> calc_partitions(int n){
    // cout << "C1" << endl;
    vector<vector<vector<int>>> prevpartitions = {{{1}}};
    for (int i = 2; i <= n; i++){
        // cout << "A" << endl;
        vector<vector<int>> newpartition = calc_partition(prevpartitions, i);
        prevpartitions.push_back(newpartition);
        // cout << "B" << endl;
    }
    return prevpartitions;
}

/*
@brief
Add extra partitions if required. Up to partition number n,

In:
    @param n: the partition number;
    @param prevpartitions: the previous known number of partitions;

Out:
    @return The partitions up to partition number n.
*/
vector<vector<vector<int>>> calc_more_partitions(vector<vector<vector<int>>> prevpartitions, int n){
    if (n <= prevpartitions.size() - 1){
        return prevpartitions;
    }
    for (int i = prevpartitions.size() + 1; i <= n; i++){
        vector<vector<int>> newpartition = calc_partition(prevpartitions, i);
        prevpartitions.push_back(newpartition);
    }
    return prevpartitions;
}

/*
@brief Collapses the partitions into vectors ordered largest value to smallest.

In:
    - @param one_dim_states: the not collapsed partitions.
    - @param cutoff: _unused_

Out:
    - @return The collapsed partitions ordered largest to smallest.
*/
vector<vector<vector<int>>> collapse_partitions(vector<vector<vector<int>>> one_dim_states, int cutoff){
    vector<vector<vector<int>>> newstates;
    for (vector<vector<int>> curparti : one_dim_states){
        vector<vector<int>> newparti;
        for (vector<int> curpartii : curparti){
            vector<int> newpartii = vector<int>(curpartii[0]+1, 0); // Works due to first value always being highest value due to it being sorted heighest to lowest.
            for (int val : curpartii){
                newpartii[val]++;
            }
            // for (int i = cutoff; i > -1; i--){
            //     vector<int> newpartiii(newpartii);
            //     newpartiii[0] = i;
            //     newparti.push_back(newpartiii);
            // }
            newparti.push_back(newpartii);
        }
        newstates.push_back(newparti);
    }
    return newstates;
}



/*
@brief
Class containing our Markov Decision Process solver.

Parameters:
    - @param dim: the number of different types of cores;
    - @param rates: the Amdahl's law rates;
    - @param speeds: the different speeds that cores run on;
    - @param cores: vector describing the number of cores we have of type i.
    - @param maxsize: _unused_
    - @param arrival rate: the normal arrival rate of jobs;
    - @param departure rate: the normal departure rate of jobs;
    - @param partitions: the vector containing all possible partition states;
    - @param cutoff: the maximum number of jobs allowed in the system to wait,
                required for policy iterations;
    - @param valuefunc: the vector describing the current expected value of the MDP;
    - @param partitionsizes: contains the size of cumalitive number of entries in
            the partition up to variable n.
    - @param dims: the vector describing the partition size for core type i;
    - @param uniformization_constant: the constant to uniformise our markov
        decision process;
    - @param eps: the epsilon after which we feel like no progress can be made
        anymore.

Functions:
    - @tparam build_markov_chain: builds the actual markov chain's adjacency graph;
    - @tparam calc_index: calculates the index from a partition;
    - @tparam get_valuefunc: gets the value of the value func for a specific
        index
    - @tparam binsearch_partitions: helper function for binsearch_partitions;
    - @tparam index_to_partition_indices: calculates the partition indices belonging
        to a specific index;
    - @tparam index_to_partitions: calculates the partition indices across multiple
        jobs.
    - @tparam SAStep: computes a succesive approximation step.
*/
class MarkovDP{
    int dim;
    vector<double> rates;
    vector<double> speeds;
    vector<int> cores;
    int maxsize;
    double arrivalrate;
    double departurerate;
    vector<vector<vector<int>>> partitions;
    int cutoff;
    vector<double> valuefunc;
    vector<int> partitionsizes;
    vector<int> dims;
    double uniformization_constant;
    double eps;

    // Save data
    vector<int> actions;
    vector<vector<vector<int>>> partition_chosen;
    vector<int> num_jobs_allocated;
    vector<int> jobs_in_system;
    vector<vector<int>> num_cores_in_use;

    public:

        MarkovDP(int dimp, vector<double> ratesp, vector<double> speedsp, vector<int> coresp, int maxsizep, double arrivalratep, double departureratep, int cutoffp, double epsp){
            // cout << "A" << endl;
            cout << "Setting up the MDP solver..." << endl;
            if (!(ratesp.size() == coresp.size() && dimp == ratesp.size())){
                cout << "Please make sure to have all core sizes be identical" << endl;
                exit(1);
            }
            if (dimp <= 0){
                cout << "Please make sure to have at least one set of cores" << endl;
                exit(1);
            }
            if (epsp <= 0){
                cout << "Please make sure to have eps > 0" << endl;
                exit(1);
            }
            dim = dimp;
            rates = ratesp;
            speeds = speedsp;
            cores = coresp;

            // cout << "cores[0]': " << cores[0] << endl;

            maxsize = maxsizep;
            arrivalrate = arrivalratep;
            departurerate = departureratep;
            cutoff = cutoffp;
            eps = epsp;
            // cout << "B" << endl;

            build_markov_chain();
            cout << "Done setting up the MDP solver." << endl;

            // vector<vector<vector<int>>> partitions;
            // adjlist = buildMarkovChain();
        }

        void SA_general(){
            int n = 0;
            tuple<double, double, double> cur_eps = {2*eps, 0, 2*eps};
            // cout << "Hi" << endl;
            // cout << eps << ", " << cur_eps << endl;
            while (get<0>(cur_eps) > eps){
                // cur_eps = SAStep();
                cur_eps = SAStepversion2();
                cout << "n: " << n << ", cur eps: " << get<0>(cur_eps) << ", min_change: " << get<1>(cur_eps) << ", max_change: " << get<2>(cur_eps) << endl;
                n++;
            }
            cout << "Storing data..." << endl;
            SAStepversion2savedata();
            cout << "Data stored!" << endl;
        }

        void save_actions(string file_name = "action_data.txt"){
            cout << "Saving data..." << endl;
            ofstream action_data (file_name);
            // ofstream compressed_data ("compressed_states_data.txt");
            // cout << "cut off: " << cutoff << endl;
            // vector<double> compressed_probs (cutoff+15,0);
            // cout << "compressed_probs size: " << compressed_probs.size() << endl;
            if (action_data.is_open()){
                action_data << "Index,Action number, actual_action dim1, actual_action dim2, cores in use dim1, cores in use dim2, number of jobs allocated, jobs in the system \n";
                for (int i = 0; i < actions.size(); i++){
                    action_data << i << "," << actions[i] << ",";
                    for (int j = 0; j < partition_chosen[i].size(); j++)
                    {
                        for (int k = 0; k < partition_chosen[i][j].size()-1; k++){
                            action_data << partition_chosen[i][j][k] << ";";
                        }
                        action_data << partition_chosen[i][j][partition_chosen[i][j].size()-1] << ",";
                    }
                    for (int j = 0; j < num_cores_in_use[i].size(); j++)
                    {
                        action_data << num_cores_in_use[i][j] << ",";
                    }
                    action_data << num_jobs_allocated[i] << "," << jobs_in_system[i] << "\n";
                }
                action_data.close();
            }
            else{
                cout << "Had problems opening file. :(" << endl;
                exit(1);
            }
            cout << "Data saved" << endl;
        }

        void MonteCarlo(int iters, bool load_actions = false, string actions_file = "action_data.txt"){
            vector<double> probs (actions.size(),0);
            probs[0] = 1;

            if (load_actions){
                ifstream curfile (actions_file);
                if (curfile.is_open()){
                    string data;
                    getline(curfile, data);
                    while (getline(curfile, data)){
                        if (data != ""){
                            std::vector<std::string> tokens;
                            size_t pos = 0;
                            string delimiter = ",";
                            std::string token;
                            while (((pos = data.find(delimiter)) != std::string::npos)){
                                token = data.substr(0, pos);
                                tokens.push_back(token);
                                data.erase(0, pos + delimiter.length());
                            }
                            actions[stoi(tokens[0])] = stoi(tokens[1]);
                        }
                    }
                }
                else{
                    cout << "Not found actual file" << endl;
                }
            }

            // cout << "A" << endl;

            for (int i = 0; i < iters; i++){
                if (i % 1 == 0){
                    cout << "i: " << i << "/" << iters << endl;
                }
                vector<double> new_probs(actions.size(),0);
                for (int node = 0; node < actions.size(); node++){
                    vector<vector<int>> curpartition = index_to_partitions(node);
                    int waiting_jobs = curpartition[0][0];
                    vector<vector<int>> actualpartitions;
                    vector<int> cur_free_cores;

                    // cout << "i: " << i << endl;

                    vector<vector<int>> helper_partition;
                    for (int k = 1; k < dim + 1; k++){
                        helper_partition.push_back(curpartition[k]);
                    }
                    assert(node == calc_index(helper_partition, waiting_jobs)), "Partition's inverse is incorrect!";

                    assert(dim == curpartition.size()-1);

                    int cur_jobs = waiting_jobs;
                    for (int l = 1; l < curpartition.size(); l++){
                        actualpartitions.push_back(curpartition[l]);
                        int cur_sum = 0;
                        for (int k = 0; k < curpartition[l].size(); k++){// Note that there are never jobs allocated to 0 cores. Thus this is stil correct.
                            cur_sum += k*curpartition[l][k];
                            cur_jobs += curpartition[l][k];
                        }

                        assert(cores[l-1] - cur_sum >= 0);

                        cur_free_cores.push_back(cores[l-1] - cur_sum);
                    }
                    // double new_val = 0;
                    double sum_rates = 0;
                    if (actions[node]==0){
                        // cout << "A1" << endl;
                        for (int dimmy = 0; dimmy < dim; dimmy++){
                            for (int k = 0; k < actualpartitions[dimmy].size(); k++){ // Why does this need to be actualpartitions.size()???
                                if (actualpartitions[dimmy][k] == 0){ // Skip positions where we don't have cores.
                                    continue;
                                }
                                vector<vector<int>> copied_partition (actualpartitions);
                                if (k < copied_partition[dimmy].size()){
                                    copied_partition[dimmy][k] -= 1;
                                }

                                new_probs[calc_index(copied_partition, waiting_jobs)] += (actualpartitions[dimmy][k] *
                                            (calcAmdahlsLaw(speeds[dimmy], rates[dimmy], actualpartitions[dimmy][k]))/
                                            uniformization_constant) *
                                            probs[node];
                                sum_rates += actualpartitions[dimmy][k] * (calcAmdahlsLaw(speeds[dimmy], rates[dimmy], actualpartitions[dimmy][k]))/uniformization_constant;
                            }
                        }
                        // Chance of another job arriving.
                        if (waiting_jobs < cutoff){
                            new_probs[calc_index(actualpartitions, waiting_jobs + 1)] += arrivalrate/uniformization_constant * probs[node];
                            sum_rates += arrivalrate/uniformization_constant;
                        }

                        // cout << "sum_rates: " << sum_rates << endl;

                        // Chance nothing changes in the time step.
                        new_probs[node] += (1-sum_rates)*probs[node];
                    }
                    else{
                        int j = actions[node];
                        j--;
                        // cout << "A2" << endl;

                        vector<int> possible_allocations (cur_free_cores);
                        for (int k = 0; k < dim; k++){
                            possible_allocations[k] = min(possible_allocations[k], waiting_jobs);
                        }
                        int cur_dim = 1;


                        vector<int> helper_dims;
                        for (int k = 0; k < dim; k++){
                            if (possible_allocations[k] > 0){
                                cur_dim *= (1+partitionsizes[possible_allocations[k]-1]);
                                helper_dims.push_back(1+partitionsizes[possible_allocations[k]-1]);
                            }
                            else{
                                cur_dim *= 1;
                                helper_dims.push_back(1);
                            }
                        }

                        // cout << "A3" << endl;

                        vector<int> allocation = SA_index_to_partition_indices(j, helper_dims);
                        vector<vector<int>> allocations = SAindex_to_partitions(allocation);
                        vector<vector<int>> non_updated_allocation(actualpartitions);

                        int sum_allocs = 0;

                        for (int k = 0; k < allocations.size(); k++){
                            vector<int> allocationss = allocations[k];
                            if (allocationss.size() != 1 || allocationss[0] != 0){
                                if (allocationss.size() > non_updated_allocation[k].size()){
                                    int total_size = non_updated_allocation[k].size();
                                    for (int l = 0; l < allocationss.size()-total_size; l++){
                                        non_updated_allocation[k].push_back(0);
                                    }
                                }
                                for (int l = 0; l < allocationss.size(); l++){
                                    non_updated_allocation[k][l] += allocationss[l];
                                    sum_allocs += l*allocationss[l];
                                }
                            }
                            if (sum_allocs > waiting_jobs){
                                break;
                            }
                        }

                        if (sum_allocs > waiting_jobs){
                            continue;
                        }

                        // cout << "A4" << endl;

                        int after_alloc_waiting_jobs = waiting_jobs - sum_allocs;

                        assert(after_alloc_waiting_jobs >= 0);

                        double new_val = after_alloc_waiting_jobs;
                        double sum_rates = 0;

                        for (int dimmy = 0; dimmy < dim; dimmy++){
                            for (int k = 0; k < non_updated_allocation[dimmy].size(); k++){ // Why does this need to be actualpartitions.size()???
                                if (non_updated_allocation[dimmy][k] == 0){ // Skip positions where we don't have cores.
                                    continue;
                                }
                                vector<vector<int>> copied_partition (non_updated_allocation);
                                copied_partition[dimmy][k] -= 1;
                                double addition = (non_updated_allocation[dimmy][k] *
                                            (calcAmdahlsLaw(speeds[dimmy], rates[dimmy], k))/
                                            uniformization_constant);
                                new_probs[calc_index(copied_partition, after_alloc_waiting_jobs)] += addition * probs[node];
                                sum_rates += addition;
                            }
                        }

                        // cout << "A5" << endl;

                        // Chance of another job arriving.
                        if (after_alloc_waiting_jobs < cutoff){
                            new_probs[calc_index(non_updated_allocation, after_alloc_waiting_jobs+1)] += arrivalrate/uniformization_constant * probs[node];
                            sum_rates += arrivalrate/uniformization_constant;
                        }

                        assert(sum_rates <= 1);

                        new_probs[calc_index(non_updated_allocation, after_alloc_waiting_jobs)] += (1-sum_rates)*probs[node];

                        // cout << "A6" << endl;
                    }
                }
                probs = new_probs;
            }

            // cout << "B" << endl;

            ofstream whole_data ("whole_states_data.txt");
            ofstream compressed_data ("compressed_states_data.txt");
            cout << "cut off: " << cutoff << endl;
            vector<double> compressed_probs (cutoff+15,0);
            cout << "compressed_probs size: " << compressed_probs.size() << endl;
            if (whole_data.is_open()){
                whole_data << "Index,Chance\n";
                for (int i = 0; i < actions.size(); i++){
                    vector<vector<int>> curpartition = index_to_partitions(i);
                    int waiting_jobs = curpartition[0][0];
                    vector<vector<int>> actualpartitions;
                    vector<int> cur_free_cores;

                    // cout << "i: " << i << endl;

                    vector<vector<int>> helper_partition;
                    for (int j = 1; j < dim + 1; j++){
                        helper_partition.push_back(curpartition[j]);
                    }
                    assert(i == calc_index(helper_partition, waiting_jobs)), "Partition's inverse is incorrect!";

                    assert(dim == curpartition.size()-1);

                    int cur_jobs = waiting_jobs;
                    for (int j = 1; j < curpartition.size(); j++){
                        actualpartitions.push_back(curpartition[j]);
                        int cur_sum = 0;
                        for (int k = 0; k < curpartition[j].size(); k++){// Note that there are never jobs allocated to 0 cores. Thus this is stil correct.
                            cur_sum += k*curpartition[j][k];
                            cur_jobs += curpartition[j][k];
                        }

                        assert(cores[j-1] - cur_sum >= 0);

                        cur_free_cores.push_back(cores[j-1] - cur_sum);
                    }
                    compressed_probs[cur_jobs] += probs[i];

                    whole_data << i << "," << probs[i] << "\n";
                    // if (probs[i] != 0){
                    //     cout << "Hi" << endl;
                    // }
                }
                whole_data.close();
            }
            else{
                cout << "Could not open file. Aborting..." << endl;
                exit(1);
            }
            cout << "compressed_probs size: " << compressed_probs.size() << endl;
            if (compressed_data.is_open()){
                compressed_data << "Jobs,Chance\n";
                for (int i = 0; i < compressed_probs.size(); i++){
                     compressed_data << i << "," << compressed_probs[i] << "\n";
                }
                compressed_data.close();
            }
            else{
                cout << "Could not open file. Aborting..." << endl;
                exit(1);
            }
        }

    private:

        /*
        @brief
        Builds the "adjacency graph" for the MDP.

        In:
            - N/A

        Out:
            - N/A

        Side effects:

            - Creates the correct number of partitions;

            - Calculates the corresponding dimensions;

            - Creates the correct value function;

            - Creates the correct uniformisation constant.

        */
        void build_markov_chain(){
            vector<vector<vector<int>>> mpartitions = calc_partitions(cores[0]);
            for (int i = 0; i < dim ; i++){
                mpartitions = calc_more_partitions(mpartitions, cores[i]);
            }
            mpartitions = collapse_partitions(mpartitions, cutoff);


            partitions = mpartitions;
            vector<int> cum_size;
            cum_size.push_back(mpartitions[0].size());
            for (int i = 1; i < mpartitions.size(); i++){
                cum_size.push_back(cum_size[i-1] + mpartitions[i].size());
            }

            partitionsizes = cum_size;
            int actualdims = cutoff + 1;

            dims.push_back(cutoff + 1);
            for (int core : cores){
                // Include the 0 "partition" as well.
                // Assumes no zero cores existing
                dims.push_back(cum_size[core-1]+1);
                actualdims *= cum_size[core-1] + 1;
            }
            valuefunc = vector<double> (actualdims, 0);

            cout << "Total array size: " << actualdims << endl;

            if (cutoff <= 0){
                cout << "Please make sure to at least allow for at least one item to wait in a queue" << endl;
                exit(1);
            }
            uniformization_constant = arrivalrate; // Assuming that we can have at least one item in the queue
            for (int i = 0; i < dim; i++){
                uniformization_constant += cores[i] * calcAmdahlsLaw(speeds[i], rates[i], 1);
            }

            actions = vector<int>(actualdims, 0);
            num_jobs_allocated = vector<int>(actualdims, 0);
            partition_chosen = vector<vector<vector<int>>>(actualdims);
            jobs_in_system = vector<int>(actualdims, 0);
            num_cores_in_use = vector<vector<int>>(actualdims);
        }


        /*
        @brief Calculates the actual index of a goal we try and achieve. For this
        we calculate the indices for all the different types of cores using
        binsearch_partitions.

        In:
            - @param goals: the goals for all the different indices of type i;
            - @param waiting: the number of waiting tasks.

        Out:
            - @return actualindex, the index in the multi-dimensional core space

        @see binsearch_partitions.
        */
        int calc_index(vector<vector<int>> goals, int waiting){
            vector<int> indices;
            indices.push_back(waiting);
            for (vector<int> goal: goals){
                indices.push_back(binsearch_partitions(goal));
            }

            int actualindex = 0;
            int curdim = 1;
            for (int i = 0; i < indices.size(); i++){
                int index = indices[i];
                actualindex += index*curdim;
                curdim *= dims[i];
            }

            return actualindex;
        }

        /*
        @brief Calculate the value of the value func at a specific partition.

        In:
            - @param goals: The total multi-dimensional partition set;
            - @param waiting: The total number of waiting jobs;

        Out:
            - @return value of the value func at the index of the multi-dimensional
                    partition set.
        */
        double get_valuefunc(vector<vector<int>> goals, int waiting){
            return valuefunc[calc_index(goals, waiting)];
        }

        /*
        @brief Calculates the index of a partition in the total partition set.

        In:
            - @param goal: The partition to calculate the index for;

        Out:
            - @return lower: the index of the partition.
        */
        int binsearch_partitions(vector<int> goal){
            int sum{0};
            int sum2{0};
            int numzero = goal[0];
            for (int i = 0; i < goal.size(); i++){
                sum += i*goal[i]; // Check if goal is flattened array or not????? 29-5
            }
            if (sum == 0){
                return 0;
            }
            sum--;
            // Note that the partitions are sorted backwards i.e. it's sorted with by the highest elements first.
            int lower = 0;
            int heigher = partitions[sum].size()-1;
            int mid = (lower + heigher)/2;

            while (lower != heigher - 1 && lower != heigher){
                mid = (lower + heigher)/2;
                if (goal.size() > partitions[sum][mid].size()){ // Make sure that the heighest value is at least the same.
                    heigher = mid;
                    continue;
                }
                else if (goal.size() < partitions[sum][mid].size()){ // Make sure that the heighest value is at least the same.
                    lower = mid;
                    continue;
                }
                bool found_something = false;
                for (int i = goal.size()-1; i > 0; i--){ // Skip 0, because it doesn't matter, since it is already determined by the rest.
                    // I am confused by the usage if sum here 29-5. Think I fixed it 29-5.
                    if (goal[i] > partitions[sum][mid][i]){
                        // If any index has less than our current cores allocated
                        // we are already looking at an index way further down the
                        // stack and therefore we are looking too far.
                        heigher = mid;
                        found_something = true;
                        break;
                    }
                    else if (goal[i] < partitions[sum][mid][i]){
                        // If any index has more than our current cores allocated
                        // we are looking at an index way to soon and therefore we
                        // should looking at a further index.
                        lower = mid;
                        found_something = true;
                        break;
                    }
                }
                // In the specific instance that we won't have any differing values
                // from mid, the value is exactly mid.
                if (!found_something){
                    if (sum == 0){
                        return mid + 1;
                    }
                    return mid+1+partitionsizes[sum-1];
                }
            }
            if (heigher < partitions[sum].size()){
                bool possible_heigher = true;
                if (goal.size() > partitions[sum][heigher].size()){ // Make sure that the heighest value is at least the same.
                    possible_heigher = false;
                }
                else if (goal.size() < partitions[sum][heigher].size()){ // Make sure that the heighest value is at least the same.
                    possible_heigher = false;
                }
                bool found_something = false;
                for (int i = goal.size()-1; i > 0 && possible_heigher; i--){ // Skip 0, because it doesn't matter, since it is already determined by the rest.
                    // I am confused by the usage if sum here 29-5. Think I fixed it 29-5.
                    if (goal[i] > partitions[sum][heigher][i]){
                        // If any index has less than our current cores allocated
                        // we are already looking at an index way further down the
                        // stack and therefore we are looking too far.
                        found_something = true;
                        break;
                    }
                    else if (goal[i] < partitions[sum][heigher][i]){
                        // If any index has more than our current cores allocated
                        // we are looking at an index way to soon and therefore we
                        // should looking at a further index.
                        found_something = true;
                        break;
                    }
                }
                if (!found_something && possible_heigher){
                    if (sum == 0){
                        return heigher + 1;
                    }
                    return heigher+1+partitionsizes[sum-1];
                }
            }
            if (sum == 0){
                return lower + 1;
            }
            return lower+1+partitionsizes[sum-1];
        }

        /*
        @brief Calculates the multi-dimensional partition indices of a one-dimensional
        index.

        In:
            - @param index: The one-dimensional index to split up into dim-dimensional
                indices;

        Out:
            - @return goal: The multi-dimensional partition indices.
        */
        vector<int> index_to_partition_indices(int index){
            vector<int> goal;
            int curdim = 1;
            int prevdim = curdim;
            for (int mydim : dims){
                curdim *= mydim;
                goal.push_back((index % curdim)/prevdim);
                prevdim = curdim;
            }
            return goal;
        }

        /*
        @brief Calculates the multi-dimensional partition values.

        In:
            - @param index: The one-dimensional index to split up into dim-dimensional
                partitions.

        Out:
            - @return goals: The dim-dimensional list of partitions.
        */
        vector<vector<int>> index_to_partitions(int index){
            vector<int> goal_indices = index_to_partition_indices(index);
            vector<vector<int>> goals;

            goals.push_back({{goal_indices[0]}}); // The first value isn't a partition.
                                                  // It is merely the number of waiting jobs.
            for (int i = 1; i < goal_indices.size(); i++){
                int goal_index = goal_indices[i];
                if (goal_index == 0){
                    goals.push_back({{0}});
                    continue;
                }
                int num_cores_working = -1;
                for (int j = 0; j < partitionsizes.size(); j++){
                    if (partitionsizes[j] > goal_index-1){
                        num_cores_working = j;
                        break;
                    }
                }
                if (num_cores_working == -1){
                    cout << "Not found a partition containing this index, aborting..." << endl;
                    exit(1);
                }
                if (num_cores_working == 0){
                    goals.push_back(partitions[num_cores_working][goal_index-1]);
                }
                else{
                    goals.push_back(partitions[num_cores_working][goal_index - 1 - partitionsizes[num_cores_working-1]]);
                }
            }
            return goals;
        }

        vector<int> SA_index_to_partition_indices(int index, vector<int> curdims){
            vector<int> goal;
            int curdim = 1;
            int prevdim = curdim;
            for (int mydim : curdims){
                curdim *= mydim;
                goal.push_back((index % curdim)/prevdim);
                prevdim = curdim;
            }
            return goal;
        }

        vector<vector<int>> SAindex_to_partitions(vector<int> goal_indices){
            vector<vector<int>> goals;

            for (int i = 0; i < goal_indices.size(); i++){
                int goal_index = goal_indices[i];
                if (goal_index == 0){
                    goals.push_back({{0}});
                    continue;
                }
                int num_cores_working = -1;
                for (int j = 0; j < partitionsizes.size(); j++){
                    if (partitionsizes[j] > goal_index-1){
                        num_cores_working = j;
                        break;
                    }
                }
                if (num_cores_working == -1){
                    cout << "Not found a partition containing this index, aborting..." << endl;
                    exit(1);
                }
                if (num_cores_working == 0){
                    goals.push_back(partitions[num_cores_working][goal_index-1]);
                }
                else{
                    goals.push_back(partitions[num_cores_working][goal_index - 1 - partitionsizes[num_cores_working-1]]);
                }
            }
            return goals;
        }

        tuple<double, double, double> SAStepversion2(){
            vector<double> newvaluefunc(valuefunc.size(), 0);

            double min_change = 1e20;
            double max_change = 0;
            // We maybe need to do choice in another time step.
            // This is due to the fact that we now actually don't make instantaneous
            // decisions.

            for (int i = 0; i < valuefunc.size(); i++){

                // cout << "A" << endl;
                if (i % 10000 == 0){
                    cout << "Done: " << i/(double)valuefunc.size() << "%" << endl;
                }
                double cur_best_choice = 1e20;
                vector<vector<int>> curpartition = index_to_partitions(i);
                int waiting_jobs = curpartition[0][0];
                vector<vector<int>> actualpartitions;
                vector<int> cur_free_cores;

                // cout << "i: " << i << endl;

                vector<vector<int>> helper_partition;
                for (int j = 1; j < dim + 1; j++){
                    helper_partition.push_back(curpartition[j]);
                }
                assert(i == calc_index(helper_partition, waiting_jobs)), "Partition's inverse is incorrect!";

                assert(dim == curpartition.size()-1);

                int cur_jobs = waiting_jobs;
                for (int j = 1; j < curpartition.size(); j++){
                    actualpartitions.push_back(curpartition[j]);
                    int cur_sum = 0;
                    for (int k = 0; k < curpartition[j].size(); k++){// Note that there are never jobs allocated to 0 cores. Thus this is stil correct.
                        cur_sum += k*curpartition[j][k];
                        cur_jobs += curpartition[j][k];
                    }

                    assert(cores[j-1] - cur_sum >= 0);

                    cur_free_cores.push_back(cores[j-1] - cur_sum);
                }

                // Allocate new jobs to cores.
                { // Just some scoping :)
                    vector<int> possible_allocations (cur_free_cores);
                    for (int k = 0; k < dim; k++){
                        possible_allocations[k] = min(possible_allocations[k], waiting_jobs);
                    }
                    int cur_dim = 1;
                    vector<int> helper_dims;
                    for (int k = 0; k < dim; k++){
                        if (possible_allocations[k] > 0){
                            cur_dim *= (1+partitionsizes[possible_allocations[k]-1]);
                            helper_dims.push_back(1+partitionsizes[possible_allocations[k]-1]);
                        }
                        else{
                            cur_dim *= 1;
                            helper_dims.push_back(1);
                        }
                    }

                    for (int j = 0; j < cur_dim; j++){
                        vector<int> allocation = SA_index_to_partition_indices(j, helper_dims);
                        vector<vector<int>> allocations = SAindex_to_partitions(allocation);
                        vector<vector<int>> non_updated_allocation(actualpartitions);

                        int sum_allocs = 0;

                        for (int k = 0; k < allocations.size(); k++){
                            vector<int> allocationss = allocations[k];
                            if (allocationss.size() != 1 || allocationss[0] != 0){
                                if (allocationss.size() > non_updated_allocation[k].size()){
                                    int total_size = non_updated_allocation[k].size();
                                    for (int l = 0; l < allocationss.size()-total_size; l++){
                                        non_updated_allocation[k].push_back(0);
                                    }
                                }
                                for (int l = 0; l < allocationss.size(); l++){
                                    non_updated_allocation[k][l] += allocationss[l];
                                    sum_allocs += l*allocationss[l];
                                }
                            }
                            if (sum_allocs > waiting_jobs){
                                break;
                            }
                        }

                        if (sum_allocs > waiting_jobs){
                            continue;
                        }

                        int after_alloc_waiting_jobs = waiting_jobs - sum_allocs;

                        assert(after_alloc_waiting_jobs >= 0);

                        double new_val = after_alloc_waiting_jobs;
                        double sum_rates = 0;

                        for (int dimmy = 0; dimmy < dim; dimmy++){
                            for (int k = 0; k < non_updated_allocation[dimmy].size(); k++){ // Why does this need to be actualpartitions.size()???
                                if (non_updated_allocation[dimmy][k] == 0){ // Skip positions where we don't have cores.
                                    continue;
                                }
                                vector<vector<int>> copied_partition (non_updated_allocation);
                                copied_partition[dimmy][k] -= 1;
                                double addition = (non_updated_allocation[dimmy][k] *
                                            (calcAmdahlsLaw(speeds[dimmy], rates[dimmy], k))/
                                            uniformization_constant);
                                new_val += addition * get_valuefunc(copied_partition, after_alloc_waiting_jobs);
                                sum_rates += addition;
                            }
                        }

                        // Chance of another job arriving.
                        if (after_alloc_waiting_jobs < cutoff){
                            new_val += arrivalrate/uniformization_constant * get_valuefunc(non_updated_allocation, after_alloc_waiting_jobs+1);
                            sum_rates += arrivalrate/uniformization_constant;
                        }

                        assert(sum_rates <= 1);

                        new_val += (1-sum_rates)*get_valuefunc(non_updated_allocation, after_alloc_waiting_jobs);
                        if (new_val < cur_best_choice){
                            actions[i] = j+1;
                            cur_best_choice = new_val;
                        }
                    }

                }

                // Don't allocate new jobs to cores
                double new_val = cur_jobs;
                double sum_rates = 0;
                for (int dimmy = 0; dimmy < dim; dimmy++){
                    for (int k = 0; k < actualpartitions[dimmy].size(); k++){ // Why does this need to be actualpartitions.size()???
                        if (actualpartitions[dimmy][k] == 0){ // Skip positions where we don't have cores.
                            continue;
                        }
                        vector<vector<int>> copied_partition (actualpartitions);
                        if (k < copied_partition[dimmy].size()){
                            copied_partition[dimmy][k] -= 1;
                        }
                        double addition = (actualpartitions[dimmy][k] *
                                    (calcAmdahlsLaw(speeds[dimmy], rates[dimmy], k))/
                                    uniformization_constant);
                        new_val += addition * get_valuefunc(copied_partition, waiting_jobs);
                        sum_rates += addition;
                    }
                }
                // Chance of another job arriving.
                if (waiting_jobs < cutoff){
                    new_val += arrivalrate/uniformization_constant * get_valuefunc(actualpartitions, waiting_jobs + 1);
                    sum_rates += arrivalrate/uniformization_constant;
                }

                assert(sum_rates <= 1);

                // Chance nothing changes in the time step.
                new_val += (1-sum_rates)*get_valuefunc(actualpartitions, waiting_jobs);

                if (new_val < cur_best_choice){
                    actions[i] = 0;
                    cur_best_choice = new_val;
                }
                newvaluefunc[i] = cur_best_choice;
                if (newvaluefunc[i] - valuefunc[i] < min_change){
                    min_change = newvaluefunc[i] - valuefunc[i];
                }
                if (newvaluefunc[i] - valuefunc[i] > max_change){
                    max_change = newvaluefunc[i] - valuefunc[i];
                }
            }
            // for (int i = 0; i < valuefunc.size(); i++){
            //     newvaluefunc[i] -= min_change;
            // }

            valuefunc = newvaluefunc;

            assert(max_change >= min_change);

            return {max_change - min_change, min_change, max_change};
        }

        tuple<double, double, double> SAStepversion2savedata(){
            vector<double> newvaluefunc(valuefunc.size(), 0);

            double min_change = 1e20;
            double max_change = 0;
            // We maybe need to do choice in another time step.
            // This is due to the fact that we now actually don't make instantaneous
            // decisions.

            for (int i = 0; i < valuefunc.size(); i++){

                // cout << "A" << endl;
                if (i % 10000 == 0){
                    cout << "Done: " << i/(double)valuefunc.size() << "%" << endl;
                }
                double cur_best_choice = 1e20;
                vector<vector<int>> curpartition = index_to_partitions(i);
                int waiting_jobs = curpartition[0][0];
                vector<vector<int>> actualpartitions;
                vector<int> cur_free_cores;

                // cout << "i: " << i << endl;

                vector<vector<int>> helper_partition;
                for (int j = 1; j < dim + 1; j++){
                    helper_partition.push_back(curpartition[j]);
                }
                assert(i == calc_index(helper_partition, waiting_jobs)), "Partition's inverse is incorrect!";

                assert(dim == curpartition.size()-1);

                int cur_jobs = waiting_jobs;
                for (int j = 1; j < curpartition.size(); j++){
                    actualpartitions.push_back(curpartition[j]);
                    int cur_sum = 0;
                    for (int k = 0; k < curpartition[j].size(); k++){// Note that there are never jobs allocated to 0 cores. Thus this is stil correct.
                        cur_sum += k*curpartition[j][k];
                        cur_jobs += curpartition[j][k];
                    }
                    num_cores_in_use[i].push_back(cur_sum);

                    assert(cores[j-1] - cur_sum >= 0);

                    cur_free_cores.push_back(cores[j-1] - cur_sum);
                }

                // Allocate new jobs to cores.
                { // Just some scoping :)
                    vector<int> possible_allocations (cur_free_cores);
                    for (int k = 0; k < dim; k++){
                        possible_allocations[k] = min(possible_allocations[k], waiting_jobs);
                    }
                    int cur_dim = 1;
                    vector<int> helper_dims;
                    for (int k = 0; k < dim; k++){
                        if (possible_allocations[k] > 0){
                            cur_dim *= (1+partitionsizes[possible_allocations[k]-1]);
                            helper_dims.push_back(1+partitionsizes[possible_allocations[k]-1]);
                        }
                        else{
                            cur_dim *= 1;
                            helper_dims.push_back(1);
                        }
                    }

                    for (int j = 0; j < cur_dim; j++){
                        vector<int> allocation = SA_index_to_partition_indices(j, helper_dims);
                        vector<vector<int>> allocations = SAindex_to_partitions(allocation);
                        vector<vector<int>> non_updated_allocation(actualpartitions);

                        int sum_allocs = 0;

                        for (int k = 0; k < allocations.size(); k++){
                            vector<int> allocationss = allocations[k];
                            if (allocationss.size() != 1 || allocationss[0] != 0){
                                if (allocationss.size() > non_updated_allocation[k].size()){
                                    int total_size = non_updated_allocation[k].size();
                                    for (int l = 0; l < allocationss.size()-total_size; l++){
                                        non_updated_allocation[k].push_back(0);
                                    }
                                }
                                for (int l = 0; l < allocationss.size(); l++){
                                    non_updated_allocation[k][l] += allocationss[l];
                                    sum_allocs += l*allocationss[l];
                                }
                            }
                            if (sum_allocs > waiting_jobs){
                                break;
                            }
                        }

                        if (sum_allocs > waiting_jobs){
                            continue;
                        }

                        int after_alloc_waiting_jobs = waiting_jobs - sum_allocs;

                        assert(after_alloc_waiting_jobs >= 0);

                        double new_val = after_alloc_waiting_jobs;
                        double sum_rates = 0;

                        for (int dimmy = 0; dimmy < dim; dimmy++){
                            for (int k = 0; k < non_updated_allocation[dimmy].size(); k++){ // Why does this need to be actualpartitions.size()???
                                if (non_updated_allocation[dimmy][k] == 0){ // Skip positions where we don't have cores.
                                    continue;
                                }
                                vector<vector<int>> copied_partition (non_updated_allocation);
                                copied_partition[dimmy][k] -= 1;
                                double addition = (non_updated_allocation[dimmy][k] *
                                            (calcAmdahlsLaw(speeds[dimmy], rates[dimmy], k))/
                                            uniformization_constant);
                                new_val += addition * get_valuefunc(copied_partition, after_alloc_waiting_jobs);
                                sum_rates += addition;
                            }
                        }

                        // Chance of another job arriving.
                        if (after_alloc_waiting_jobs < cutoff){
                            new_val += arrivalrate/uniformization_constant * get_valuefunc(non_updated_allocation, after_alloc_waiting_jobs+1);
                            sum_rates += arrivalrate/uniformization_constant;
                        }

                        assert(sum_rates <= 1);

                        new_val += (1-sum_rates)*get_valuefunc(non_updated_allocation, after_alloc_waiting_jobs);
                        if (new_val < cur_best_choice){
                            actions[i] = j+1;
                            num_jobs_allocated[i] = sum_allocs;
                            partition_chosen[i] = allocations;
                            cur_best_choice = new_val;
                            jobs_in_system[i] = after_alloc_waiting_jobs;
                        }
                    }

                }

                // Don't allocate new jobs to cores
                double new_val = cur_jobs;
                double sum_rates = 0;
                for (int dimmy = 0; dimmy < dim; dimmy++){
                    for (int k = 0; k < actualpartitions[dimmy].size(); k++){ // Why does this need to be actualpartitions.size()???
                        if (actualpartitions[dimmy][k] == 0){ // Skip positions where we don't have cores.
                            continue;
                        }
                        vector<vector<int>> copied_partition (actualpartitions);
                        if (k < copied_partition[dimmy].size()){
                            copied_partition[dimmy][k] -= 1;
                        }
                        double addition = (actualpartitions[dimmy][k] *
                                    (calcAmdahlsLaw(speeds[dimmy], rates[dimmy], k))/
                                    uniformization_constant);
                        new_val += addition * get_valuefunc(copied_partition, waiting_jobs);
                        sum_rates += addition;
                    }
                }
                // Chance of another job arriving.
                if (waiting_jobs < cutoff){
                    new_val += arrivalrate/uniformization_constant * get_valuefunc(actualpartitions, waiting_jobs + 1);
                    sum_rates += arrivalrate/uniformization_constant;
                }

                assert(sum_rates <= 1);

                // Chance nothing changes in the time step.
                new_val += (1-sum_rates)*get_valuefunc(actualpartitions, waiting_jobs);

                if (new_val < cur_best_choice){
                    actions[i] = 0;
                    cur_best_choice = new_val;
                    num_jobs_allocated[i] = 0;
                    vector<vector<int>> allocations = vector<vector<int>>(dim, {0});
                    partition_chosen[i] = allocations;
                    cur_best_choice = new_val;
                    jobs_in_system[i] = waiting_jobs;
                }
                newvaluefunc[i] = cur_best_choice;
                if (newvaluefunc[i] - valuefunc[i] < min_change){
                    min_change = newvaluefunc[i] - valuefunc[i];
                }
                if (newvaluefunc[i] - valuefunc[i] > max_change){
                    max_change = newvaluefunc[i] - valuefunc[i];
                }
            }
            // for (int i = 0; i < valuefunc.size(); i++){
            //     newvaluefunc[i] -= min_change;
            // }

            valuefunc = newvaluefunc;

            assert(max_change >= min_change);

            return {max_change - min_change, min_change, max_change};
        }

        double SAStep(){
            vector<double> newvaluefunc(valuefunc.size(), 0);

            double min_change = 1e20;
            double max_change = 0;
            // We maybe need to do choice in another time step.
            // This is due to the fact that we now actually don't make instantaneous
            // decisions.

            for (int i = 0; i < valuefunc.size(); i++){

                if (i % 10000 == 0){
                    cout << "Done: " << i/(double)valuefunc.size() << "%" << endl;
                }
                // cout << i << endl;
                double cur_best_choice = 1e20;
                // cout << "A'" << endl;
                vector<vector<int>> curpartition = index_to_partitions(i);
                // cout << "A" << endl;
                int waiting_jobs = curpartition[0][0];
                vector<vector<int>> actualpartitions;
                vector<int> cur_free_cores;

                vector<vector<int>> helper_partition;
                for (int j = 1; j < dim + 1; j++){
                    helper_partition.push_back(curpartition[j]);
                }
                cout << "i: " << i << ", calc_index: " << calc_index(helper_partition, waiting_jobs) << endl;
                // cout << "Waiting jobs: " << waiting_jobs << endl;
                // for (vector<int> partition : helper_partition){
                //     cout << "Partition:";
                //     for (int val : partition){
                //         cout << ", " << val;
                //     }
                //     cout << ", Calculated partition index: " << binsearch_partitions(partition);
                //     cout << endl;
                // }
                // cout << "Partition size[0] : " << partitionsizes[0] << endl;
                // cout << "-- Test run of binary search partitions --" << endl;
                // cout << binsearch_partitions({0,2}) << endl;

                assert(i == calc_index(helper_partition, waiting_jobs)), "Partition's inverse is incorrect!";

                assert(dim == curpartition.size()-1);

                int cur_jobs = waiting_jobs;
                for (int j = 1; j < curpartition.size(); j++){
                    actualpartitions.push_back(curpartition[j]);
                    int cur_sum = 0;
                    for (int k = 0; k < curpartition[j].size(); k++){// Note that there are never jobs allocated to 0 cores. Thus this is stil correct.
                        cur_sum += k*curpartition[j][k];
                        cur_jobs += curpartition[j][k];
                    }

                    assert(cores[j-1] - cur_sum >= 0);

                    cur_free_cores.push_back(cores[j-1] - cur_sum);
                    // if (i == 40){
                    //     cout << cores[j-1] - cur_sum << endl;
                    // }
                }
                // if (i == 40){
                //     cout << "Cur partition size: " << curpartition.size() << endl;
                // }
                vector<vector<int>> partitions;

                // cout << "Cur_free_cores_size: " << cur_free_cores.size() << endl;
                // cout << "dim: " << dim << ", cur_partition_size: " << curpartition.size() << endl;
                // assert (cur_free_cores.size() == dim);

                // cout << "Cur_waiting: " << waiting_jobs << endl;

                // cout << "a" << endl;

                // Allocate new jobs to cores.
                for (int j = 1; j <= waiting_jobs; j++){
                    for (int dimmy = 1; dimmy < dim; dimmy++){
                        for (int k = 1; k < min(waiting_jobs, cur_free_cores[dimmy]); k++){
                            vector<vector<int>> copied_partition (actualpartitions);
                            // cout << "b" << endl;
                            if (k < copied_partition[dimmy].size()){
                                // cout << "c" << endl;
                                copied_partition[dimmy][k] += 1;
                            }
                            else {
                                vector<int> new_dim_vec(copied_partition[dimmy]);
                                // cout << "d" << endl;
                                for (int l = 0; l < k - copied_partition[dimmy].size() + 1; l++){
                                    new_dim_vec.push_back(0);
                                }
                                new_dim_vec[k] += 1;
                                copied_partition[dimmy] = new_dim_vec;
                            }
                            // cout << "B" << endl;
                            double new_val = cur_jobs + get_valuefunc(copied_partition, waiting_jobs-k);
                            if (new_val < cur_best_choice){
                                cur_best_choice = new_val;
                            }
                        }
                    }
                }

                // cout << "B'" << endl;
                // Don't allocate jobs to cores.
                double new_val = cur_jobs;
                double sum_rates = 0;
                for (int dimmy = 0; dimmy < dim; dimmy++){
                    for (int k = 0; k < actualpartitions[dimmy].size(); k++){ // Why does this need to be actualpartitions.size()???
                        if (actualpartitions[dimmy][k] == 0){ // Skip positions where we don't have cores.
                            continue;
                        }
                        vector<vector<int>> copied_partition (actualpartitions);
                        if (k < copied_partition[dimmy].size()){
                            copied_partition[dimmy][k] -= 1;
                        }
                        new_val += (actualpartitions[dimmy][k] *
                                    (calcAmdahlsLaw(speeds[dimmy], rates[dimmy], actualpartitions[dimmy][k]))/
                                    uniformization_constant) *
                                    get_valuefunc(copied_partition, waiting_jobs);
                        // cout << "C" << endl;
                        // WHY i here????
                        sum_rates += actualpartitions[dimmy][k] * (calcAmdahlsLaw(speeds[dimmy], rates[dimmy], actualpartitions[dimmy][k]))/uniformization_constant;
                        // if (actualpartitions[dimmy][k] * (calcAmdahlsLaw(speeds[dimmy], rates[dimmy], actualpartitions[dimmy][k]))/uniformization_constant > 1e9){
                        //     cout << "Dimmy: " << dimmy << ", k: " << k << endl;
                        // }
                    }
                }
                        // if (new_val < cur_best_choice){
                        //     cur_best_choice = new_val;
                        // }
                // cout << "D" << endl;
                // Chance of another job arriving.
                if (waiting_jobs < cutoff){
                    // cout << "E" << endl;
                    new_val += arrivalrate/uniformization_constant * get_valuefunc(actualpartitions, waiting_jobs + 1);
                    sum_rates += arrivalrate/uniformization_constant;
                }

                // Chance nothing changes in the time step.
                new_val += (1-sum_rates)*get_valuefunc(actualpartitions, waiting_jobs);


                // cout << "new_val is: " << new_val << endl;
                // cout << "sum_rate is: " << sum_rates << endl;

                if (new_val < cur_best_choice){
                    cur_best_choice = new_val;
                }
                newvaluefunc[i] = cur_best_choice;
                if (newvaluefunc[i] - valuefunc[i] < min_change){
                    min_change = newvaluefunc[i] - valuefunc[i];
                }
                if (newvaluefunc[i] - valuefunc[i] > max_change){
                    max_change = newvaluefunc[i] - valuefunc[i];
                }

                // cout << "max_change: " << max_change << ", min_change: " << min_change << endl;
            }
            for (int i = 0; i < valuefunc.size(); i++){
                valuefunc[i] -= min_change;
            }

            valuefunc = newvaluefunc;

            // cout << "max_change: " << max_change << ", min_change: " << min_change << endl;
            assert(max_change >= min_change);

            return max_change - min_change;
        }



};

int main(){
    // Modeled after 12700K: https://en.wikipedia.org/wiki/Alder_Lake with general
    // With maximum departure rate: 8×3,6 + 2,7×4 = 39,6
    // MarkovDP alder_lake = MarkovDP(2, {0.8, 0.8}, {3.6, 2.7}, {8, 4}, 100, 25, 1, 100, 0.0001);


    for (int i = 1; i < 8; i++){
        cout << (8.f/float(i))*calcAmdahlsLaw(3.6, 0.8, i) << endl;
    }

    // MarkovDP alder_lake = MarkovDP(2, {0.8, 0.8}, {3.6, 2.7}, {8, 4}, 100, 25, 1, 100, 100);
    // MarkovDP alder_lake = MarkovDP(2, {0.8, 0.8}, {3.6, 2.7}, {1, 1}, 100, 25, 1, 100, 0.0001);
    // alder_lake.SA_general();
    // alder_lake.save_actions();
    // alder_lake.MonteCarlo(10);
}