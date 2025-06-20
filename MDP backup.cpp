#include <stdio.h>
#include <vector>
#include <string>
#include <tuple>
#include <utility>
#include <iostream>
#include <assert.h>

// TODO:
//      1. Add binary search indexing of partitions; DONE
//      2. Apply uniformisation;                     ALMOST DONE
//      3. Calculate SA;                             WIP
//      4. Add some fancy way to output vals;
//      5. Create plots in Matplotlib.

//      Check that I do vector initialisation correctly.
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
float calcAmdahlsLaw(float speed, float p, float n){
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
    for (int i = 1; i < n; i++){
        vector<vector<int>> curpartition = prevpartitions[i];
        for (vector<int> parti : curpartition){
            // Always have the sum in sorted order to prevent duplicates.
            if (parti[0] > i){
                break;
            }
            vector<int> newparti = {i};
            for (int item : parti){
                newparti.push_back(i);
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
    vector<vector<vector<int>>> prevpartitions = {{{0}}};
    for (int i = 1; i <= n; i++){
        vector<vector<int>> newpartition = calc_partition(prevpartitions, i);
        prevpartitions.push_back(newpartition);
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
    for (int i = prevpartitions.size(); i <= n; i++){
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
            vector<int> newpartii = vector<int>(0, curpartii[0]+1); // Works due to first value always being highest value due to it being sorted heighest to lowest.
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
    vector<float> rates;
    vector<float> speeds;
    vector<int> cores;
    int maxsize;
    float arrivalrate;
    float departurerate;
    vector<vector<vector<int>>> partitions;
    int cutoff;
    vector<float> valuefunc;
    vector<int> partitionsizes;
    vector<int> dims;
    float uniformization_constant;
    float eps;

    public:
        MarkovDP(int dim, vector<float> rates, vector<float> speeds, vector<int> cores, int maxsize, float arrivalrate, float departurerate, int cutoff, float eps){
            if (!(rates.size() == cores.size() && dim == rates.size())){
                cout << "Please make sure to have all core sizes be identical" << endl;
                exit(1);
            }
            if (dim <= 0){
                cout << "Please make sure to have at least one set of cores" << endl;
                exit(1);
            }
            if (eps <= 0){
                cout << "Please make sure to have eps > 0" << endl;
                exit(1);
            }
            dim = dim;
            rates = rates;
            speeds = speeds;
            cores = cores;
            maxsize = maxsize;
            arrivalrate = arrivalrate;
            departurerate = departurerate;
            cutoff = cutoff;
            eps = eps;
            build_markov_chain();
            // vector<vector<vector<int>>> partitions;
            // adjlist = buildMarkovChain();
        }

        void SA_general(){
            int n = 0;
            int cur_eps = 2*eps;
            while (cur_eps > eps){
                cur_eps = SAStep();
                cout << cur_eps << endl;
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
            // vector<vector<vector<int>>> one_dim_states;
            vector<vector<vector<int>>> mpartitions = calc_partitions(cores[0]);
            for (int i = 0; i < dim ; i++){
                mpartitions = calc_more_partitions(mpartitions, cores[i]);
                // one_dim_states.push_back(partitions[i+1]);
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
                dims.push_back(cum_size[dim]);
                actualdims *= cum_size[dim];
            }
            valuefunc = vector<float> (0, actualdims);

            if (cutoff > 0){
                cout << "Please make sure to at least allow for at least one item to wait in a queue" << endl;
                exit(1);
            }
            uniformization_constant = arrivalrate; // Assuming that we can have at least one item in the queue
            for (int i = 0; i < dim; i++){
                uniformization_constant += cores[i] * calcAmdahlsLaw(speeds[i], rates[i], 1);
            }
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
        float get_valuefunc(vector<vector<int>> goals, int waiting){
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
                sum2 += goal[i];
            }
            // Note that the partitions are sorted backwards i.e. it's sorted with by the highest elements first.
            int lower = 0;
            int heigher = partitions.size()-1;
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
                for (int i = sum2; i > 0; i--){ // Skip 0, because it doesn't matter, since it is already determined by the rest.
                    // I am confused by the usage if sum here 29-5. Think I fixed it 29-5.
                    if (goal[i] > partitions[sum][mid][i]){
                        // If any index has less than our current cores allocated
                        // we are already looking at an index way further down the
                        // stack and therefore we are looking too far.
                        heigher = mid;
                        break;
                    }
                    else if (goal[i] < partitions[sum][mid][i]){
                        // If any index has more than our current cores allocated
                        // we are looking at an index way to soon and therefore we
                        // should looking at a further index.
                        lower = mid;
                        break;
                    }
                }
            }
            return lower;
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
            for (int mydim : dims){
                curdim *= mydim;
                goal.push_back(index % (curdim + 1)); // This might be an off by one error...
                // Need to remove the first dimensions value when looking at the second one.
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
            goals.push_back({{goal_indices[0]}}); // The first value isn't a partition
                                                // It is merely the number of waiting jobs.
            for (int i = 1; i < goal_indices.size(); i++){
                int goal_index = goal_indices[i];
                int num_cores_working = -1;
                for (int i = 0; i < partitionsizes.size(); i++){
                    if (partitionsizes[i] >= goal_index){
                        num_cores_working = i;
                        break;
                    }
                }
                if (num_cores_working == -1){
                    cout << "Not found a partition containing this index, aborting..." << endl;
                    exit(1);
                }
                goals.push_back(partitions[num_cores_working][goal_index-partitionsizes[num_cores_working-1]]);
            }
            return goals;
        }

        float SAStep(){
            vector<float> newvaluefunc(0, valuefunc.size());

            float min_change = 1e20;
            float max_change = 0;
            // We maybe need to do choice in another time step.
            // This is due to the fact that we now actually don't make instantaneous
            // decisions.

            for (int i = 0; i < valuefunc.size(); i++){
                float cur_best_choice = 1e20;
                vector<vector<int>> curpartition = index_to_partitions(i);
                int waiting_jobs = curpartition[0][0];
                vector<vector<int>> actualpartitions;
                vector<int> cur_free_cores;

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
                float cur_val = get_valuefunc(actualpartitions, waiting_jobs);
                vector<vector<int>> partitions;

                // Allocate new jobs to cores.
                for (int j = 1; j <= waiting_jobs; j++){
                    for (int dimmy = 0; dimmy < dim; dim++){
                        for (int k = 1; k < min(waiting_jobs, cur_free_cores[dimmy]); k++){
                            vector<vector<int>> copied_partition (actualpartitions);
                            if (k < copied_partition[dimmy].size()){
                                copied_partition[dimmy][k] += 1;
                            }
                            else {
                                vector<int> new_dim_vec(copied_partition[dimmy]);
                                for (int l = 0; l < copied_partition[dimmy].size() - k + 1; l++){
                                    new_dim_vec.push_back(0);
                                }
                                new_dim_vec[k] += 1;
                                copied_partition[dimmy] = new_dim_vec;
                            }
                            float new_val = cur_jobs + get_valuefunc(copied_partition, waiting_jobs-k);
                            if (new_val < cur_best_choice){
                                cur_best_choice = new_val;
                            }
                        }
                    }
                }

                // Don't allocate jobs to cores.
                float new_val = cur_jobs;
                float sum_rates = 0;
                for (int dimmy = 0; dimmy < dim; dim++){
                    for (int k = 0; k < actualpartitions.size(); k++){
                        if (actualpartitions[dimmy][k] == 0){ // Skip positions where we don't have cores.
                            continue;
                        }
                        vector<vector<int>> copied_partition (actualpartitions);
                        if (k < copied_partition[dimmy].size()){
                            copied_partition[dimmy][k] -= 1;
                        }
                        new_val += (actualpartitions[dimmy][k] * (calcAmdahlsLaw(speeds[i], rates[i], k))/uniformization_constant) * get_valuefunc(actualpartitions, waiting_jobs);
                        sum_rates += actualpartitions[dimmy][k] * (calcAmdahlsLaw(speeds[i], rates[i], k));
                        if (new_val < cur_best_choice){
                            cur_best_choice = new_val;
                        }
                    }
                }
                // Chance of another job arriving.
                if (waiting_jobs < cutoff){
                    new_val += arrivalrate/uniformization_constant * get_valuefunc(actualpartitions, waiting_jobs + 1);
                    sum_rates += arrivalrate/uniformization_constant;
                }

                // Chance nothing changes in the time step.
                new_val += (1-sum_rates)* get_valuefunc(actualpartitions, waiting_jobs);

                if (new_val < cur_best_choice){
                    cur_best_choice = new_val;
                }
                newvaluefunc[i] = new_val;
                if (newvaluefunc[i] - valuefunc[i] < min_change){
                    min_change = newvaluefunc[i] - valuefunc[i];
                }
                if (newvaluefunc[i] - valuefunc[i] > max_change){
                    max_change = newvaluefunc[i] - valuefunc[i];
                }
            }
            valuefunc = newvaluefunc;

            assert(max_change > min_change);

            return max_change - min_change;
        }
};

int main(){
    MarkovDP mymarkov = MarkovDP(1, {0.8}, {1}, {2}, 20, 1.5, 1, 20, 0.1);
    mymarkov.SA_general();
}