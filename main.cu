#include "fm.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "Data.h"
#include "fm_learn.h"

// =================
// Helper Functions
// =================

// I/O routines
void save(std::ofstream& fsave, particle_t* parts, int num_parts, double size) {
    static bool first = true;

    if (first) {
        fsave << num_parts << " " << size << std::endl;
        first = false;
    }

    for (int i = 0; i < num_parts; ++i) {
        fsave << parts[i].x << " " << parts[i].y << std::endl;
    }

    fsave << std::endl;
}


/*
// Particle Initialization
void init_particles(particle_t* parts, int num_parts, double size, int part_seed) {
    std::random_device rd;
    std::mt19937 gen(part_seed ? part_seed : rd());

    int sx = (int)ceil(sqrt((double)num_parts));
    int sy = (num_parts + sx - 1) / sx;

    std::vector<int> shuffle(num_parts);
    for (int i = 0; i < shuffle.size(); ++i) {
        shuffle[i] = i;
    }

    for (int i = 0; i < num_parts; ++i) {
        // Make sure particles are not spatially sorted
        std::uniform_int_distribution<int> rand_int(0, num_parts - i - 1);
        int j = rand_int(gen);
        int k = shuffle[j];
        shuffle[j] = shuffle[num_parts - i - 1];

        // Distribute particles evenly to ensure proper spacing
        parts[i].x = size * (1. + (k % sx)) / (1 + sx);
        parts[i].y = size * (1. + (k / sx)) / (1 + sy);

        // Assign random velocities within a bound
        std::uniform_real_distribution<float> rand_real(-1.0, 1.0);
        parts[i].vx = rand_real(gen);
        parts[i].vy = rand_real(gen);
    }
}
*/


// Command Line Option Processing

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    srand ( time(NULL) );
	try {
		
		const std::string param_task		= "c";
		const std::string param_train_file	= "../data/ml-tag.train";
		const std::string param_test_file	= "../data/ml-tag.test";

		double param_init_stdev	= 0.1;
		int param_num_iter	= 100;
		double param_learn_rate	= 0.01;
		const std::string param_method		= "sgd";

		const std::string param_do_sampling	= "do_sampling";
		const std::string param_do_multilevel	= "do_multilevel";
		const std::string param_num_eval_cases  = "num_eval_cases";
		// (1) Load the data
		std::cout << "Loading train...\t" << std::endl;
		Data train(
			0,
			! (!param_method.compare("mcmc")), // no original data for mcmc
			! (!param_method.compare("sgd") || !param_method.compare("sgda")) // no transpose data for sgd, sgda
		);
		train.load(param_train_file);

		std::cout << "Loading test... \t" << std::endl;
		Data test(
			0,
			! (!param_method.compare("mcmc")), // no original data for mcmc
			! (!param_method.compare("sgd") || !param_method.compare("sgda")) // no transpose data for sgd, sgda
		);
		test.load(param_test_file);

		Data* validation = NULL;

		// (2) Setup the factorization machine
		fm_model fm;
		{
            uint num_all_attribute = std::max(train.num_feature, test.num_feature);
			fm.num_attribute = num_all_attribute;
			fm.init_stdev = param_init_stdev;
			// set the number of dimensions in the factorization
			{ 
				std::vector<int> dim(3);
                dim[0] = 1;
                dim[1] = 1;
                dim[2] = 8;
				assert(dim.size() == 3);
				fm.k0 = dim[0] != 0;
				fm.k1 = dim[1] != 0;
				fm.num_factor = dim[2];					
			}			
			fm.init();		
			
		}

		// (3) Setup the learning method:
		fm_learn* fml;
		if (! param_method.compare("sgd")) {
	 		fml = new fm_learn();
			fml->num_iter = param_num_iter;
		} else {
			throw "unknown method";
		}
		fml->fm = &fm;
		fml->max_target = train.max_target;
		fml->min_target = train.min_target;
		if (! param_task.compare("r") ) {
			fml->task = 0;
		} else if (! param_task.compare("c") ) {
			fml->task = 1;
			for (uint i = 0; i < train.target.dim; i++) { if (train.target(i) <= 0.0) { train.target(i) = -1.0; } else {train.target(i) = 1.0; } }
			for (uint i = 0; i < test.target.dim; i++) { if (test.target(i) <= 0.0) { test.target(i) = -1.0; } else {test.target(i) = 1.0; } }
		} else {
			throw "unknown task";
		}
		
        fml->init();
        // set the regularization; for standard SGD, groups are not supported
        { 
            std::vector<double> reg(3);
            reg[2] == 0.01;
            assert((reg.size() == 0) || (reg.size() == 1) || (reg.size() == 3));
            if (reg.size() == 0) {
                fm.reg0 = 0.0;
                fm.regw = 0.0;
                fm.regv = 0.0;
            } else if (reg.size() == 1) {
                fm.reg0 = reg[0];
                fm.regw = reg[0];
                fm.regv = reg[0];
            } else {
                fm.reg0 = reg[0];
                fm.regw = reg[1];
                fm.regv = reg[2];
            }		
        }
		{
			std::vector<double> lr(1, param_learn_rate);
			assert((lr.size() == 1) || (lr.size() == 3));
			if (lr.size() == 1) {
				fml->learn_rate = lr[0];
				fml->learn_rates.init(lr[0]);
			} else {
				fml->learn_rate = 0;
				fml->learn_rates(0) = lr[0];
				fml->learn_rates(1) = lr[1];
				fml->learn_rates(2) = lr[2];
			}
		}

		auto start_time = std::chrono::steady_clock::now();

		// () learn		
		fml->learn(&train, &test);

		// () Prediction at the end  (not for mcmc and als)
		std::cout << "Final\t" << "Train=" << fml->evaluate(&train) << "\tTest=" << fml->evaluate(&test) << std::endl;

		auto end_time = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff = end_time - start_time;
		double seconds = diff.count();

		// Finalize
		std::cout << "Simulation Time = " << seconds << " seconds \n";
				 	

	} catch (std::string &e) {
		std::cerr << std::endl << "ERROR: " << e << std::endl;
	} catch (char const* &e) {
		std::cerr << std::endl << "ERROR: " << e << std::endl;
	}



}
