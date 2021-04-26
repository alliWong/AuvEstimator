'''
Hyperparameters wrapped in argparse
This file contains most of tuanable parameters   

You can change the values by changing their default fields or by command-line
arguments. For example, "python main.py --filter-scales 2 5 --K 50"
'''

import argparse

# USER INPUT: simulation parameters 
def get_sim():
    parser = argparse.ArgumentParser(description='Simulated ground truth trajectory')
    
    # Time properties
    parser.add_argument('--dt', type = int, default = 0.001,
                        help='simulation time step [s]')
    parser.add_argument('--startTime', type = int, default = 0,
                        help='simulation start time [s]')
    parser.add_argument('--endTime', type = int, default = 50,
                        help='simulation end time [s]')   

    # Underwater vehicle lumped parameters                     
    parser.add_argument('--sim_m', type = int, default = 225,
                        help='mass/inertia [kg]')
    parser.add_argument('--sim_I', type = int, default = 100,
                        help='simulation end time [s]')   
    parser.add_argument('--sim_bxr', type = int, default = 40,
                        help='simulation start time [s]')
    parser.add_argument('--endTime', type = int, default = 50,
                        help='simulation end time [s]')   











    parser.add_argument('--K', type=int, default=10,
                        help='# of words')
    parser.add_argument('--alpha', type=int, default=25,
                        help='Using only a subset of alpha pixels in each image') 

    ## Recognition system (requires tuning)
    parser.add_argument('--L', type=int, default=3,
                        help='# of layers in spatial pyramid matching (SPM)')

    ##
    sim = parser.parse_args()
    return sim