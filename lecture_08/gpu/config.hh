#ifndef __CONFIG_H__
#define __CONFIG_H__

//#define THREADS_PER_BLOCK 256

//Set this to 0 if you want to turn off debugging mode
#define DEBUG 0

//Set this to 0 if you want to turn off max threading mode 
// to do block scaling experiments. Or set this to 1 and pass 
// in THREADS_PER_BLOCK as a paramter to CGSolver to do Thread Scaling Experiments
#define MAX_THREADED_MODE 1

//Set this line to 0 if you want to turn off error checking.
// May improve speed
#define ERROR_CHECKING 0

//Setting this to 1 will suppress any uneeded print statements in main
#define SCALING_EXPERIMENTS 1


#endif