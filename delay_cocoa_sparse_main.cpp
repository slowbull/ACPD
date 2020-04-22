/* main function for aynchronous cocoa+
 */
#include "LIBSVMDatapoint.h"
#include "delay_cocoa_sparse_nodes.h"
#include "cocoa_computations.h"
#include "mpi.h"
#include<cmath>
#include<algorithm>
#include<cstring>
#include<fstream>
#include<sys/stat.h>
#include<cassert>

/*
int In_loop, Out_loop;
int group_size;
int max_delay;
double C;
double seconds;
int d1;
int N;
double gammas; // for cocoa gammas=1/K for cocoa+ gammas=1/K
*/

int main(int argc, char* argv[]){

	int group_size;
	int max_delay;
	long In_loop;
	int Out_loop;
	double C;
	double wait;
	long d1;
	long N;
	double gammas;
	long sparsity; // # dims send to serer.
	char* root;

	if(argc != 12){
		printf("Input arguments. 1 group_size 2 max_delay 3 In_loop 4 out_loop 5 C(regularization parameter) 6 crash time for worker1 7 dimension of data  8 num training size 9 gammas 10 sparsity 11 datapath\n ");
		return 0;
	}
	else{
		group_size = std::stoi(argv[1],nullptr);
		max_delay = std::stoi(argv[2],nullptr);
		In_loop = std::stol(argv[3],nullptr);
		Out_loop = std::stoi(argv[4],nullptr);
		C = std::stod(argv[5],nullptr);
		wait = std::stod(argv[6],nullptr);
		d1 = std::stol(argv[7],nullptr);
		N = std::stol(argv[8],nullptr);
		gammas = std::stod(argv[9],nullptr);
		sparsity = std::stol(argv[10],nullptr);
		root = argv[11];
	}

	

	int max_epoch = 100; 

	long size = d1;
	std::vector<double> w(size,0);
	std::vector<double> alpha;

	double test_obj, train_acc, test_acc;
	double train_obj, train_dualobj;
	bool evaluation;
	int numtasks, taskid;
    long	iters=0;
	std::vector<double> timer(max_epoch+1,0);
	double starttime=0, endtime=0;
	double train_L2=0;
	std::vector<double>  work_wait_time(2,0);
	
	
	//init mpi
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

	// initialize using svm data.
    std::vector<LIBSVMDatapoint *> datapoints;

	if(taskid!=0){
		//char* root = "/home/ubuntu/";
		//char* root = "/home/jonny/ZHOU/Data/libsvm/url/4/";
		char images_path[200];
		char char_taskid[21];
		sprintf(char_taskid, "%02d", taskid-1);
		std::strcpy(images_path, root);
		std::strcat(images_path,"mat"); 
		std::strcat(images_path, char_taskid);
		printf("read data from %s\n", images_path);

		std::ifstream data_file_input(images_path);
		std::string datapoint_line;
		long datapoint_count = 0;
		while (std::getline(data_file_input, datapoint_line)) {
			datapoints.push_back(new LIBSVMDatapoint(datapoint_line, datapoint_count++));
		}

		alpha.resize(datapoint_count, 0);
		printf("task id %d , num %zu\n", taskid, datapoints.size());
	}

	MPI_Barrier(MPI_COMM_WORLD);
	assert(numtasks-1>=group_size);

	// sigma
	double sigma = (numtasks-1) * gammas;
	double step = 0;

	//int print_freq = N/((numtasks-1)*In_loop);
	if(taskid==0){
		printf("group_size %d, max_delay %d, In_loop %ld, Out_loop %d, C %f,   sleeping time   %f original dim %ld sparse dim %ld \n", group_size, max_delay, In_loop, Out_loop, C, wait, d1, sparsity);
		printf("Epoch     train_obj   train_dual_obj   train_dual_gap   train_acc    test_obj    test_acc     L2norm    Time   work_time wait_time \n");
	}
	for(int epoch=0; epoch<max_epoch; epoch++){
		// evaluation stage do not neeed to evaluate in this experiment.
		MPI_Bcast(&w[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		evaluation = true;
		if(taskid==0){
			iters = 1;
			server_node(w, train_obj, train_dualobj, train_acc, train_L2, work_wait_time, iters, step, evaluation, gammas, C, numtasks-1, max_delay, group_size, N, sparsity);
			printf("%d     %.8f    %.8f    %.4e     %.2f      %.4f     %.2f     %.4e       %.2f      %.4f  %.4f\n", epoch, train_obj, train_dualobj, train_obj-train_dualobj,\
						   	train_acc, test_obj, test_acc, train_L2, timer[epoch], work_wait_time[0], work_wait_time[1]);
		}
		else{
			worker_node(datapoints, w, alpha, sigma, gammas, step, C, work_wait_time, iters, evaluation, taskid, numtasks-1, N, wait, epoch, sparsity);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		// update stage
		if(taskid==0) starttime = MPI_Wtime();
		evaluation = false;
		if(taskid==0){
			//iters = N/(In_loop*group_size);
			iters = Out_loop;
			server_node(w, train_obj, train_dualobj, train_acc, train_L2, work_wait_time, iters, step, evaluation, gammas, C, numtasks-1, max_delay, group_size, N, sparsity);
		}
		else{
			iters = In_loop;
			worker_node(datapoints, w, alpha, sigma, gammas, step, C, work_wait_time, iters, evaluation, taskid, numtasks-1, N, wait, epoch, sparsity);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		if(taskid==0) {
			endtime = MPI_Wtime();
			timer[epoch+1] = timer[epoch] + endtime - starttime;
		}
	}

MPI_Finalize();

return 0;
}
