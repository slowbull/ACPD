/* nodes.h.
 *
 */

#include<vector>
#include "mpi.h"
#include "LIBSVMDatapoint.h"
#include "cocoa_computations.h"

// idea: if iter==iters, let sserver receive all updates from workers!!!


void server_node(std::vector<double> &w, double &sum_obj, double &sum_dualobj, double &sum_acc, double &sum_L2, std::vector<double> &sum_work_wait_time,\
			long iters, double step, bool evaluation, double gamma, double C, int numworkers, int max_delay, int group_size, long sum_num, long sparsity){

	// grad vector used to receive gradient.
	long para_size = w.size();

	MPI_Status status_1, status_2;
	MPI_Request req[numworkers];
	double local_obj = 0, local_dualobj=0;
	double acc = 0;
	int dest;
	sum_acc = 0;
	sum_obj = 0;
	sum_L2 = 0;

	if(evaluation){
		// in sgd case, jsut need to compute objective value and compute accuracy. 

		MPI_Reduce(&local_obj, &sum_obj, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		sum_obj /= sum_num;
		sum_obj += com_reg(w,C);

		MPI_Reduce(&local_dualobj, &sum_dualobj, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		sum_dualobj /= sum_num;
		sum_dualobj -= com_reg(w,C);

		// accuracy   computation
		MPI_Reduce(&acc, &sum_acc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		sum_acc /= sum_num;

		// work wait time.
		std::vector<double> local_work_wait_time(2, 0); 
		std::fill(sum_work_wait_time.begin(), sum_work_wait_time.end(), 0);
		MPI_Reduce(&local_work_wait_time[0], &sum_work_wait_time[0], 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		for(size_t i=0; i<sum_work_wait_time.size(); i++)
			sum_work_wait_time[i] /= numworkers;
	}
	else{
		std::vector<std::vector<double>> w_workers(numworkers, std::vector<double>(para_size, 0));
		long iter = 0;

		while(iter < iters){
			iter++;

			std::vector<int> dest_list(numworkers, 0); //store weather receive gradient from workers. 1 or 0.
			int cur_group_size = 0;

			while((cur_group_size < group_size && iter<iters) || (cur_group_size < numworkers && iter==iters)){
				MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status_1);
				dest = status_1.MPI_SOURCE;
				std::vector<double> grad(sparsity*2, 0); 

				//receive gradient from workers.
				MPI_Recv(&grad[0], sparsity*2, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &status_2);

				// update parameters.
				for(long j=0; j<sparsity; j++){
					w[grad[j*2]] += gamma * grad[j*2+1];
					for(int worker_id=0; worker_id<numworkers; worker_id++){
						w_workers[worker_id][grad[2*j]] += gamma * grad[2*j+1];
					}
				}

				cur_group_size += 1;
				dest_list[dest-1] = 1;
			}	

			// send w to workers.
			int req_cnt = 0;
			for(int i=0; i<numworkers; i++){
				if(dest_list[i]!=0){
					std::vector<double> projected_delta_w = project_grad(w_workers[i], sparsity);
					if (iter < iters)
						projected_delta_w.push_back(1.0);
					else
						projected_delta_w.push_back(0.0);
					MPI_Isend(&projected_delta_w[0], 2*sparsity+1, MPI_DOUBLE, i+1, 1, MPI_COMM_WORLD, &req[req_cnt]);
					req_cnt++;
				}
			}	
			MPI_Waitall(req_cnt, req, MPI_STATUSES_IGNORE);

			// receive all delta_w from workers.
			if(iter==iters){
				std::vector<double> grad(para_size, 0); 
				std::vector<double> sum_grad(para_size, 0); 

				//receive gradient from workers.
				MPI_Reduce(&grad[0], &sum_grad[0], para_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

				// update parameters.
				for(long j=0; j<para_size; j++)
					w[j] += gamma * sum_grad[j];
			}	

		}
	}
}


void worker_node(const std::vector<LIBSVMDatapoint*> &datapoints, std::vector<double> &w, std::vector<double> &alpha, double sigma, double gamma, double step, double C, std::vector<double>& local_work_wait_time,\
				long iters, bool evaluation, int taskid, int numtasks, long sum_num, double wait, int epoch, long sparsity){

	double local_obj, sum_obj=0, local_dualobj=0, sum_dualobj=0;
	MPI_Status status;
	long local_num = datapoints.size();
	double acc, sum_acc=0;
	
	// initialize grad vector for communication
	long para_size = w.size();
	
	if(evaluation){
		local_obj = com_primal_obj(datapoints, w, C, acc);

		local_dualobj = com_dual_obj(alpha, C);

		local_obj *= local_num;
		local_dualobj *= local_num;
		acc *= local_num;

		// sending
		MPI_Reduce(&local_obj, &sum_obj, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&local_dualobj, &sum_dualobj, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&acc, &sum_acc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		std::vector<double> sum_work_wait_time(2,0);
		MPI_Reduce(&local_work_wait_time[0], &sum_work_wait_time[0], 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	else{
		double start_time, mid_time, end_time, work_time=0, wait_time=0; 
		std::vector<double> delta_w(para_size, 0);

		C = 1.0/(C*sum_num);

		int loop = 0;
		while(true){
			std::vector<double> old_alpha(alpha);

			start_time = MPI_Wtime();
			loop++;
			//srand( unsigned ( epoch*1000 + loop) );
			std::random_device rd;
			std::mt19937 gen(epoch*1000+loop);
			std::uniform_int_distribution<long> dis(0, alpha.size()-1);

			for(long i=0; i<iters; i++){
				//long idx = rand() % local_num;
				long idx = dis(gen) % local_num;
				// grad represents delta_w. 
				com_delta_alpha(datapoints[idx], w, delta_w, alpha[idx], sigma, C);
			}
			for(long i=0; i<local_num; i++){
				alpha[i] = old_alpha[i]  + gamma*(alpha[i] - old_alpha[i]);
			}

			//grad is resized to 2*sparsity
			std::vector<double> grad = project_grad(delta_w, sparsity);

			mid_time = MPI_Wtime(); 

			// crash with probability prob_crash
			if (taskid==1){
				double goal = mid_time + wait*(mid_time-start_time);
				while(goal > MPI_Wtime()) {};
			}

			mid_time = MPI_Wtime(); 
			work_time += mid_time - start_time;

			// send grad to server node.
			MPI_Send(&grad[0], 2*sparsity, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			
			// receive updated weights.
			std::vector<double> tmp_w(2*sparsity+1, 0);
			MPI_Recv(&tmp_w[0], 2*sparsity+1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
			
			end_time = MPI_Wtime(); 
			wait_time +=  end_time - mid_time;

			//MPI_Wait(&req, &status);
			int flag = tmp_w[2*sparsity];
			tmp_w.pop_back();
			for (int j = 0; j < sparsity; j++)
				w[tmp_w[2*j]] += tmp_w[2*j+1];

			if(flag == 0){
				std::vector<double> sum_grad(para_size, 0); 

				//receive gradient from workers.
				MPI_Reduce(&delta_w[0], &sum_grad[0], para_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
				break;
			}
		}
		local_work_wait_time[0] = work_time;
		local_work_wait_time[1] = wait_time;
	}	
}
