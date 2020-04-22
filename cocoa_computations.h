/* these are basic compuations. 
 */
// for different layers nn, we have to edit this file.

#ifndef COMTATIONS_H
#define COMTATIONS_H

#include "LIBSVMDatapoint.h"
#include<cmath>
#include<algorithm>
#include<vector>


void com_gradreg(std::vector<double>& grad, const std::vector<double>& w, double C){
	for(size_t i=0; i<grad.size(); i++)
		grad[i] += C*w[i]; 
}


// compute regularization
double com_reg(const std::vector<double>& w, double C){
	double reg = 0;
	for(size_t i=0; i<w.size(); i++)
		reg += 0.5*C*w[i]*w[i];
	return reg;
}

// compute accuracy 
double metric_acc(const std::vector<double>& o, const std::vector<double>& y){
	long num_sample = o.size();
	long correct = 0;
	for (long i=0; i<num_sample; i++)
		if (o[i]*y[i]>0)
			correct += 1;

	return 1.0 * correct / num_sample;
}

// compute objective value
double com_primal_lsobj(const std::vector<LIBSVMDatapoint*> datapoints, const std::vector<double>& model, double C, double & acc){
	double obj = 0; 
	long correct = 0;

	for (size_t i = 0; i < datapoints.size(); i++) {
		LIBSVMDatapoint *datapoint = datapoints[i];
		double cross_product = 0;
		for (size_t j = 0; j < datapoint->GetCoordinates().size(); j++) {
			int index = datapoint->GetCoordinates()[j];
			double weight = datapoint->GetWeights()[j];
			cross_product += model[index] * weight;
		}

		obj += std::pow((cross_product - datapoint->GetLabel()), 2);

		// correct or wrong
		if (cross_product * datapoint->GetLabel() > 0)
			correct += 1;
	}

	acc = correct * 1.0 / datapoints.size();
	obj = 0.5 * obj / datapoints.size();

	return obj;
}


// compute objective value
double com_primal_l1hingeobj(const std::vector<LIBSVMDatapoint*> &datapoints, const std::vector<double>& model, double C, double & acc){
	double obj = 0; 
	long correct = 0;

	for (size_t i = 0; i < datapoints.size(); i++) {
		LIBSVMDatapoint *datapoint = datapoints[i];
		double cross_product = 0;
		for (size_t j = 0; j < datapoint->GetCoordinates().size(); j++) {
			int index = datapoint->GetCoordinates()[j];
			double weight = datapoint->GetWeights()[j];
			cross_product += model[index] * weight;
		}

		obj += std::max(0., 1- datapoint->GetLabel()*cross_product);

		// correct or wrong
		if (cross_product * datapoint->GetLabel() > 0)
			correct += 1;
	}

	acc = correct * 1.0 / datapoints.size();
	obj = obj / datapoints.size();

	return obj;
}


double com_primal_obj(const std::vector<LIBSVMDatapoint*> &datapoints, const std::vector<double>& model, double C, double & acc){
	double obj = 0; 

	//obj = com_primal_l1hingeobj(datapoints, model, C, acc);
	obj = com_primal_lsobj(datapoints, model, C, acc);

	return obj;
}

// compute l1hinge loss dual obj
double com_dual_l1hingeobj(const std::vector<double> & alpha, double C){
	double obj = 0;
	size_t size = alpha.size();
	for(size_t i=0; i<size; i++){
		obj += alpha[i]/size;	
	}	
	return obj;
}

// compute least square loss dual obj
double com_dual_lsobj(const std::vector<double> & alpha, double C){
	double obj = 0;
	size_t size = alpha.size();
	for(size_t i=0; i<size; i++){
		obj += (alpha[i] - 0.5*alpha[i]*alpha[i])/size;	
	}	
	return obj;
}

double com_dual_obj(const std::vector<double> & alpha, double C){
	return com_dual_lsobj(alpha, C);
	//return com_dual_l1hingeobj(alpha, C);
}

double com_new_alpha_l1hingeloss(LIBSVMDatapoint* datapoint, const std::vector<double>& model, const std::vector<double>& delta_w, double sigma, double cur_alpha, double C){
	// compute projected gradient. 

	double cross_product = 0;
	double xnorm = 0;
	for (size_t j = 0; j < datapoint->GetCoordinates().size(); j++) {
		int index = datapoint->GetCoordinates()[j];
		double weight = datapoint->GetWeights()[j];
		cross_product += model[index] * weight + weight*delta_w[index]*sigma;
		xnorm += weight*weight;
	}

	double grad = (datapoint->GetLabel()*cross_product - 1.0) / C;
	double proj_grad = grad;
	if(cur_alpha <= 0.0)
		proj_grad = std::min(0.0, grad);
	else if(cur_alpha >= 1.0)
		proj_grad = std::max(0.0, grad);
	
	double qii = xnorm * sigma;
	double new_alpha = 1.0;
	if(std::abs(proj_grad) != 0.0){
		if(qii != 0.0)
			new_alpha = std::min(1.0, std::max(0.0, cur_alpha-grad/qii));
	}
	else
		new_alpha = cur_alpha;

	return new_alpha;
}

double com_new_alpha_lsloss(LIBSVMDatapoint* datapoint, const std::vector<double>& model, const std::vector<double>& delta_w, double sigma, double cur_alpha, double C){
	//mat tmp = y(0)*(x*w + sigma*x*delta_w);

	double cross_product = 0;
	double xnorm = 0;
	for (size_t j = 0; j < datapoint->GetCoordinates().size(); j++) {
		long index = datapoint->GetCoordinates()[j];
		double weight = datapoint->GetWeights()[j];
		cross_product += model[index] * weight + weight*delta_w[index]*sigma;
		xnorm += weight*weight;
	}
	cross_product *= datapoint->GetLabel();
	//double qii = xnorm * sigma;
	double qii = xnorm;

	double d_alpha = (1.0-cur_alpha-cross_product)/(1.0+qii*C);
	return (d_alpha + cur_alpha);
}

void com_delta_alpha(LIBSVMDatapoint* datapoint, const std::vector<double> &model, std::vector<double> &delta_w, double& alpha, double sigma, double C){

	 // new alpha
	//double new_alpha = com_new_alpha_l1hingeloss(datapoints[idx], model, delta_w, sigma, alpha[idx], C);
	double new_alpha = com_new_alpha_lsloss(datapoint, model, delta_w, sigma, alpha, C);

	for (size_t j=0; j < datapoint->GetCoordinates().size(); j++){
		long index = datapoint->GetCoordinates()[j];
		double weight = datapoint->GetWeights()[j];
		delta_w[index] += (new_alpha-alpha)*C*datapoint->GetLabel()*weight;
	}
	
	alpha = new_alpha;
}	

double com_duality_residual(LIBSVMDatapoint* datapoint, const std::vector<double> &model, double alpha, std::vector<double> &delta_w, double C, double sigma){
	 // new alpha
	double cross_product = 0;
	for (size_t j=0; j < datapoint->GetCoordinates().size(); j++){
		long index = datapoint->GetCoordinates()[j];
		double weight = datapoint->GetWeights()[j];
		cross_product += (model[index] - sigma*delta_w[index]) * weight;
	}
	double residual = (cross_product - datapoint->GetLabel()) + alpha;
	
	for (size_t j=0; j < datapoint->GetCoordinates().size(); j++){
		long index = datapoint->GetCoordinates()[j];
		double weight = datapoint->GetWeights()[j];
		delta_w[index] += residual*weight*C;
	}
	return residual;
}	

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {
	 // initialize original index locations	
	std::vector<size_t> idx(v.size());
	iota(idx.begin(), idx.end(), 0);
	std::partial_sort(idx.begin(), idx.begin()+1000, idx.end(), [&v](size_t i1, size_t i2){return std::abs(v[i1])>std::abs(v[i2]);});
	return idx;
	}

double norm(const std::vector<double> &vec, const std::vector<long> &idx){
	double ecnorm = 0;
	for(size_t j=0; j<idx.size(); j++){
		long index = idx[j];
		ecnorm += vec[index]*vec[index];
	}	

	return std::sqrt(ecnorm);
}

double norm(const std::vector<double> &vec){
	double ecnorm = 0;
	for(size_t j=0; j<vec.size(); j++){
		ecnorm += vec[j]*vec[j];
	}	

	return std::sqrt(ecnorm);
}


std::vector<double> project_grad(std::vector<double> &delta_w, long sparsity){
	std::vector<double> grad(2*sparsity, 0);
	std::vector<size_t> sorted_idx = sort_indexes(delta_w);

	// compute grad.
	for (long i =0; i < sparsity; i++){
		size_t id = sorted_idx[i];
		grad[2*i] = (double) id;
		grad[2*i+1] = delta_w[id];
		delta_w[id] = 0;
	}
	
	return grad;
}

#endif
