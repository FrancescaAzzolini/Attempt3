//The first four lines are always the same

#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  //We recall the Data from the R file as a matrix
  DATA_MATRIX(Z);
  int n = Z.rows();
  int p = Z.cols();
  
  // PARAMETER_VECTOR is for vectors, PARAMETER is for scalars
  
  PARAMETER_VECTOR(mu1);
  //PARAMETER_VECTOR(mu2);
  PARAMETER_VECTOR(C);
  PARAMETER_VECTOR(log_sd1);
  PARAMETER_VECTOR(log_sd2);
  PARAMETER(transformed_rho1);
  //PARAMETER(transformed_rho2);
  PARAMETER(a);
  
  //We could fix a
  //Type a=0.3;
  //We define a new vector mu2 (right now, are parameters are mu1 and C)
  vector<Type> mu2(2); //the 2 in parenthesis means this is a 2-dim vector
  //We define each component of the new vector mu2
  // in C++ we start counting from 0
  mu2(0)=C(0);
  mu2(1)=C(1)+mu1(1);
  //We need to transform log_sd and rho to satisfy the conditions we know they satisfy 
  vector<Type> sd1 = exp(log_sd1);
  vector<Type> sd2 = exp(log_sd2);
  Type rho1=2.0/ (1.0+ exp(-transformed_rho1)) - 1.0;
  //We only use one rho
  //Type rho2=2.0/ (1.0+ exp(-transformed_rho2)) - 1.0;
  
  // That's how you construct a matrix
  //We actually don't use Sigma in the computations
  //matrix<Type> Sigma(p,p);
  //Sigma.row(0)<< a*a*sd1[0]*sd1[0]+(1-a)*(1-a)*sd2[0]*sd2[0], a*a*sd1[1]*sd1[0]*rho1+(1-a)*(1-a)*sd2[1]*sd2[0]*rho1;
  //Sigma.row(1)<< a*a*sd1[1]*sd1[0]*rho1+(1-a)*(1-a)*sd2[1]*sd2[0]*rho1, a*a*sd1[1]*sd1[1]+(1-a)*(1-a)*sd2[1]*sd2[1];
  
  matrix<Type> Sigma1(p,p);
  Sigma1.row(0)<< sd1[0]*sd1[0], sd1[1]*sd1[0]*rho1;
  Sigma1.row(1)<< sd1[0]*sd1[1]*rho1, sd1[1]*sd1[1];
  
  matrix<Type> Sigma2(p,p);
  Sigma2.row(0)<< sd2[0]*sd2[0], sd2[1]*sd2[0]*rho1;
  Sigma2.row(1)<< sd2[0]*sd2[1]*rho1, sd2[1]*sd2[1];
  
  //We define the two residual vectors of dimension p
  vector<Type> residual1(p);
  vector<Type> residual2(p);
  //vector<Type> mu=a*mu1+(1-a)*mu2;
  
  //We define neglogL and impose a starting value 0
  Type neglogL=0.0;
  
  // using namespace is the "library" of TMB
  using namespace density;
  //Those are commands to construct multivariate gaussians 
  MVNORM_t<Type> neg_log_dmvnorm1(Sigma1);
  MVNORM_t<Type> neg_log_dmvnorm2(Sigma2);
  
  //cycle for every element in the Data Z
  for (int i=0; i<n; i++)
  {
    //We construct the two residual vectors and evaluate the neg-log-likelihood function
    residual1 = vector<Type>(Z.row(i)) - mu1;
    residual2= vector<Type>(Z.row(i)) - mu2;
    neglogL -=log(a*exp(-neg_log_dmvnorm1(residual1))+(1-a)*exp(-neg_log_dmvnorm2(residual2)));
  }
  //These are the elements the command "report" will return in the R file
  REPORT(Sigma1);
  REPORT(Sigma2);
  REPORT(mu1);
  REPORT(mu2);
  REPORT(rho1);
  REPORT(log_sd1);
  REPORT(log_sd2);
  REPORT(a);
  return neglogL;  
}

