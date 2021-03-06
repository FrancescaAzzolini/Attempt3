rm(list=ls())
setwd("O:/Documents/Experiment/Attempt3")

#The values mu1=c(2,1), sd1=c(2,1), C=c(7, exp(2)), sd2=c(2,2) and a=0.3/a=0.5 work!!
#The values mu1=c(2,1), sd1=c(2,1), C=c(4, exp(2)), sd2=c(4,4) and a=0.3/a=0.5 work!!
#The values mu1=c(1,1), sd1=c(1,1), C=c(8, exp(3)), sd2=c(4,4) and a=0.3 don't work 
#The values mu1=c(1,1), sd1=c(1,1), C=c(7, exp(2)), sd2=c(04,4) and a=0.3 work!!! 

#It appears that increasing exp(l) makes the code blow, but it's not necessary.
#For example, C=(20, exp(2)) doesn't work 

#Note that something similar happened in 1d as well; increasing too much 
#the values of mu1 and mu2 the program would return NaN

#It doesn't seem that the fact that sd_1 != sd_2 matters; also, the components of
#sd_1 and sd_2 can be different. If htey get too big, though, we have the same problem





library(TMB)
library(MASS)
#library(plot3D)
p=2
#n=100000 gives a much better result than n=10000
n=100000

a=0.3

#We want to give some data to the c++ file; in order to do so, we construct a gaussian mixture and we get some
#random values, which will be our Data. Note that in this way we have the right "mu1", "mu2" etc, so we can
#compare the values to the estimations.

#first 2-dim gaussian
mu1=c(2,1)
sd1=c(2,1)
rho1=0.5



Sigma1=matrix(c(sd1[1]*sd1[1], sd1[2]*sd1[1]*rho1,
                sd1[2]*sd1[1]*rho1, sd1[2]*sd1[2]),
              nrow=p, ncol=p, byrow=T)
#Sigma1

#second 2-dim gaussian
#with mu2 "greater" than mu1 (it's enough to have that one component is greater than the other). It's a way
#to ensure that the two means aren't changed in the estimation
C=c(7, exp(2))
mu2=c(C[1],mu1[2]+C[2])

#We could define mu2 without saying it's "greater" than mu1
#mu2=c(2,5)
sd2=c(2,2)
#We actually assume rho1=rho2, so we don't actually need it
rho2=0.5


Sigma2=matrix(c(sd2[1]*sd2[1], sd2[2]*sd2[1]*rho2,
                sd2[2]*sd2[1]*rho2, sd2[2]*sd2[2]),
              nrow=p, ncol=p, byrow=T)

#Sigma2

set.seed(47)


n1=rbinom(1, size=n, prob=a)
n2=n-n1
#simulate the datas of the two 2-dim gaussians, with the weights n1 and n2
X=MASS::mvrnorm(n=n1, mu=mu1, Sigma1) 
Y=MASS::mvrnorm(n=n2, mu=mu2, Sigma2)

#Combine the gaussians
Z=rbind(X,Y)
#Mix the random values
Z=Z[sample(n,n),]

#I actually don't use Sigma later on 
#We're making strong assumptions about the correlations as rho1=rho2 and that Y and X are independent
Sigma=matrix(c(a^2*sd1[1]*sd1[1]+(1-a)^2*sd2[1]*sd2[1], a^2*sd1[2]*sd1[1]*rho1+(1-a)^2*sd2[2]*sd2[1]*rho1,
               a^2*sd1[2]*sd1[1]*rho1+(1-a)^2*sd2[2]*sd2[1]*rho1, a^2*sd1[2]*sd1[2]+(1-a)^2*sd2[2]*sd2[2]),
             nrow=p, ncol=p, byrow=T)
#Sigma


#That was an attempt to write the mixture as in the 1-dim case, but it doesn't work, probably because of Sigmas being matrices
#Just ignore

#components = sample(1:2,prob=c(a,(1-a)),size=n,replace=TRUE)
#mus = c(mu1,mu2)
#Sigmas = c(Sigma1,Sigma2)
#set.seed(47)
#Z = MASS::mvrnorm(n=n,mu=mus[components], Sigmas[components])


#for the graph, only the upper half
pairs(Z, lower.panel = NULL) 


#mean of the gaussian mixture
mu=a*mu1+(1-a)*mu2

#We define data and parameters for the optimization we will do afterwards 
#Note: we could define them after compiling the c++ file!
data=list(Z=Z)
parameters=list(mu1=rep(0,p), C=rep(1,p), log_sd1=rep(0,p), log_sd2=rep(0,p), transformed_rho1=0, a=0)


#We compile the c++ file and we "fix" it in R
compile("Boh3.cpp")
dyn.load(TMB::dynlib("Boh3"))



model=MakeADFun(data, parameters, DLL="Boh3")

#optimization funciton, to evaluate the maximum. The commands "lower" and "upper" 
#are used to impose some bounds to the evaluation of the parameters
fit=nlminb(model$par, model$fn, model$gr, lower = c(-100,-100,-100,-100,-100,-100,-100,-100,-2,0), upper= c(100,100,100,100,100,100,100,100,2,1))


rep=sdreport(model)

#We return the parameters and compare them with the original values
fit$par
rep
mu1
mu2
model$report()
Sigma1
Sigma2
log(sd1)
log(sd2)
a
#2.0/ (1.0+ exp(-1.101)) - 1.0

