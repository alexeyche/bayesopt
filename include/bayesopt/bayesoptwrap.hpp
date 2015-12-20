#ifndef _BAYESOPTWRAP_HPP_
#define _BAYESOPTWRAP_HPP_


#include "bayesopt.h"
#include "bayesopt/bayesopt.hpp"

#include "gauss_distribution.hpp"
#include "student_t_distribution.hpp"
#include "nonparametricprocess.hpp"

using namespace bayesopt;

  template <typename Distr>
  class ContinuousModelWrap: public bayesopt::ContinuousModel
  {
   public:
    virtual ~ContinuousModelWrap() {
    }

  ContinuousModelWrap(size_t dim, bopt_params params)
    : ContinuousModel(dim,params)
  {
  }


  double evaluateSample( const vectord &Xi )
  {
    return mF(mDims,&Xi[0],NULL,mOtherData);
  };


  void set_eval_funct(eval_func f)
  {  mF = f; }


  void save_other_data(void* other_data)
  {  mOtherData = other_data; }


  void setBoundingBox(const double* lb, const double* ub) {
    vectord lbV(mDims);
    vectord ubV(mDims);

    for(size_t di=0; di<mDims; ++di) {
      lbV[di] = lb[di];
      ubV[di] = ub[di];
    }
    ContinuousModel::setBoundingBox(lbV, ubV);
  }


  const size_t& getDimSize() const {
    return mDims;
  }


  void optimize(double* res) {
    vectord resV(mDims);
    ContinuousModel::optimize(resV);
    for(size_t di=0; di<mDims; ++di) {
      res[di] = resV[di];
    }
  }


  void initWithPoints(const double *x, const double *y, size_t nsamples) {
    matrixd xV(nsamples, mDims);
    vectord yV(nsamples);

    for(size_t i=0; i<nsamples; ++i) {
      yV[i] = y[i];
      for(size_t j=0; j<mDims; ++j) {
        xV(i, j) = x[i*mDims + j];
      }
    }

    ContinuousModel::initWithPoints(xV, yV);
  }

  Distr* getPrediction(const double* x) {
    vectord query(mDims);
    for(size_t xi=0; xi<mDims; ++xi) {
      query[xi] = x[xi];
    }
    ProbabilityDistribution* p = ContinuousModel::getPrediction(query);
    Distr *d = dynamic_cast<Distr*>(p);
    if(!d) {
      throw std::runtime_error("Failed to cast distribution");
    }
    return new Distr(*d);
  }

  protected:
    void* mOtherData;
    eval_func mF;
  };

  typedef GaussianDistribution GaussianDistributionWrap;
  typedef ContinuousModelWrap<GaussianDistributionWrap> ContinuousModelGaussWrap;
  
  typedef StudentTDistribution StudentTDistributionWrap;
  typedef ContinuousModelWrap<StudentTDistributionWrap> ContinuousModelStudentTWrap;
  
  


#endif