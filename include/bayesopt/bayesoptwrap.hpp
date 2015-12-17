#ifndef _BAYESOPTWRAP_HPP_
#define _BAYESOPTWRAP_HPP_


#include "bayesopt.h"
#include "bayesopt/bayesopt.hpp"

#include "gauss_distribution.hpp"

using namespace bayesopt;

  template <typename Distr>
  class ContinuousModelWrap: public bayesopt::ContinuousModel
  {
   public:
    virtual ~ContinuousModelWrap() {
      if(cache) {
        delete cache;
      }
    }

  ContinuousModelWrap(size_t dim, bopt_params params)
    : ContinuousModel(dim,params)
    , cache(NULL)
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
    if(cache) {
      delete cache;
    }
    cache = new Distr(p);
    return cache;
  }

  protected:
    void* mOtherData;
    eval_func mF;
    Distr *cache;
  };



  class GaussianDistributionWrap {
  public:
    GaussianDistributionWrap(ProbabilityDistribution *p) {
      d = dynamic_cast<GaussianDistribution*>(p);
      if(!d) {
        throw std::runtime_error("Can't cast to gaussian distribtion");
      }
    }

    double getMean() {
      if(!d) throw std::runtime_error("Distribution is not defined");
      return d->getMean();
    }

    double getStd() {
      if(!d) throw std::runtime_error("Distribution is not defined");
      return d->getStd();
    }
    double pdf(double v) {
      if(!d) throw std::runtime_error("Distribution is not defined");
      return d->pdf(v);
    }

  private:
    GaussianDistribution *d;
  };

  typedef ContinuousModelWrap<GaussianDistributionWrap> ContinuousModelGaussWrap;




#endif