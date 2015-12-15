#ifndef _BAYESOPTWPR_HPP_
#define _BAYESOPTWPR_HPP_


#include "bayesopt.h"
#include "bayesopt/bayesopt.hpp"


  class ContinuousModelWrap: public bayesopt::ContinuousModel 
  {
   public:

    ContinuousModelWrap(size_t dim, bopt_params params);

    virtual ~ContinuousModelWrap(){};

    double evaluateSample( const vectord &Xi );

    void set_eval_funct(eval_func f);


    void save_other_data(void* other_data);
   
   	void setBoundingBox(const double* lb, const double* ub);

   	const size_t& getDimSize() const;

   	void optimize(double* res);
  protected:
    void* mOtherData;
    eval_func mF;
  };



#endif