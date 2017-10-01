#ifndef CAFFE2_OPERATORS_SCALE_WITH_CLIP
#define CAFFE2_OPERATORS_SCALE_WITH_CLIP

#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class ScaleWithClipOp final : public Operator<Context> {
	public:
	 ScaleWithClipOp(const OperatorDef& operator_def, Workspace* ws)
	 	: Operator<Context>(operator_def, ws), 
	 	  OP_SINGLE_ARG(float, "max_scale", max_scale_, 1e4) {
	 	CAFFE_ENFORCE(max_scale_ >= 0.0, "max_scale must be >= 0");
	 }
	 USE_OPERATOR_CONTEXT_FUNCTIONS;

	 bool RunOnDevice() override;

	private:
	 T max_scale_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SCALE_WITH_CLIP