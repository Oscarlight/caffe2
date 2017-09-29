#include "caffe2/operators/max_like.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool MaxLikeOp<float, Context>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  const auto* Xdata = X.template data<float>();
  auto* Ydata = Y->template mutable_data<float>();
  ConstEigenVectorArrayMap<float> Xvec(Xdata, X.size());
  EigenVectorArrayMap<float> Yvec(Ydata, Y->size());
  
}



REGISTER_CPU_OPERATOR(MaxLike, MaxLikeOp<float, CPUContext>);

OPERATOR_SCHEMA(MaxLike)
    .NumInputs(1)
    .NumOutputs(1)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc(R"DOC(
Return a tensor with the same shape as the input one, but filled with
its max value.
    )DOC")
    .Input(0, "input", "Input tensor")
    .Output(0, "output", "Filled with its max value");

} // namespace caffe2