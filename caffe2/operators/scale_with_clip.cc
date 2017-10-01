#include "caffe2/utils/math.h"
#include "caffe2/operators/scale_with_clip.h"

namespace caffe2 {

template <>
bool ScaleWithClipOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_ENFORCE_GE(max_scale_, 0.0);
  Y->ResizeLike(X);
  const auto* Xdata = X.template data<float>();
  auto* Ydata = Y->template mutable_data<float>();

  ConstEigenVectorArrayMap<float> Xvec(Xdata, X.size());
  EigenVectorArrayMap<float> Yvec(Ydata, Y->size());
  Yvec = Xvec.abs();
  Yvec = Yvec.max(Yvec.maxCoeff())/Yvec;
  float max_ = Yvec.maxCoeff();
  Yvec = (Yvec - 1.0) * (max_scale_ - 1.0)/(max_ - 1.0) + 1.0;
  return true;
};

REGISTER_CPU_OPERATOR(ScaleWithClip, ScaleWithClipOp<float, CPUContext>);

OPERATOR_SCHEMA(ScaleWithClip)
    .NumInputs(1)
    .NumOutputs(1)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc(R"DOC(
Return a tensor with the same shape as the input one, but each value is the
ratio of the maxinum absolute value / its absolute value. The ratio is clip 
by max_scale.
    )DOC")
    .Arg("max_scale", "The max scale")
    .Input(0, "input", "Input tensor")
    .Output(0, "output", "Filled with its max value");

} // namespace caffe2