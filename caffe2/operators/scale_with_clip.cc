#include "caffe2/utils/math.h"
#include "caffe2/operators/scale_with_clip.h"

namespace caffe2 {

template <>
bool ScaleWithClipOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  auto* Y_no_clip = Output(1);
  CAFFE_ENFORCE_GE(max_scale_, 0.0);
  Y->ResizeLike(X);
  Y_no_clip->ResizeLike(X);
  const auto* Xdata = X.template data<float>();
  auto* Ydata = Y->template mutable_data<float>();
  auto* Ydata_no_clip = Y_no_clip->template mutable_data<float>();

  ConstEigenVectorArrayMap<float> Xvec(Xdata, X.size());
  EigenVectorArrayMap<float> Yvec(Ydata, Y->size());
  EigenVectorArrayMap<float> Yvec_no_clip(Ydata_no_clip, Y->size());
  Yvec_no_clip = Xvec.abs();
  Yvec_no_clip = Yvec_no_clip.max(Yvec_no_clip.maxCoeff())/Yvec_no_clip;
  
  float max_ = Yvec_no_clip.maxCoeff();
  Yvec = (Yvec_no_clip - 1.0) * (max_scale_ - 1.0)/(max_ - 1.0) + 1.0;

  return true;
};

REGISTER_CPU_OPERATOR(ScaleWithClip, ScaleWithClipOp<float, CPUContext>);

OPERATOR_SCHEMA(ScaleWithClip)
    .NumInputs(1)
    .NumOutputs(2)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc(R"DOC(
Return a tensor with the same shape as the input one, but each value is the
ratio of the maxinum absolute value / its absolute value. The ratio is clip 
by max_scale.
    )DOC")
    .Arg("max_scale", "The max scale")
    .Input(0, "input", "Input tensor")
    .Output(0, "output", "Ratio vector w/ clipping")
    .Output(1, "output", "Ratio vector w/o clipping");


} // namespace caffe2