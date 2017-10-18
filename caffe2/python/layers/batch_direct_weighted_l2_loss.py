## @package batch_direct_weighted_l2_loss
# Module caffe2.python.layers.batch_direct_weighted_l2_loss
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import (
    ModelLayer,
)
from caffe2.python.layers.tags import (
    Tags
)
import numpy as np


class BatchDirectWeightedL2Loss(ModelLayer):

    def __init__(self, model, input_record,
        max_scale = 1.0,
        name='batch_direct_weighted_l2_loss', 
        **kwargs):
        super(BatchDirectWeightedL2Loss, self).__init__(
            model, name, input_record, **kwargs)

        assert schema.is_schema_subset(
            schema.Struct(
                ('label', schema.Scalar()),
                ('prediction', schema.Scalar())
            ),
            input_record
        )
        self.tags.update([Tags.EXCLUDE_FROM_PREDICTION])
        self.max_scale = max_scale
        self.output_schema = schema.Struct(
            ('loss', schema.Scalar(
                np.float32,
                self.get_next_blob_reference('loss')
                )
            ),
            ('l2_metric', schema.Scalar(
                np.float32,
                self.get_next_blob_reference('l2_metric')
                )
            ),
            ('scaled_l2_metric', schema.Scalar(
                np.float32,
                self.get_next_blob_reference('scaled_l2_metric')
                )
            ),            
        )

    def add_ops(self, net):
        prediction = self.input_record.prediction()

        label = self.input_record.label.field_blobs()
        if self.input_record.label.field_type().base != (
                self.input_record.prediction.field_type().base):

            label = net.Cast(
                label,
                net.NextScopedBlob('cast_label'),
                to=schema.data_type_for_dtype(
                    self.input_record.prediction.field_type()
                )
            )

        label = net.StopGradient(
            label,
            net.NextScopedBlob('stopped_label')
        )

        l2dist = net.SquaredL2Distance(
            [label, prediction],
            net.NextScopedBlob('l2_dist')
        )

        net.AveragedLoss(l2dist, self.output_schema.l2_metric())

        scaler, scaler_no_clip = net.ScaleWithClip(
            [label], 
            [net.NextScopedBlob('scaler'), net.NextScopedBlob('scaler_no_clip')],
            max_scale = self.max_scale,
        )

        scaler = net.StopGradient(
            scaler,
            net.NextScopedBlob('stopped_scaler')
        )

        scaler_no_clip = net.StopGradient(
            scaler_no_clip,
            net.NextScopedBlob('stopped_scaler_no_clip')
        )

        scaler = net.Squeeze(
            scaler,
            net.NextScopedBlob('squeezed_scaler'),
            dims=[1]
        )

        scaler_no_clip = net.Squeeze(
            scaler_no_clip,
            net.NextScopedBlob('squeezed_scaler_no_clip'),
            dims=[1]
        )

        scaled_loss = net.Mul(
            [l2dist, scaler],
            net.NextScopedBlob('scaled_loss')
        )

        scaled_loss_no_clip = net.Mul(
            [l2dist, scaler_no_clip],
            net.NextScopedBlob('scaled_loss_no_clip')
        )

        net.AveragedLoss(
            scaled_loss, self.output_schema.loss()
        )
        net.AveragedLoss(
            scaled_loss_no_clip, self.output_schema.scaled_l2_metric()
        )
