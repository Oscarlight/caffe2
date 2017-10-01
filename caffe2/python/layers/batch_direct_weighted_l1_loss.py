## @package batch_direct_weighted_l1_loss
# Module caffe2.python.layers.batch_direct_weighted_l1_loss
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


class BatchDirectWeightedL1Loss(ModelLayer):

    def __init__(self, model, input_record,
        max_scale = 1.0,
        name='batch_direct_weighted_l1_loss', 
        **kwargs):
        super(BatchDirectWeightedL1Loss, self).__init__(
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
        self.output_schema = schema.Scalar(
            np.float32,
            self.get_next_blob_reference('output'))

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

        l1dist = net.L1Distance(
            [label, prediction],
            net.NextScopedBlob('l1')
        )

        scaler = net.ScaleWithClip(
            [label], 
            net.NextScopedBlob('scaler'), 
            max_scale = self.max_scale,
        )

        scaler = net.StopGradient(
            scaler,
            net.NextScopedBlob('stopped_scaler')
        )

        scaler = net.Squeeze(
            scaler,
            net.NextScopedBlob('squeezed_scaler'),
            dims=[1]
        )

        scaled_loss = net.Mul(
            [l1dist, scaler],
            net.NextScopedBlob('scaled_loss')
        )

        net.AveragedLoss(scaled_loss, self.output_schema.field_blobs())
