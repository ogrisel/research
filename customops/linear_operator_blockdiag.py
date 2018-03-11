# Copyright 2018 Olivier Grisel. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Composes one or more `LinearOperators`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.util.tf_export import tf_export


__all__ = ["LinearOperatorBlockDiag"]


@tf_export("customops.LinearOperatorBlockDiag")
class LinearOperatorBlockDiag(linear_operator.LinearOperator):
    def __init__(self,
                 block_operators,
                 is_non_singular=None,
                 is_self_adjoint=None,
                 is_positive_definite=None,
                 is_square=None,
                 name=None):
        # Validate operators.
        check_ops.assert_proper_iterable(block_operators)
        block_operators = list(block_operators)
        if not block_operators:
            raise ValueError(
                "Expected a non-empty list of block operators. Found: %s"
                % block_operators)
        self._block_operators = block_operators

        # Validate dtype.
        dtype = block_operators[0].dtype
        for operator in block_operators:
            if operator.dtype != dtype:
                name_type = (str((o.name, o.dtype)) for o in block_operators)
                raise TypeError(
                    "Expected all operators to have the same dtype. Found %s"
                    % "   ".join(name_type))

        # Auto-set and check hints.
        if all(operator.is_non_singular for operator in block_operators):
            if is_non_singular is False:
                raise ValueError(
                    "The assembly of non-singular block operators is always"
                    " non-singular.")
            is_non_singular = True

        # TODO: Check other hints.

        # Initialization.
        graph_parents = []
        for operator in block_operators:
            graph_parents.extend(operator.graph_parents)

        if name is None:
            name = "LinearOperatorBlockDiag_of_"
            name += "_".join(operator.name for operator in block_operators)

        with ops.name_scope(name, values=graph_parents):
            super(LinearOperatorBlockDiag, self).__init__(
                dtype=dtype,
                graph_parents=graph_parents,
                is_non_singular=is_non_singular,
                is_self_adjoint=is_self_adjoint,
                is_positive_definite=is_positive_definite,
                is_square=is_square,
                name=name)

    @property
    def operators(self):
        return self._block_operators

    def _shape(self):
        matrix_shape = tensor_shape.TensorShape(
            [tf.Dimension(sum(o.range_dimension.value
                              for o in self.operators)),
             tf.Dimension(sum(o.domain_dimension.value
                              for o in self.operators))])

        # Get broadcast batch shape.
        # broadcast_shape checks for compatibility.
        batch_shape = self.operators[0].batch_shape
        for operator in self.operators[1:]:
            batch_shape = common_shapes.broadcast_shape(
                batch_shape, operator.batch_shape)

        return batch_shape.concatenate(matrix_shape)

    def _shape_tensor(self):
        if self.shape.is_fully_defined():
            return ops.convert_to_tensor(
                self.shape.as_list(), dtype=dtypes.int32, name="shape")

        matrix_shape = array_ops.stack([
            tf.reduce_sum(o.range_dimension_tensor()
                          for o in self.operators[0]),
            tf.reduce_sum(o.domain_dimension_tensor()
                          for o in self.operators[0]),
        ])

        # Dummy Tensor of zeros.  Will never be materialized.
        zeros = array_ops.zeros(shape=self.operators[0].batch_shape_tensor())
        for operator in self.operators[1:]:
            zeros += array_ops.zeros(shape=operator.batch_shape_tensor())
        batch_shape = array_ops.shape(zeros)

        return array_ops.concat((batch_shape, matrix_shape), 0)

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        # If self.operators = [A, B], and not adjoint, then
        # matmul_order_list = [B, A].
        # As a result, we return A.matmul(B.matmul(x))
        if adjoint:
            matmul_order_list = self.operators
        else:
            matmul_order_list = list(reversed(self.operators))

        result = matmul_order_list[0].matmul(
            x, adjoint=adjoint, adjoint_arg=adjoint_arg)
        for operator in matmul_order_list[1:]:
            result = operator.matmul(result, adjoint=adjoint)
        return result

    def _determinant(self):
        result = self.operators[0].determinant()
        for operator in self.operators[1:]:
            result *= operator.determinant()
        return result

    def _log_abs_determinant(self):
        result = self.operators[0].log_abs_determinant()
        for operator in self.operators[1:]:
            result += operator.log_abs_determinant()
        return result

    def _solve(self, rhs, adjoint=False, adjoint_arg=False):
        # TODO(langmore) Implement solve using solve_ls if some intermediate
        # operator maps to a high dimensional space.
        # In that case, an exact solve may still be possible.

        # If self.operators = [A, B], and not adjoint, then
        # solve_order_list = [A, B].
        # As a result, we return B.solve(A.solve(x))
        if adjoint:
            solve_order_list = list(reversed(self.operators))
        else:
            solve_order_list = self.operators

        solution = solve_order_list[0].solve(
            rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
        for operator in solve_order_list[1:]:
            solution = operator.solve(solution, adjoint=adjoint)
        return solution

    def _add_to_tensor(self, x):
        return self.to_dense() + x


if __name__ == '__main__':
    import tensorflow as tf

    with tf.Session() as session:
        a = tf.linalg.LinearOperatorIdentity(
            num_rows=3, dtype=tf.float32)
        tril = [[1., 2.], [3., 4.]]
        b = tf.linalg.LinearOperatorLowerTriangular(tril)

        bd = LinearOperatorBlockDiag([a, b])
        assert bd.shape.as_list() == [5, 5]
        square_bd = tf.matmul(bd, bd)
        square_bd_dense = square_bd.to_dense()
        print(session.run(square_bd_dense))
