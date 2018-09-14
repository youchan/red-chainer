module Chainer::Functions::Activation
  # Softmax activation function.
  class Softmax < Chainer::Function
    def initialize(axis: 1)
      @axis = axis
    end

    def forward(x)
      y = x[0] - x[0].max(axis: @axis, keepdims: true)
      Numo::NMath.exp(y.inplace)
      y.inplace / y.sum(axis: @axis, keepdims: true)

      @x_shape = x[0].shape
      retain_inputs([])
      retain_outputs([0])
      [y]
    end

    def backward(x, gy)
      y = @output_data[0]
      gx = y * gy[0]
      sumdx = gx.sum(axis: @axis, keepdims: true)
      gx.inplace - y * sumdx

      [gx]
    end


    # Softmax function.
    #
    # This function computes its softmax along an axis. Let
    # :math:`x = (x_1, x_2, \\dots, x_D)^{\\top}` be the D dimensional index
    # array and :math:`f(x)` be the D dimensional input array. For each index
    # :math:`x` of the input array :math:`f(x)`, it computes the probability
    # :math:`p(x)` defined as
    # :math:`p(x) = {\\exp(f(x)) \\over \\sum_{d} \\exp(f(x_d))}`.
    #
    # Args:
    #     x (~chainer.Variable): Input variable.
    #
    # Returns:
    #     ~chainer.Variable: Output variable.
    def self.softmax(x, axis: 1)
      Softmax.new(axis: axis).(x)
    end
  end
end
