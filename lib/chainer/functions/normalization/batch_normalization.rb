module Chainer
  module Functions
    module Normalization
      class BatchNormalizationFunction < Chainer::Function
        attr_reader :running_mean, :running_var
        # Batch normalization function with fixed statistics.
        # This is a variant of batch normalization, where the mean and variance
        # statistics are given by the caller as fixed variables. This is
        # used on testing mode of the batch normalization layer, where batch
        # statistics cannot be used for prediction consistency.
        #
        # @param [Chainer::Variable] x Input variable.
        # @param [Chainer::Variable] gamma Scaling parameter of normalized data.
        # @param [Chainer::Variable] beta Shifting parameter of scaled normalized data.
        # @param [Chainer::Variable] mean Shifting parameter of input.
        # @param [Chainer::Variable] var Square of scaling parameter of input.
        # @param [float] eps Epsilon value for numerical stability.
        def self.fixed_batch_normalization(x, gamma, beta, mean, var, eps: 2e-5)
          old_train = Chainer.configuration.train
          Chainer.configuration.train = false
          norm = self.new(eps: eps, mean: nil, var: nil, decay: 0.0).(x, gamma, beta, mean, var)
          Chainer.configuration.train = old_train
          norm
        end
      
        def initialize(eps: 2e-5, mean: nil, var: nil, decay: 0.9) 
          @running_mean = mean
          @running_var = var
          @eps = eps
          @mean_cache = nil
          @decay = decay
        end

        def forward(inputs)
          x, gamma, beta = inputs[0], inputs[1], inputs[2]
          if Chainer.configuration.train
            if @running_mean.nil?
              @running_mean = Numo::NArray[*gamma].new_zeros
              @running_var = Numo::NArray[*gamma].new_zeros
            else
              @running_mean = Numo::NArray[*@running_mean]
              @running_var = Numo::NArray[*@running_var]
            end
          elsif inputs.size == 5
            @fixed_mean = inputs[3]
            @fixed_var = inputs[4]
          end

          head_ndim = gamma.ndim + 1
          gamma_expander = [1] + gamma.shape + [1] * (x.ndim - head_ndim)
          gamma = gamma.reshape(*gamma_expander)
          beta_expander = [1] + beta.shape + [1] * (x.ndim - head_ndim)
          beta = beta.reshape(*beta_expander)
        
          if Chainer.configuration.train
            axis = [0] + (head_ndim...(x.ndim)).to_a
            mean = x.mean(axis: axis)
            # FIXME: numpy.var
            var = x.var(axis: axis)
            var.inplace + @eps
          else
            mean = @fixed_mean
            var = @fixed_var + @eps
          end

          @std = Numo::NMath.sqrt(var)

          mean_expander = [1] + mean.shape + [1] * (x.ndim - head_ndim)
          x_mu = x - mean.reshape(*mean_expander)
          std_expander = [1] + @std.shape + [1] * (x.ndim - head_ndim)
          x_mu /= @std.reshape(*std_expander)
          @x_hat = x_mu
          y = gamma * @x_hat
          y.inplace + beta

          if Chainer.configuration.train
            m = x.size.div(gamma.size)
            adjust = m / [m - 1.0, 1.0].max
            @running_mean.inplace * @decay
            temp_ar = Numo::NArray[*mean]
            temp_ar.inplace * (1 - @decay)
            @running_mean += temp_ar
            
            @running_var *= @decay
            temp_ar = Numo::NArray[*var]
            temp_ar.inplace * ((1 - @decay) * adjust)
            @running_var.inplace + temp_ar
          end

          [y,]
        end

        def backward(inputs, grad_outputs)
          x, gamma = inputs[0], inputs[1]
          gy = grad_outputs[0]
          head_ndim = gamma.ndim + 1
          m = gamma.class[x.size.div(gamma.size)][0]
          axis = [0] + (head_ndim...(x.ndim)).to_a

          if inputs.size == 5
            mean = inputs[3]
            var = inputs[4]
            std = Numo::NMath.sqrt(var)
            gs = gamma / std
            gbeta = gy.sum(axis: axis)

            mean_expander = [1] + mean.shape + [1] * (x.ndim - head_ndim)
            x_mu = x - mean.reshape(*mean_expander)
            std_expander = [1] + std.shape + [1] * (x.ndim - head_ndim)
            x_mu /= std.reshape(*std_expander)
            x_hat = x_mu
            ggamma = (gy * x_hat).sum(axis: axis)
            gmean = -gs * gbeta
            gvar = -0.5 * gamma / var * ggamma
            gs_expander = [1] + gs.shape + [1] * (x.ndim - head_ndim)
            gx = gs.reshape(*gs_expander)
            return [gx, ggamma, gbeta, gmean, gvar]
          end

          gbeta = gy.sum(axis: axis)
          ggamma = (gy * @x_hat).sum(axis: axis)
          tmp = (gamma / @std)
          tmp_expander = [1] + tmp.shape + [1] * (x.ndim - head_ndim)
          tmp = tmp.reshape(*tmp_expander)

          ggamma_expander = [1] + ggamma.shape + [1] * (x.ndim - head_ndim)
          gbeta_expander = [1] + gbeta.shape + [1] * (x.ndim - head_ndim)
          
          gx = tmp * (gy - (@x_hat * ggamma.reshape(*ggamma_expander) + gbeta.reshape(*gbeta_expander)) / m )

          [gx, ggamma, gbeta]
        end
      end
    end
  end
end
