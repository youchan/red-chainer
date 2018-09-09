module Chainer
  module Links
    module Model
      class Classifier < Chain
        attr_accessor :compute_accuracy
        attr_reader :predictor

        def initialize(predictor, lossfun=Functions::Loss::SoftmaxCrossEntropy.method(:softmax_cross_entropy), accfun=Functions::Evaluation::Accuracy.method(:accuracy))
          super()
          @lossfun = lossfun
          @accfun = accfun
          @y = nil
          @loss = nil
          @accuracy = nil
          @compute_accuracy = true

          init_scope do
            @predictor = predictor
          end
        end

        def call(*args)
          t = args.pop
          x = args
          @y = nil
          @accuracy = nil
          @y = @predictor.(*x)

          @loss = @lossfun.call(@y, t)
          Chainer::Reporter.save_report({loss: @loss}, self)
          if @compute_accuracy
            @accuracy = @accfun.call(@y, t)
            Chainer::Reporter.save_report({accuracy: @accuracy}, self)
          end
          @loss
        end
      end
    end
  end
end
