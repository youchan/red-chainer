class SkipGram < Chainer::Chain
  def initialize(n_vocab, n_units, loss_func)
    super()
    init_scope do
      @embed = Chainer::Links::Connection::EmbedID.new(nil, n_vocab, n_units, initial_weight: Chainer::Iinitializers::Uniform.new(1. / n_units), loss_func: loss_func)
      @loss_func = loss_func
    end
  end

  def call(x, contexts):
    e = @embed.(contexts)
    shape = e.shape
    x = Chainer::Functions::Array::BroadcastTo.broadcast_to(x[:, None], (shape.batch_size, shape.n_context))
    e = Chainer::Functions::Array::Reshape.reshape(e, (shape.batch_size * shape.n_context, shape.n_units))
    x = Chainer::Functions::Array::Reshape.reshape(x, (shape.batch_size * shape.n_context,))
    loss = @loss_func(e, x)
    reporter.report({'loss': loss}, self)
    loss
  end
end
