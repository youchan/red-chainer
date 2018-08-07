class CBoW < Chainer::Chain
  def initialize(n_vocab, n_units, loss_func)
    super()
    init_scope do
      @embed = Chainer::Links::Connection::EmbedID.new(nil, n_vocab, n_units, initial_weight: I.Uniform(1. / n_units), loss_func: loss_func)
      @loss_func = loss_func
    end
  end

  def call(x, contexts):
    e = self.embed(contexts)
    h = F.sum(e, axis=1) * (1. / contexts.shape[1])
    loss = self.loss_func(h, x)
    reporter.report({'loss': loss}, self)
    return loss
  end
end
