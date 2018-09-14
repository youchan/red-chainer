class RNNForLM < Chainer::Chain
  def initialize(n_vocab, n_units)
    super()
    init_scope do
      @embed = Chainer::Links::Connection::EmbedID.new(n_vocab, n_units)
      @l1 = Chainer::Links::Connection::LSTM.new(n_units, out_size: n_units)
      @l2 = Chainer::Links::Connection::LSTM.new(n_units, out_size: n_units)
      @l3 = Chainer::Links::Connection::Linear.new(n_units, out_size: n_vocab)
    end

    params do |param|
      param.data[*([:*] * param.data.shape.size)] = param.data.class.new(param.data.shape).rand(-0.1, 0.1)
    end
  end

  def reset_state
    @l1.reset_state()
    @l2.reset_state()
  end

  def call(x)
    h0 = @embed.(x)
    h1 = @l1.(Chainer::Functions::Noise::Dropout.dropout(h0))
    h2 = @l2.(Chainer::Functions::Noise::Dropout.dropout(h1))
    @l3.(Chainer::Functions::Noise::Dropout.dropout(h2))
  end
end

# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.
class ParallelSequentialIterator < Chainer::Dataset::Iterator
  attr_reader :is_new_epoch, :epoch

  def initialize(dataset, batch_size, repeat: true)
    @dataset = dataset
    @batch_size = batch_size  # batch size
    # Number of completed sweeps over the dataset. In this case, it is
    # incremented if every word is visited at least once after the last
    # increment.
    @epoch = 0
    # True if the epoch is incremented at the last iteration.
    @is_new_epoch = false
    @repeat = repeat
    # Offsets maintain the position of each sequence in the mini-batch.
    @offsets = batch_size.times.map {|i| i * length / batch_size }
    # NOTE: this is not a count of parameter updates. It is just a count of
    # calls of ``__next__``.
    @iteration = 0
  end

  # This iterator returns a list representing a mini-batch. Each item
  # indicates a different position in the original sequence. Each item is
  # represented by a pair of two word IDs. The first word is at the
  # "current" position, while the second word at the next position.
  # At each iteration, the iteration count is incremented, which pushes
  # forward the "current" position.
  def next
    # If not self.repeat, this iterator stops at the end of the first
    # epoch (i.e., when all words are visited once).
    if !@repeat && @iteration * @batch_size >= length
      raise StopIteration.new
    end
    cur_words = get_words
    @iteration += 1
    next_words = get_words

    epoch = @iteration * @batch_size / length
    @is_new_epoch = @epoch < epoch
    @epoch = epoch if @is_new_epoch

    cur_words.zip(next_words)
  end

  def length
    @dataset.size
  end

  def epoch_detail
    # Floating point version of epoch.
    @iteration * @batch_size / length
  end

  def get_words
    # It returns a list of current words.
    @offsets.map {|offset| @dataset[(offset + @iteration) % length]}
  end

  def serialize(serializer)
    # It is important to serialize the state to be recovered on resume.
    @iteration = serializer.('iteration', @iteration)
    @epoch = serializer.('epoch', @epoch)
  end
end

# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater < Chainer::Training::StandardUpdater

  def initialize(train_iter, optimizer, bprop_len, device = -1)
    super(train_iter, optimizer, device: device)
    @bprop_len = bprop_len
  end

  # The core part of the update routine can be customized by overriding.
  def update_core
    loss = 0
    # When we pass one iterator and optimizer to StandardUpdater.__init__,
    # they are automatically named 'main'.
    train_iter = get_iterator(:main)
    optimizer = get_optimizer(:main)

    # Progress the dataset iterator for bprop_len words at each iteration.
    @bprop_len.times do |i|
      # Get the next batch (a list of tuples of two word IDs)
      batch = train_iter.next()

      # Concatenate the word IDs to matrices and send them to the device
      # self.converter does this job
      # (it is chainer.dataset.concat_examples by default)
      x, t = @converter.(batch, device: @device)

      # Compute the loss at this time step and accumulate it
      loss += optimizer.target.(Chainer::Variable.new(x), Chainer::Variable.new(t))
    end

    optimizer.target.cleargrads()  # Clear the parameter gradients
    loss.backward()  # Backprop
    loss.unchain_backward()  # Truncate the graph

    optimizer.update()  # Update the parameters
  end
end

class BagOfWords
  attr_reader :counter, :vocabularies, :ids

  def initialize(name, enum)
    @ids = {}
    @counter = {}
    @vocabularies = {}
    add(name, enum)
  end

  def add(name, enum)
    ids = []
    @ids[name] = ids
    enum.each do |v|
      @counter[v.word] = @counter[v.word].yield_self{|c| c ? c + 1 : 0 }
      unless @vocabularies.key?(v.word)
        @vocabularies[v.word] = @vocabularies.size
      end
      ids << @vocabularies[v.word]
    end
  end

  def [](id)
    @vocabularies.keys[id]
  end

  def id(word)
    @vocabularies[word]
  end
end
