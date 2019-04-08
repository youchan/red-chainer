require "numo/linalg"
require "chainer"
require "optionparser"
require_relative "./model"

# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
compute_perplexity = Proc.new do |result|
  result['perplexity'] = np.exp(result['main/loss'])
  if result.key?('validation/main/loss')
    result['val_perplexity'] = Numo::NMath.exp(result['validation/main/loss'])
  end
end


args = {
  unitsize: 650,
  batchsize: 20,
  bproplen: 35,
  epoch: 39,
  gradclip: 5,
  resume: nil,
  model: "model.npz",
  negativesize: 5,
  out: 'result',
  test: false,
}

opt = OptionParser.new
opt.on("-u", "--unitsize VALUE", "Number of units (default: #{args[:unitsize]})") { |v| args[:unitsize] = v.to_i }
opt.on("-b", "--batchsize VALUE", "Number of words in each mini-batch (default: #{args[:batchsize]})") { |v| args[:batchsize] = v.to_i }
opt.on("-l", "--bproplene VALUE", "Number of words in each mini-batch(= length of truncated BPTT)  (default: #{args[:bproplen]})") { |v| args[:bproplen] = v.to_i }
opt.on("-e", "--epoch VALUE", "Number of sweeps over the dataset to train (default: #{args[:epoch]})") { |v| args[:epoch] = v.to_i }
opt.on("-c", "--gradclip VALUE", "Gradient norm threshold to clip (default: #{args[:gradclip]})") { |v| args[:gradclip] = v.to_i }
opt.on("-r", "--resume VALUE", "Resume the training from snapshot") { |v| args[:resume] = v }
opt.on("-m", "--model VALUE", "The model to use: skipgram or cbow (default: #{args[:model]})") { |v| args[:model] = v }
opt.on("--negativesize VALUE", "Number of negative sample (default: #{args[:negativesize]})") { |v| args[:negativesize] = v.to_f }
opt.on("-o", "--out VALUE", "Directory to output the result (default: #{args[:out]})") { |v| args[:out] = v }
opt.on("--test", "Use tiny datasets for quick tests") { args[:test] = true }
opt.parse!(ARGV)


# Load the Penn Tree Bank long word sequence dataset

train = Datasets::PennTreebank.new(type: :train)
valid = Datasets::PennTreebank.new(type: :valid)
test = Datasets::PennTreebank.new(type: :test)

bow = BagOfWords.new(:train, train)
bow.add(:valid, valid)
bow.add(:test, test)
n_vocab =  bow.vocabularies.size + 1
puts "#vocab = #{n_vocab}"

if args[:test]
  train_ids = bow.ids[:train].take(100)
  valid_ids = bow.ids[:valid].take(100)
  test_ids = bow.ids[:valid].take(100)
else
  train_ids = bow.ids[:train]
  valid_ids = bow.ids[:valid]
  test_ids = bow.ids[:test]
end

train_iter = ParallelSequentialIterator.new(train_ids, args[:batchsize])
val_iter = ParallelSequentialIterator.new(valid_ids, 1, repeat: false)
test_iter = ParallelSequentialIterator.new(test_ids, 1, repeat: false)

# Prepare an RNNLM model
rnn = RNNForLM.new(n_vocab, args[:unitsize])
model = Chainer::Links::Model::Classifier.new(rnn)
model.compute_accuracy = false  # we only want the perplexity

# Set up an optimizer
optimizer = Chainer::Optimizers::SGD.new(lr: 1.0)
optimizer.setup(model)
optimizer.add_hook(Chainer::GradientClipping.new(args[:gradclip]))

# Set up a trainer
updater = BPTTUpdater.new(train_iter, optimizer, args[:bproplen])
trainer = Chainer::Training::Trainer.new(updater, stop_trigger: [args[:epoch], 'epoch'], out: args[:out])

eval_model = model.copy  # Model with shared params and distinct states
eval_rnn = eval_model.predictor
trainer.extend(Chainer::Training::Extensions::Evaluator.new(
    val_iter,
    eval_model,
    device: -1,
    # Reset the RNN state at the beginning of each evaluation
    eval_hook: -> (_) { eval_rnn.reset_state }
))

interval = args[:test] ? 10 : 500
trainer.extend(Chainer::Training::Extensions::LogReport.new(postprocess: compute_perplexity, trigger: [interval, 'iteration']))
trainer.extend(
  Chainer::Training::Extensions::PrintReport.new(['epoch', 'iteration', 'perplexity', 'val_perplexity']),
  trigger: [interval, 'iteration']
)
trainer.extend(Chainer::Training::Extensions::ProgressBar.new(update_interval: args[:test] ? 1 : 10))
trainer.extend(Chainer::Training::Extensions::Snapshot.snapshot)
trainer.extend(Chainer::Training::Extensions::Snapshot.snapshot_object(target: model) {|it| "model_iter_#{it}" })
if args[:resume]
  Chainer::Serializers::MarshalDeserializer.load_file(args[:resume], trainer)
end

trainer.run()

# Evaluate the final model
print('test')
eval_rnn.reset_state()
evaluator = Chainer::Training::Extensions::Evaluator.new(test_iter, eval_model, device: -1)
result = evaluator.()

puts "test perplexity: #{Math.exp(result['/main/loss'])}"

# Serialize the final model
Chainer::Serializers::MarshalSerializer.save_file(args[:model], model)
