require 'chainer'
require __dir__ + '/models/skip-gram'
require __dir__ + '/models/cbow'
require 'optparse'

args = {
  unitsize: 100,
  windowsize: 100,
  batchsize: 1000,
  negativesize: 5,
  epoch: 300,
  out: 'result',
  out_type: 'hsm',
  resume: nil,
  model: 'skipgram',
}

opt = OptionParser.new
opt.on('-u', '--unitsize VALUE', "Number of units (default: #{args[:unitsize]})") { |v| args[:unitsize] = v.to_i }
opt.on('-w', '--windowsize VALUE', "Window size (default: #{args[:windowsize]})") { |v| args[:windowsize] = v.to_i }
opt.on('-b', '--batchsize VALUE', "Number of words in each mini-batch (default: #{args[:batchsize]})") { |v| args[:batchsize] = v.to_i }
opt.on('-n', '--negativesize VALUE', "Number of negative sample (default: #{args[:negativesize]})") { |v| args[:negativesize] = v.to_f } 
opt.on('-e', '--epoch VALUE', "Number of sweeps over the dataset to train (default: #{args[:epoch]})") { |v| args[:epoch] = v.to_i }
opt.on('-o', '--out VALUE', "Directory to output the result (default: #{args[:out]})") { |v| args[:out] = v }
opt.on('-t', '--out-type VALUE', "Output model type: hsm or ns or original (default: #{args[:out_type]})") { |v| args[:out_type] = v }
opt.on('-r', '--resume VALUE', "Resume the training from snapshot") { |v| args[:resume] = v }
opt.on('-m', '--model VALUE', "The model to use: skipgram or cbow (default: #{args[:model]})") { |v| args[:model] = v }
opt.parse!(ARGV)

if args[:model] == 'skipgram'
  puts 'Using Skip-Gram model'
  model_class = SkipGram
elsif args[:model] == 'cbow'
  puts 'Using Continuous Bag of Words model'
  model_class = CBoW
end

model = Chainer::Links::Model::Classifier.new(model_class.new(n_classes: class_labels))

optimizer = Chainer::Optimizers::MomentumSGD.new(lr: args[:learnrate])
optimizer.setup(model)

train_iter = Chainer::Iterators::SerialIterator.new(train, args[:batchsize])
test_iter = Chainer::Iterators::SerialIterator.new(test, args[:batchsize], repeat: false, shuffle: false)

updater = Chainer::Training::StandardUpdater.new(train_iter, optimizer, device: -1)
trainer = Chainer::Training::Trainer.new(updater, stop_trigger: [args[:epoch], 'epoch'], out: args[:out])

trainer.extend(Chainer::Training::Extensions::Evaluator.new(test_iter, model, device: -1))

trainer.extend(Chainer::Training::Extensions::ExponentialShift.new('lr', 0.5), trigger: [25, 'epoch'])

frequency = args[:frequency] == -1 ? args[:epoch] : [1, args[:frequency]].max
trainer.extend(Chainer::Training::Extensions::Snapshot.new, trigger: [frequency, 'epoch'])

trainer.extend(Chainer::Training::Extensions::LogReport.new)
trainer.extend(Chainer::Training::Extensions::PrintReport.new(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(Chainer::Training::Extensions::ProgressBar.new)

if args[:resume]
  Chainer::Serializers::MarshalDeserializer.load_file(args[:resume], trainer)
end

trainer.run

