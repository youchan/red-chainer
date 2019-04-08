# Example to generate text from a recurrent neural network language model.
#
# This code is ported from following implementation.
# https://github.com/chainer/chainer/example/ptb/gentxt.py

require "optparse"
require "chainer"
require_relative "./model"
require_relative "./dataset"

args =
{
  model: "model.npz",
  primetext: "愛",
  seed: 123,
  unit: 650,
  sample: 1,
  length: 20,
  gpu: -1
}

opts = OptionParser.new
opts.on("-m", "--model VALUE", "model data, saved by train_ptb.rb") { |v| args[:model] = v }
opts.on("-p", "--primetext VALUE", "base text data, used for text generation") { |v| args[:primetext] = v }
opts.on("-s", "--seed VALUE", "random seeds for text generation") { |v| args[:seed] = v.to_i }
opts.on("-u", "--unit VALUE", "number of units") { |v| args[:unit] = v.to_i }
opts.on("--sample VALUE", "negative value indicates NOT use random choice") { |v| args[:sample] = v.to_i }
opts.on("--length VALUE", "length of the generated text") { |v| args[:length] = v.to_i }
opts.on("--gpu VALUE", "GPU ID (negative value indicates CPU)") { |v| args[:gpu] = v.to_i }

opts.parse!(ARGV)

Numo::NArray.srand(args[:seed])
Chainer.configuration.train = false

# load vocabulary
train = UtamapLyrics.new(type: :train)
valid = UtamapLyrics.new(type: :valid)
test = UtamapLyrics.new(type: :test)
bow = BagOfWords.new(:train, train)
bow.add(:valid, valid)
bow.add(:test, test)

# should be same as n_units , described in train_ptb.py
n_units = args[:unit]
n_vocab = bow.vocabularies.size + 1
rnn = RNNForLM.new(n_vocab, n_units)
model = Chainer::Links::Model::Classifier.new(rnn)

Chainer::Serializers::MarshalDeserializer.load_file(args[:model], model)

model.predictor.reset_state()

primetext = args[:primetext]

if bow.vocabularies.key?(primetext)
  prev_word = Chainer::Variable.new(Numo::Int32[bow.id(primetext)])
else
  puts "ERROR: Unfortunately #{primetext} is unknown."
  exit
end

prob = Chainer::Functions::Activation::Softmax.softmax(model.predictor.(prev_word))
print primetext + ' '

args[:length].times do |i|
  prob = Chainer::Functions::Activation::Softmax.softmax(model.predictor.(prev_word))
  if args[:sample] > 0
    probability = Numo::DFloat.cast(prob.data[0, true])
    probability.inplace / probability.sum
    sum = 0.0
    rand = rand()
    index = probability.to_a.find_index{|p| sum += p; rand < sum }
  else
    index = prob.data.max_index
  end

  if bow[index] == '<eos>'
    print "."
  else
    print bow[index] + " "
  end

  prev_word = Chainer::Variable.new(Numo::Int32[index])
end

puts

