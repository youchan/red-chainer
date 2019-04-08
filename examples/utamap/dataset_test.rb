require_relative "./dataset"

dataset = UtamapLyrics.new(type: :test)

puts dataset.map(&:word).join(" ")
