class UtamapLyrics < Dataset
  Record = Struct.new(:word)
  DATA_DIR = ""
  VALID_TYPES = [:train, :test, :valid]

  def initialize(type: :train)
    unless VALID_TYPES.include?(type)
      valid_types_label = VALID_TYPES.collect(&:inspect).join(", ")
      message = "Type must be one of [#{valid_types_label}]: #{type.inspect}"
      raise ArgumentError, message
    end
    @type = type

    super()

    @metadata.id = "utamap-lyrics-#{@type}"
    @metadata.name = "Utamap Lyrics: #{@type}"
    @metadata.description = "Collect from Utamap(https://www.utamap.com)"
    @metadata.url = "https://www.utamap.com"
    @metadata.licenses = ["Closed"]

    create_data
  end

  def each(&block)
    return to_enum(__method__) unless block_given?

    parse_data(data_path, &block)
  end

  private

  def create_data
    n_samples = {train: 500, test: 10, valid: 50}
    nm = Natto::MeCab.new(output_format_type: :wakati)
    VALID_TYPES.each do |type|
      base_name = "utamap.#{type}.txt"
      data_path = cache_dir_path + base_name
      unless data_path.exist?
        File.open(data_path, "w") do |out_file|
          Dir.glob(DATA_DIR + *.txt).sample(n_samples[type]).each do |name|
            File.open(name) do |in_file|
              in_file.each_line do |line|
                out_file.puts nm.parse(line)
                end
            end
          end
        end
      end
    end
  end

  def parse_data(data_path)
    File.open(data_path) do |f|
      f.each_line do |line|
        line.split.each do |word|
          yield(Record.new(word.strip))
        end
      end
    end
  end
end
