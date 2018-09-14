module Chainer::Serializers

    # Serializer for dictionary.

    # This is the standard serializer in Chainer. The hierarchy of objects are
    # simply mapped to a flat dictionary with keys representing the paths to
    # objects in the hierarchy.

    # .. note::
    #    Despite of its name, this serializer DOES NOT serialize the
    #    object into external files. It just build a flat dictionary of arrays
    #    that can be fed into :func:`numpy.savez` and
    #    :func:`numpy.savez_compressed`. If you want to use this serializer
    #    directly, you have to manually send a resulting dictionary to one of
    #    these functions.

    # Args:
    #     target (dict): The dictionary that this serializer saves the objects
    #         to. If target is None, then a new dictionary is created.
    #     path (str): The base path in the hierarchy that this serializer
    #         indicates.

    # Attributes:
    #     target (dict): The target dictionary. Once the serialization completes,
    #         this dictionary can be fed into :func:`numpy.savez` or
    #         :func:`numpy.savez_compressed` to serialize it in the NPZ format.
  class DictionarySerializer < Chainer::Serializer
    def initialize(targett = {}, path = "")
      @target = target
      @path = path
    end

    def [](key)
      key = key.strip('/')
      DictionarySerializer.new(@target, @path + key + '/')
    end

    def call(key, value)
      key = key.sub(/\A\//, "")
      ret = value
      @target[@path + key] = value.to_a
      ret
    end

    
    # Saves an object to the file in NPZ format.
    #
    # This is a short-cut function to save only one object into an NPZ file.
    #
    # Args:
    #     filename (str): Target file name.
    #     obj: Object to be serialized. It must support serialization protocol.
    #     compression (bool): If ``True``, compression in the resulting zip file
    #         is enabled.
    def self.save_npz(filename, obj, compression: true)
      s = DictionarySerializer.new
      s.save(obj)
      open(filename, 'wb') do |f|
        if compression
            numpy.savez_compressed(f, **s.target)
        else
            numpy.savez(f, **s.target)
        end
      end
    end
  end
end
