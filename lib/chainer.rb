require "weakref"

require "chainer/version"

require 'chainer/configuration'
require 'chainer/function'
require 'chainer/initializer'
require 'chainer/initializers/init'
require 'chainer/initializers/constant'
require 'chainer/initializers/normal'
require 'chainer/link'
require 'chainer/links/connection/linear'
require 'chainer/variable'
require 'chainer/variable_node'
require 'chainer/utils/argument'
require 'chainer/utils/initializer'
require 'chainer/utils/variable'
require 'chainer/utils/array'
require 'chainer/functions/activation/relu'
require 'chainer/functions/math/basic_math'
require 'chainer/functions/connection/linear'
require 'chainer/parameter'

require 'numo/narray'

module Chainer
  def self.configure
    yield(configuration)
  end

  def self.configuration
    @configuration ||= Configuration.new
  end
end

