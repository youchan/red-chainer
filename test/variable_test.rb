# frozen_string_literal: true

require 'chainer'

class Constant < Chainer::Function
  def initialize(outputs)
    @outputs = outputs
  end
  def forward_cpu(inputs)
    return @outputs
  end

  def backward_cpu(inputs, grad_outputs)
    return inputs.map{|_| _.new_zeros()}.to_a
  end
end

def constant(xs, value)
  return Constant.new(value).(*xs)
end

class Chainer::VariableTest < Test::Unit::TestCase
  data = {
    'test1' => {x_shape: [10], c_shape: [2, 5], label: '(2, 5), Numo::SFloat'},
    'test2' => {x_shape: [], c_shape: [1], label: '(1), Numo::SFloat'}}

  def _setup(data)
    @x_shape = data[:x_shape]
    @label = data[:label]
    @c_shape = data[:c_shape]
    @x = Numo::SFloat.new(@x_shape).rand(2) - 1
    @a = Numo::SFloat.new(@x_shape).rand(9.9) + 0.1

    if @x_shape.size != 0
        @size = Numo::NArray.cast(@x_shape).prod().to_i
    else
        @size = 1
    end
    @c = Numo::DFloat.new(@size).seq(0).reshape(*@c_shape).cast_to(Numo::SFloat)
  end

  def check_attributes(gpu)
    x = Chainer::Variable.new(@x)
    assert_equal(@x.shape, x.shape)
    assert_equal(@x.ndim, x.ndim)
    assert_equal(@x.size, x.size)
    assert_equal(@x.class, x.dtype)
    assert(x.requires_grad)
    assert(x.node.requires_grad)
  end

  data(data)
  def test_attributes_cpu(data)
    _setup(data)
    check_attributes(false)
  end

  def check_len(gpu)
    x = Chainer::Variable.new(@x)
    if x.ndim == 0
      assert_equal(1, x.size)
    else
      assert_equal(@x_shape[0], x.size)
    end
  end

  data(data)
  def test_len_cpu(data)
    _setup(data)
    check_len(false)
  end

  def check_label(expected, gpu)
    c = Chainer::Variable.new(@c)
    assert_equal(expected, c.label)
  end

  data(data)
  def test_label_cpu(data)
    _setup(data)
    check_label(@label, false)
  end

  def check_backward(inputs, intermediates, outputs, retain_grad)
    outputs.each do |o|
      o.backward(retain_grad: retain_grad)
    end
    assert(inputs.map{|x| !x.grad.nil?}.all?)
    if retain_grad
      assert(intermediates.map{|x| !x.grad.nil?}.all?)
    else
      assert(intermediates.map{|x| x.grad.nil?}.all?)
    end
    assert(outputs.map{|x| !x.grad.nil?}.any?)
  end

  def create_linear_chain(length, gpu)
    x = Chainer::Variable.new(@x)
    ret = [x]
    length.times{|i|
      ret.push(constant([ret[i]], [@a]))
    }

    ret[-1].grad = ret[-1].data.yield_self{|x| x.class.zeros(x.shape) }
    return ret
  end

  data(data)
  def test_backward_cpu(data)
    _setup(data)
    ret = create_linear_chain(2, false)
    check_backward([ret[0]], [ret[1]], [ret[2]], false)
  end

  data(data)
  def test_backward_cpu_retain_grad(data)
    _setup(data)
    ret = create_linear_chain(2, false)
    check_backward([ret[0]], [ret[1]], [ret[2]], true)
  end


  # data(data)
  # def test_unchain(data)
  #   _setup(data)
  #   ret = create_linear_chain(3, false)
  #   old_rank = ret[1].rank
  #   ret[1].unchain()
  #   assert_nil(ret[1].creator)
  #   assert_equal(ret[1].rank, old_rank)
  #   check_backward([ret[1]], [ret[2]], [ret[3]], false)
  # end


  data(data)
  def test_set_none_to_creator(data)
    _setup(data)
    ret = create_linear_chain(3, false)
    old_rank = ret[1].rank
    ret[1].creator = nil
    assert_nil(ret[1].creator)
    assert_equal(ret[1].rank, old_rank)
    check_backward([ret[1]], [ret[2]], [ret[3]], false)
  end

  data(data)
  def test_set_none_and_original_to_creator(data)
    _setup(data)
    ret = create_linear_chain(2, false)
    old_rank = ret[1].rank
    creator = ret[1].creator
    ret[1].creator = nil
    assert_nil(ret[1].creator)
    assert_equal(ret[1].rank, old_rank)

    ret[1].node.rank = -1
    ret[1].creator = creator
    assert_equal(ret[1].creator, creator)
    assert_equal(ret[1].rank, creator.rank + 1)
    check_backward([ret[0]], [ret[1]], [ret[2]], false)
  end

  data(data)
  def test_set_fresh_creator(data)
    _setup(data)
    v = Chainer::Variable.new
    f = Chainer::Function.new
    v.creator = f
    assert_equal(v.creator, f)
    assert_equal(v.rank, 1)
  end

  # data(data)
  # def test_unchain_backward_cpu(data)
  #   _setup(data)
  #   ret = create_linear_chain(3, false)
  #   ret[1].unchain_backward()
  #   self.check_backward([ret[1]], [ret[2]], [ret[3]], false)
  # end

  def test_grad_type_check_pass()
    a = Chainer::Variable.new(Numo::SFloat.new([3]))
    a.grad = Numo::SFloat.new([3])
  end

  def test_grad_type_check_type()
    a = Chainer::Variable.new(Numo::SFloat.new([]))
    #assert_raise(TypeError) { ## No Error
      a.grad = Numo::SFloat.new()
    #}
  end

  def test_grad_type_check_dtype()
    a = Chainer::Variable.new(Numo::SFloat.new([3]))
    assert_raise(TypeError) {
      a.grad = Numo::DFloat.new([3])
    }
  end

  def test_grad_type_check_shape()
    a = Chainer::Variable.new(Numo::SFloat.new([3]))
    assert_raise(TypeError) {
      a.grad = Numo::SFloat.new([2])
    }
  end
end
