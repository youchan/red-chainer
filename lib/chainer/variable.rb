module Chainer
  class Variable
    attr_accessor :requires_grad, :node

    def initialize(data=nil, name: nil, grad: nil, requires_grad: true)
      unless data.nil? || data.is_a?(Numo::NArray)
        raise TypeError, "Numo::NArray are expected."
      end

      @data = data
      @grad = grad
      @requires_grad = requires_grad
      @node = VariableNode.new(variable: self, name: name, grad: grad)
    end

    def inspect
      {data: @data.inspect, grad: @grad.inspect, requires_grad: @requires_grad.inspect}.inspect
    end

    def data
      return @data
    end

    def data=(d)
      @data = d
      @node.set_data_type(d)
    end

    def name
      return @node.name
    end

    def name=(n)
      @node.name = n
    end

    def label
      @node.label
    end

    def creator
      @node.creator
    end

    def creator=(func)
      @node.creator = func
    end

    def grad
      @node.grad
    end

    def grad=(g)
      @node.set_grad_with_check(g, nil, self)
    end

    def shape
      self.data.shape
    end

    def ndim
      self.data.ndim
    end

    def size
      self.data.size
    end

    def dtype
      self.data.class
    end

    def rank
      @node.rank
    end

    def reshape(*shape)
      self.data.reshape(*shape)
    end

    def cleargrad
      @node.grad = nil
    end

    def backward(retain_grad: false)
      return if self.creator.nil?

      if self.data.size == 1 && self.grad.nil?
        self.grad = self.data.new_ones
      end

      funcs = [self.creator]

      while func = funcs.pop
        outputs = func.outputs.map(&:__getobj__)
        in_data = func.inputs.map(&:data)
        out_grad = outputs.map { |y| y.nil? ? nil : y.grad }

        func.output_data = outputs.map { |y| y.nil? ? nil : y.data }
        gxs = func.backward(in_data, out_grad)

        raise "Unmatched matries size: gxs.size(#{gxs.size}) != in_data.size(#{in_data.size})" unless gxs.size == in_data.size

        unless func.retain_after_backward
          func.output_data = nil
        end

        unless retain_grad
          outputs.each do |y|
            if y && y != @node
              y.grad = nil
            end
          end
        end

        seen_vars = []
        need_copy = []

        func.inputs.zip(gxs).each do |x, gx|
          next if gx.nil?
          next unless x.requires_grad

          Utils::Variable.check_grad_type(func, x, gx)

          id_x = x.object_id
          if x.creator.nil? # leaf
            if x.grad.nil?
              x.grad = gx
              need_copy << id_x
            else
              if need_copy.include?(id_x)
                x.grad = Utils::Array.force_array(x.grad + gx)
                need_copy.delete(id_x)
              else
                x.grad += gx
              end
            end
          else # not leaf
            funcs << x.creator
            if seen_vars.include?(id_x)
              if need_copy.include?(id_x)
                x.grad = Utils::Array.force_array(gx + x.grad)
                need_copy.delete(id_x)
              else
                x.grad += gx
              end
            else
              x.grad = gx
              seen_vars << id_x
              need_copy << id_x
            end
          end
        end
      end 
    end

    # Deletes the reference to the creator of this variable.
    #
    # This method deletes the reference to the creator from the corresponding
    # variable node. Unlike :meth:`unchain_backward`, it does not backtrack
    # the graph.
    #
    # This method is equivalent to ``self.creator = None``.
    def unchain
        self.creator = nil
    end

    # Deletes references between variable nodes and functions backward.
    #
    # After this method completes, intermediate variable nodes and functions
    # that are not referenced from anywhere are deallocated by reference
    # count GC. Also this variable itself deletes the reference to its
    # creator function from the node, i.e. the node becomes root in the
    # computation graph. It indicates that backprop after unchaining stops at
    # this variable. This behavior is useful to implement truncated BPTT.
    def unchain_backward
      cand_funcs = []
      seen_set = Set.new()

      add_cand = Proc.new do |cand|
        if cand && !seen_set.include?(cand)
          cand_funcs.append(cand)
          seen_set.add(cand)
        end
      end

      add_cand.(self.creator)

      while cand_funcs.size > 0
        func = cand_funcs.pop
        func.inputs.each do |var|
          add_cand.(var.creator)
        end
        func.unchain()
      end
    end

    def -@
      Functions::Math::Neg.new.(self) 
    end

    def +(other)
      if other.instance_of?(Chainer::Variable)
        Functions::Math::Add.new.(*[self, other])
      else
        Functions::Math::AddConstant.new(other).(self)
      end
    end

    def -(other) 
      if other.instance_of?(Chainer::Variable)
        Functions::Math::Sub.new.(*[self, other])
      else
        Functions::Math::AddConstant.new(-other).(self)
      end
    end

    def *(other)
      if other.instance_of?(Chainer::Variable)
        Functions::Math::Mul.new.(*[self, other])
      else
        Functions::Math::MulConstant.new(other).(self)
      end
    end

    def /(other)
      if other.instance_of?(Chainer::Variable)
        Functions::Math::Div.new.(*[self, other])
      else
        Functions::Math::MulConstant.new(1 / other).(self)
      end
    end

    def **(other) 
      if other.instance_of?(Chainer::Variable)
        Functions::Math::PowVarVar.new.(*[self, other])
      else
        Functions::Math::PowVarConst.new(other).(self)
      end
    end

    def retain_data
      @node.data = @data
    end

    # when left side is Numeric value and right side is Chainer::Value, call this method.
    def coerce(other)
      other = self.data.class[*other] if other.kind_of?(Numeric)
      [Chainer::Variable.new(other, requires_grad: false), self]
    end
  end
end

