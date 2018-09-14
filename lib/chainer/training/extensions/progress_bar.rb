require 'erb'

module Chainer
  module Training
    module Extensions
      class ProgressBar < Extension
        def initialize(training_length: nil, update_interval: 100,  bar_length: 50, out: STDOUT)
          @training_length = training_length
          @status_template = nil
          @update_interval = update_interval
          @bar_length = bar_length
          @out = out
          @out.sync = true
          @recent_timing = []
        end

        def call(trainer)
          if @training_length.nil?
            t = trainer.stop_trigger
            raise TypeError, "cannot retrieve the training length #{t.class}" unless t.is_a?(Chainer::Training::Triggers::IntervalTrigger)
            @training_length = [t.period, t.unit]
          end

          if @status_template.nil?
            @status_template = ERB.new("<%= sprintf('%10d', self.iteration) %> iter, <%= self.epoch %> epoch / #{@training_length[0]} #{@training_length[1]}s\n")
          end

          length, unit = @training_length
          iteration = trainer.updater.iteration

          # print the progress bar according to interval
          return unless iteration % @update_interval == 0

          epoch = trainer.updater.epoch_detail
          now = Time.now.to_f

          @recent_timing << [iteration, epoch, now]
          @out.write("\033[J")

          if unit == 'iteration'
            rate = iteration.to_f / length
          else
            rate = epoch.to_f / length
          end

          bar_total = ('#' * (rate * @bar_length).to_i).ljust(@bar_length, '.')
          @out.write("     total #{bar_total} %6.2f%\n" % (rate * 100))

          epoch_rate = epoch - epoch.to_i
          bar_epoch = ('#' * (epoch_rate * @bar_length).to_i).ljust(@bar_length, '.')
          @out.write("this epoch #{bar_epoch} %6.2f%\n" % (epoch_rate * 100))

          status = @status_template.result(trainer.updater.bind)
          @out.write(status)

          old_t, old_e, old_sec = @recent_timing[0]
          span = now - old_sec

          speed_t = iteration - old_t == 0 ? 0.0 : (iteration - old_t) / span

          if unit == 'iteration'
            estimated_time = (length - iteration) / speed_t
          else
            speed_e = (epoch - old_e) / span
            estimated_time = (length - epoch) / speed_e
          end

          estimated_time = 0.0 if estimated_time.nan? || estimated_time.infinite?

          @out.write("%10.5g iters/sec. Estimated time to finish: #{Time.at(estimated_time).getutc.strftime("%H:%M:%S")}.\n" % speed_t)

          # move the cursor to the head of the progress bar
          @out.write("\033[4A") # TODO: Support Windows
          @out.flush

          @recent_timing.delete_at(0) if @recent_timing.size > 100
        end

        def finalize
          @out.write("\033[J") # TODO: Support Windows
          @out.flush
        end
      end
    end
  end
end
