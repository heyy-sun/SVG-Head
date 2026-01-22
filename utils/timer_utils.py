import torch

class CUDA_Timer(object):
    def __init__(self, label, logger=None, valid=True, warmup_steps=10):
        self.valid = valid
        if not valid:
            return
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
        self.label = label
        self.logger = logger
        self.counter = 0
        self.val = 0.0
        self.warmup_steps = warmup_steps

    def start(self, step):
        if self.valid and step > self.warmup_steps:
            self.starter.record()

    def end(self, step):
        if self.valid and step > self.warmup_steps:
            self.ender.record()
            self._update_val()

    def _update_val(self):
        torch.cuda.synchronize()
        time = self.starter.elapsed_time(self.ender)
        self.val = self.val * self.counter + time
        self.counter += 1
        self.val /= self.counter

        if self.logger:
            self.logger.info("[{}] ".format(self.label) + "{val " + str(time) + "ms} {avg " + str(self.val) + "ms}")
        else:
            print("[{}] ".format(self.label) + "{val " + str(time) + "ms} {avg " + str(self.val) + "ms}")

        # reset timer
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)

    def __str__(self):
        if self.valid:
            fmtstr = "[{}] " + "{avg " + str(self.val) + "ms}"
        else:
            fmtstr = "[{}] " + "\{avg -1ms\}"
        return fmtstr.format(self.label)

    def __enter__(self):
        if self.valid:
            self.starter.record()

    def __exit__(self, exc_type, exc_value, tb):
        if self.valid:
            self.ender.record()
            torch.cuda.synchronize()
            if self.logger:
                self.logger.info(self.label + " : {}ms".format(self.starter.elapsed_time(self.ender)))
            else:
                print(self.label + " : {}ms".format(self.starter.elapsed_time(self.ender)))