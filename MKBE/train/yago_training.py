from tensorpack import *


class SingleGPUTrainer(train.SimpleTrainer):
    def __init__(self, hyperparams):
        super(SingleGPUTrainer, self).__init__()
        mutable_params = ["emb_keepprob", "fm_keepprob", "mlp_keepprob", "label_smoothing"]
        self.feed = dict((k + ":0", hyperparams[k]) for k in mutable_params)
        self.feed["is_training:0"] = True

    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        with tfutils.tower.TowerContext("", is_training=True):
            assert input.setup_done()
            get_cost_fn(*input.get_input_tensors())
            self.train_op = get_opt_fn()
        return []

    def run_step(self):
        _ = self.hooked_sess.run(self.train_op, feed_dict=self.feed)
