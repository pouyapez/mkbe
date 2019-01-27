import sys, tqdm
from tensorpack import *
from tensorpack.callbacks.inference_runner import _inference_context


class TestRunner(callbacks.InferenceRunner):
    feed = {
        "InferenceTower/emb_keepprob:0": 1.0,
        "InferenceTower/fm_keepprob:0": 1.0,
        "InferenceTower/mlp_keepprob:0": 1.0,
        "InferenceTower/enc_keepprob:0": 1.0,
        "InferenceTower/is_training:0": False,
        "InferenceTower/label_smoothing:0": 0.0
    }

    def _trigger(self):
        for inf in self.infs:
            inf.before_epoch()

        self._input_source.reset_state()

        # iterate over the data, and run the hooked session
        with _inference_context(), \
                tqdm.tqdm(total=self._size, **utils.utils.get_tqdm_kwargs()) as pbar:
            num_itr = self._size if self._size > 0 else sys.maxsize
            for _ in range(num_itr):
                self._hooked_sess.run(fetches=[], feed_dict=self.feed)
                pbar.update()
        for inf in self.infs:
            inf.trigger_epoch()