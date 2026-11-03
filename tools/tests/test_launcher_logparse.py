"""Console parsing tests, using output shaped like the toolbox's interleaved tqdm bars."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from launcher import logparse

UP = "\x1b[A"


def worker(name, step, n, total, position):
    return f"\n" * position + f"\r{name} | {step}:  {100 * n // total}%|##   | {n}/{total} [00:10<00:20, 40.00frame/s]" \
        + UP * position


def overall(n, total):
    return f"\r {100 * n // total}%|##   | {n}/{total} [10:00<05:00, 90.00s/it]"


class LogParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = logparse.LogParser("train_and_test", 10)
        self.parser.feed("Preprocessing dataset...\n")

    def test_overall_bar_drives_progress_while_workers_are_listed(self):
        self.parser.feed(overall(44, 63) + worker("rec-a", "decode (OpenCV)", 13769, 17100, 1)
                         + worker("rec-b", "decode (OpenCV)", 1446, 17100, 2)
                         + worker("rec-a", "decode (OpenCV)", 13773, 17100, 1))
        state = self.parser.snapshot()
        self.assertEqual(state["phase"], "preprocessing")
        self.assertEqual((state["bar"]["desc"], state["bar"]["n"], state["bar"]["total"]), ("", 44, 63))
        self.assertEqual([(w["name"], w["step"], w["n"], w["position"]) for w in state["workers"]],
                         [("rec-a", "decode (OpenCV)", 13773, 1), ("rec-b", "decode (OpenCV)", 1446, 2)])

    def test_same_file_name_on_two_workers_stays_apart(self):
        self.parser.feed(worker("cam2_Camera_Front.mp4", "OpenCV", 10, 100, 1)
                         + worker("cam2_Camera_Front.mp4", "OpenCV", 80, 100, 2))
        self.assertEqual([w["n"] for w in self.parser.snapshot()["workers"]], [10, 80])

    def test_message_after_a_bar_is_parsed_and_completes_the_video(self):
        self.parser.feed(worker("rec-a", "transform/save", 5, 10, 1) + worker("rec-b", "decode (OpenCV)", 1, 10, 2)
                         + "[rec-a] Complete: 10 clips in 60.0s.\n")
        state = self.parser.snapshot()
        self.assertEqual(state["last_line"], "[rec-a] Complete: 10 clips in 60.0s.")
        self.assertEqual([w["name"] for w in state["workers"]], ["rec-b"])

    def test_stale_worker_bars_are_dropped(self):
        self.parser.feed(worker("rec-a", "decode (OpenCV)", 1, 10, 1))
        self.parser.feed("".join(worker("rec-b", "decode (OpenCV)", i % 10, 10, 2)
                                 for i in range(logparse.WORKER_STALE + 1)))
        self.assertEqual([w["name"] for w in self.parser.snapshot()["workers"]], ["rec-b"])

    def test_training_bar_is_unchanged_and_workers_are_hidden(self):
        self.parser.feed(worker("rec-a", "decode (OpenCV)", 1, 10, 1) + "\n")
        self.parser.feed("====Training Epoch: 0====\n")
        self.parser.feed("\rTrain epoch 0:  50%|#####     | 5/10 [00:05<00:05, 1.00it/s, loss=0.5]\n")
        state = self.parser.snapshot()
        self.assertEqual(state["phase"], "training")
        self.assertEqual(state["bar"]["desc"], "Train epoch 0")
        self.assertEqual(state["train_loss"], {"0": 0.5})
        self.assertEqual(state["workers"], [])


if __name__ == "__main__":
    unittest.main()
