"""
Monitor training progress.
"""

from typing import NamedTuple
import numpy as np
from torch.multiprocessing import Lock, Queue, Value
from bokeh.plotting import figure
from bokeh.io import output_file, save
from bokeh.layouts import row


class EpisodeData(NamedTuple):
    """
    Information for one episode of training.
    """
    length: int
    score: int
    average_value: float


class Monitor(object):
    """
    A class for monitoring training progress.
    """

    def __init__(self, log_directory):
        self.lock = Lock()

        self.episode_lengths = []
        self.episode_scores = []
        self.episode_values = []

        self.frame_counter = None

        self.queue = Queue(100)

        self.log_directory = log_directory

    def record(self, episode_data: EpisodeData):
        """
        Record an episode and print some information about it.
        """
        with self.lock:
            self.episode_lengths.append(episode_data.length)
            self.episode_scores.append(episode_data.score)
            self.episode_values.append(episode_data.average_value)

            print(f"Episode {len(self.episode_scores)} - "
                  f"Frame {self.frame_counter.value} - "
                  f"Length: {episode_data.length} - "
                  f"Score: {episode_data.score}")

            if len(self.episode_scores) % 100 == 0:
                score_plot = figure(title="Score")
                score_plot.line(range(len(self.episode_scores)),
                                self.episode_scores)
                smoothed_scores = smooth(self.episode_scores, 100)
                score_plot.line(range(len(smoothed_scores)), smoothed_scores,
                                line_color="green", line_width=5)

                value_plot = figure(title="Value")
                value_plot.line(range(len(self.episode_values)),
                                self.episode_values)

                layout = row(score_plot, value_plot)

                output_file(f'{self.log_directory}/graphs.html')
                save(layout)

    def monitor(self, frame_counter: Value, max_timesteps: int):
        """
        Loop over the queue and monitor the results.
        """
        print("Monitoring...")
        self.frame_counter = frame_counter
        while self.frame_counter.value < max_timesteps:
            data = self.queue.get()
            self.record(data)


def smooth(x: np.array, smoothing_factor: int) -> np.array:
    """
    Use a convolution filter to smooth a 1-dimensional input.
    """
    box = np.ones(smoothing_factor) / smoothing_factor
    smoothed = np.convolve(x, box, mode='valid')
    return smoothed
