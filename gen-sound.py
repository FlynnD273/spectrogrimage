import cv2
import wave
import math
import struct
import argparse

from numpy import average

parser = argparse.ArgumentParser()
parser.add_argument("input_path", help="Input file path. Should be an image file")
parser.add_argument("output_path", help="Output file path. Should be a wav file")
parser.add_argument(
    "-s", "--stretch", type=float, default=1, help="Scale the image along the X-axis"
)
parser.add_argument(
    "-r",
    "--resolution",
    type=int,
    default=200,
    help="Vertical resolution to resize the image to. Keeps aspect ratio",
)

parser.add_argument(
    "-l",
    "--log",
    action="store_true",
    help="Correct logarithmic scale for spectrograms that don't already",
)

args = parser.parse_args()

sample_width = 2
sample_rate = 44100
wav = wave.open(args.output_path, "w")
wav.setnchannels(1)
wav.setframerate(sample_rate)
wav.setsampwidth(sample_width)


def map_range(
    value: float, old_min: float, old_max: float, new_min: float, new_max: float
) -> float:
    return (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


class Tone:
    def __init__(self, frequency: float, amplitude: float) -> None:
        self.frequency = frequency
        self.amplitude = amplitude

    def sample(self, index: int) -> float:
        return (
            math.sin(index / sample_rate * 2 * math.pi * self.frequency)
            * self.amplitude
        )


freq_min = 200
freq_max = 20000

resolution = args.resolution

img = cv2.imread(args.input_path)
height, width, _ = img.shape
img = cv2.resize(img, (int(resolution * width / height), resolution))
rows, cols, _ = img.shape
duration = 10 * args.stretch * width / height

max_log = math.log(rows + 1)
for col in range(cols):
    print(f"\rProgress: {round(col / cols * 100, 2)}%  ", end="")
    tones: list[Tone] = []
    for row in range(rows):
        value = average(img[row, col]) / 255
        if args.log:
            tones.append(
                Tone(
                    map_range(
                        math.log(row + 1),
                        max_log,
                        0,
                        freq_min,
                        freq_max,
                    ),
                    value,
                )
            )
        else:
            tones.append(
                Tone(
                    map_range(
                        row,
                        rows,
                        0,
                        freq_min,
                        freq_max,
                    ),
                    value,
                )
            )

    for i in range(round(duration * sample_rate / cols)):
        sample = 0
        for tone in tones:
            sample += tone.sample(i)
        sample /= len(tones)
        wav.writeframes(struct.pack("h", int(sample * 32767)))
print()
