import os
from PIL import Image
from moviepy.editor import VideoFileClip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dirpath")
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)
parser.add_argument("--fragments")
parser.add_argument("--step", type=int, default=10)

pars = parser.parse_args()

video = VideoFileClip(pars.dirpath)

num = 0

for i in range(pars.start, pars.end, pars.step):
    frame = video.get_frame(i)
    frame_file = os.path.join(pars.fragments, f"{num}.jpg")

    img = Image.fromarray(frame)
    img = img.resize((250, int(250 * img.height / img.width)))
    img.save(frame_file)

    num += 1

video.close()


