from moviepy.editor import VideoFileClip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dirpath")
parser.add_argument("--start")
parser.add_argument("--end")
parser.add_argument("--fragment")

pars = parser.parse_args()

video = VideoFileClip(pars.dirpath)
fragm = video.subclip(pars.start, pars.end)

fragm.write_videofile(pars.fragment, codec='libx264')

video.close()
fragm.close()

