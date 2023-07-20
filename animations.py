import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse

fig, ax = plt.subplots()
def update(frame):
    ax.cla()
    # Load the image for the current frame
    image_path = f"./notebooks/figures/{args.gan_name}/{args.dataset}/images/gen_image_e-{frame+1:03d}.png"
    image = plt.imread(image_path)

    # Update the plot with the new image
    ax.imshow(image)

    # Additional customizations if needed
    ax.set_title(f"Epoch {frame+1}")
    ax.axis("off")


parser = argparse.ArgumentParser()
parser.add_argument("gan_name")
parser.add_argument("dataset")
args = parser.parse_args()

cwd = Path.cwd()
num_frames = len(list(cwd.glob(f"**/notebooks/figures/{args.gan_name}/{args.dataset}/images/*.png")))
interval = 1

anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval)


filename = f"animation_{num_frames}_samples.gif"

folder = cwd / Path('reports/animations') / Path(f'{args.gan_name}/{args.dataset}')
folder.mkdir(parents=True, exist_ok=True)

# fps = 1000 / (num_frames * interval)
fps = 10

writer = animation.FFMpegWriter(fps=fps)
print(folder.joinpath(filename))
anim.save(folder.joinpath(filename), writer=writer)
# anim.save("bhai.gif", writer=writer)

# plt.show()
