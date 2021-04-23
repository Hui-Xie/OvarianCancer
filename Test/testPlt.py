import matplotlib.pyplot as plt
import numpy as np

f = plt.figure(frameon=False)
DPI = f.dpi
W = 512
H = 496
f.set_size_inches(W / float(DPI), H / float(DPI))

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,hspace=0)  # very important for erasing unnecessary margins.

image = np.random.rand(H,W)

plt.imshow(image, cmap='gray', )
plt.axis('off')

print("Output a random image at the current directory with name: testPlt.png")
plt.savefig("./testPlt.png", dpi='figure', bbox_inches='tight', pad_inches=0)