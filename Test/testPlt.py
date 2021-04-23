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
subplot1 = plt.subplot(1, 1, 1)
subplot1.imshow(image, cmap='gray')
subplot1.axis('off')

plt.savefig("~/temp/testPlt.png",dpi='figure', bbox_inches='tight', pad_inches=0)