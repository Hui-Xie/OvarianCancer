import matplotlib.pyplot as plt
import numpy as np

f = plt.figure(frameon=False)
DPI = f.dpi
W = 512
H = 496
subplotRow = 1
subplotCol = 2
f.set_size_inches(W*subplotCol / float(DPI), H*subplotRow / float(DPI))

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,hspace=0)  # very important for erasing unnecessary margins.

image0 = np.random.rand(H,W)
image1 = (1.0-image0)**2

subplot1 = plt.subplot(subplotRow, subplotCol, 1)
subplot1.imshow(image0, cmap='gray', )
subplot1.axis('off')

subplot2 = plt.subplot(subplotRow, subplotCol, 2)
subplot2.imshow(image1, cmap='gray', )
subplot2.plot(range(0, W), np.arange(0,W), 'tab:red', linewidth=2)
subplot2.legend("skewLine", loc='lower center')
subplot2.axis('off')

print("Output a random image with 2 subplots at the current directory with name: testPlt.png")
plt.savefig("./testPlt.png", dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()
