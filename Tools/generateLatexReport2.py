
import os
import sys
sys.path.append("..")
from FilesUtilities import *

latexHead = r'''

\documentclass[12pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage[T1]{fontenc}  %do not interpret underscore

\geometry{a4paper}

\title{Ovarian Cancer Segmentation Test Report}
\author{Hui Xie}

\begin{document}
\maketitle

'''

latextIntro = r'''

\section{Introduction}

\begin{verbatim}
Network Description:
1  A 3D V model to segment ROI of size 51*171*171 pixels around primary cancer.
2  Total 36 patient data, in which training data 24 patients, validation 6 patients, 
   and test 6 patients; All 36 patients data have 50-80% 3D label.
3  For each ROI, about 10% pixels are primary cancer, other 90% pixels are background;
3  Dice coefficient is 3D dice coefficient against corresponding 3D ground truth;
4  Training data augmentation in the fly: affine in XY plane, translation in Z direction;
5  In the bottle neck of V model, the latent vector has size of 512*2*9*9;
6  Dynamic loss weight according trainin  data;

Test Result Description:
1  First 6 patients are validation data; Second 6 patients are test data;
2  Each patient extracts its 5 slices at positions of 25%, 38%, 50%, 62%, 
   75% of volume height for visual check. 
3  The topleft subimage is original input image, title is formated Raw:ID_s{slice};  

List Dice for all patients:
PatiantID	Dice
05431967	0.70550
05722020	0.73781
05096005	0.69400
03864522	0.61908
05088264	0.56666
05430021	0.87353
04641905	0.60258
04477716	0.33847
05056196	0.69583
05498934	0.87504
05311044	0.54239
04029173	0.15008

The Total average Dice is 61.26%, in which there is patient has extreme low dice. 

Simple Analysis:
1  Dice results in different patients has big difference: 87% vs 15%;
2  Big cancer gets better result;
3  Various Cancer texture give challenge, maybe more training sample will help.   


\end{verbatim}

\section{Test Result Figures}

'''


latexTail = r'''

\end{document}

'''

latexItem = r'''

\begin{figure}
	\centering
	\includegraphics[scale=1.2]{FigName}
	\caption{Caption}
\end{figure}

\clearpage  % support huge number of figures

'''

# you may need to modify the output file directory and name
outputLatex = r'''/home/hxie1/data/OvarianCancerCT/primaryROI/predictionResult/OCTestReport_20190919.tex'''

imagesPath = "/home/hxie1/data/OvarianCancerCT/primaryROI/predictionResult"

with open(outputLatex, "w") as f:
    f.write(latexHead)
    f.write(latextIntro)

    originalCwd = os.getcwd()
    os.chdir(imagesPath)
    imageList = [os.path.abspath(x) for x in os.listdir(imagesPath) if '.png' in x]
    os.chdir(originalCwd)
    imageList.sort()

    for fig in imageList:
        name = getStemName(fig, ".png")
        name = "Patient "+ name.replace("_s", " in slice")
        item = latexItem.replace('FigName', fig).replace('Caption', name)
        f.write(item)

    f.write(latexTail)

print(f'{outputLatex} has been outputed.')
























