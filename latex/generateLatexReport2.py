
import os
import sys
sys.path.append("..")
from utilities.FilesUtilities import *

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
1  A 3D V model to segment ROI of uniform physical size 147mm*147mm*147mm 
   around primary cancer.
2  Total 29 patient data which exludes the patient 
   with primary cancer size exceeding 147mm*147mm*147mm.  
   In  29 patients, training data 20 patients, validation 5 patients,
   and test 6 patients;
   The primary cancer occupies the whole volume at average 7.4%.
3  For each ROI, about 7.4% pixels are primary cancer, 
    other 92.6% pixels are background;
3  Dice coefficient is 3D dice coefficient against corresponding 3D ground truth;
4  Training data augmentation in the fly: affine in XY plane, 
    translation in Z direction;
5  In the bottle neck of V model, the latent vector has size of 1024*3*3;
6  Dynamic loss weight according trainin  data;

Test Result Description:
1  First 5 patients are validation data; Second 4 patients are test data;
2  Each patient extracts its 5-6 slices at the position having ground truth. 
3  The topleft subimage is original input image, 
    title is formated Raw:ID_s{slice};  

List Dice for all patients for fold 5 Cross valiation:
ID	    	Dice
04459696	0.56622
03389601	0.86175
05422073	0.55629
03920513	0.69620
04796135	0.28082
05739718	0.73707
04119332	0.66345
02190163	0.13406
03903461	0.49957

Average     0.555    
stdev       0.228       

Simple Analysis:
1  Dice results in different patients has big difference: 86% vs 13%;
2  Big cancer gets better result;
3  Various Cancer texture give challenge, 
    maybe more training sample will help.   


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
outputLatex = r'''/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/predictResult/color/OCTestReport_20191012.tex'''

imagesPath = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/predictResult/color"

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
























