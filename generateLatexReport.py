
import os

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

latexTail = r'''

\end{document}

'''

latexItem = r'''

\begin{figure}
	\begin{subfigure}{\linewidth}
		\centering
		\includegraphics[scale=0.7]{LabelFile}
		\caption{Ground Truth+CT}
	\end{subfigure}

	\begin{subfigure}{\linewidth}
		\centering
		\includegraphics[scale=0.7]{CTFile}
		\caption{CT}
	\end{subfigure}

	\begin{subfigure}{\linewidth}
		\centering
		\includegraphics[scale=0.7]{SegFile}
    	\caption{Segmentation+CT}
	\end{subfigure}

\caption{\detokenize{PatientID_Slice}}
\end{figure}

\clearpage  % support too many fingures

'''
outputLatex = r'''/home/hxie1/temp/OvarianCancerReport.tex'''

#imagesPath = r'''/home/hxie1/c-xwu000/data/OvarianCancerCT/Extract_uniform/segmented'''
imagesPath = r'''/home/hxie1/data/OvarianCancerCT/Extract_uniform/segmented'''

with open(outputLatex, "w") as f:
    f.write(latexHead)

    originalCwd = os.getcwd()
    os.chdir(imagesPath)
    labelMergeList = [os.path.abspath(x) for x in os.listdir(imagesPath) if '_LabelMerge.png' in x]
    os.chdir(originalCwd)

    for labelFile in labelMergeList:
        ctFile = labelFile.replace('_LabelMerge.png', '.png')
        segFile = labelFile.replace('_LabelMerge.png', '_SegMerge.png')

        basename = os.path.basename(ctFile)
        patientID_Slice = basename[0: basename.find('.png')]
        patientID_Slice = 'PatitenID_Slice: ' + patientID_Slice

        item = latexItem.replace('LabelFile', labelFile).replace('CTFile', ctFile).replace('SegFile', segFile).replace('PatientID_Slice', patientID_Slice)
        f.write(item)

    f.write(latexTail)

print(f'{outputLatex} has been outputed.')
























