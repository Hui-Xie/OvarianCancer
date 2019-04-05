# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:54:05 2018

@author: juikwang
"""

import subprocess
import sys
import os

from operator import itemgetter
from datetime import datetime

import time
import datetime
today = str(time.localtime()[1]) + "-" + str(time.localtime()[2]) + "-" + str(time.localtime()[0]) 


home_path = os.getcwd()


logfile = os.path.join(home_path, "batch.log")
logFileOut = open(logfile, "w")

texFile = os.path.join(home_path, "report_enface_images.tex")
texFileOut = open(texFile, "w")
texFileOut.write("\\documentclass[slidestop, compress, red, mathserif, xcolor=dvipsnames]{beamer}" + "\n")
texFileOut.write("\\usepackage{grffile}" + "\n")
texFileOut.write("\\usepackage{tikz}" + "\n")
texFileOut.write("\\usepackage{lmodern}" + "\n")

texFileOut.write("\\title{\\LARGE{\\textbf{\\underline{XXXXXXXXXXXXXXXXXXXXXXXXXXX}}}}" + "\n")
texFileOut.write("\\author{\\Large{\\textbf{KKKKKKKKKKKKKKKKK}}}" + "\n" )
texFileOut.write("\\institute {\\Large{\\textbf{IIIIIIIIIIIIIIIIII}}}" + "\n" )

texFileOut.write("\\date{" + today +"}" + "\n")

texFileOut.write("\\setbeamersize{text margin left=0.15cm,text margin right=0.15cm}" + "\n")

texFileOut.write("\\begin{document}" + "\n")
texFileOut.write("\\maketitle" + "\n\n\n")


# Find the files to process
for root, dirs, files in os.walk(home_path, topdown=False):
    for filename in files:
        # Check if the file exists
        if (filename.find("xxxxxxxx.kkk") != -1):
            print("Processing...", os.path.join(root))
            
            # Prepare the image files
            img = os.path.join(root, "sss.png").replace('\\', '/')
           
            # Start to write the info to latex 
            texFileOut.write("\\begin{frame}" + "\n")
            slide_title = diagnosis + " --- " + subject_name + ", {\color{ForestGreen}{" + visit_date + "}}, " + eye
            print(" ", slide_title)
            texFileOut.write("\\frametitle{\\centerline{\\large{\\textbf{\\underline{" + slide_title + " }}}}}" + "\n")            

            texFileOut.write("\\begin{tikzpicture}" + "\n")
            texFileOut.write("\\draw[step=0.5cm,color=white] (0cm,0cm) grid (12cm, 8cm);" + "\n")   

           
            texFileOut.write("\\node[anchor=north west,inner sep=0] at (0.2, 2.5) {\\includegraphics[width = 3cm, height = 3cm]{" + img + "}}; \n") 
            texFileOut.write("\\node[inner sep=0, font=\\tiny] at (1.2,2.8)   {\\textbf{\\textit{IMG}}}; \n")

            texFileOut.write("\\end{tikzpicture}" + "\n")
            texFileOut.write("\\end{frame}" + "\n\n\n")

texFileOut.write("\\end{document}" + "\n") 
texFileOut.close()
logFileOut.close() 





















