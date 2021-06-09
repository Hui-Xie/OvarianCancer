
"""
Created on Tue May 11 14:46:22 2021

@author: juikwang


Output Zeiss DICOM header

Required specific python version and library version
ENV: E-Eye013 --> test_env_1

"""

import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import sys
sys.path.append('Z:/Ray/Codes/Ray/Projects/Gitlab/modules_PY_ray')

# functions
import functionsio as fIO   # functions for input/output


folder = "Z:\\Ray\\CleanDataset\\IOWA_OCT\\2021_05_Test\\"
#file_1 = "1.2.276.0.75.2.2.42.896740156037.20170213165203155.121661155.1.dcm"
#
#######
###########
#name = os.path.join(folder, file_1)
#ds = pydicom.filereader.dcmread(name)
####
#print(ds)
#list(ds)
##(2201, 1000) Private tag data                    LT: 'CapeCodMacularCubeRawData'
##
#patient_id = ds[0x0010, 0x0020].value
#protocol = ds[0x0040, 0x0254].value
#date = ds[0x0040, 0x0244].value
#scan_time = ds[0x0040, 0x0245].value[:-4]
#
#laterality = ds[0x0020, 0x0060].value
#if laterality == "R":
#    eye = "OD"
#elif laterality == "L":
#    eye = "OS"
#else:
#    eye = "ERROR"
#                    
#[spacing_z, spacing_x] = ds[0x5200, 0x9229][0][0x0028, 0x9110][0][0x0028, 0x0030].value
### slice_thickness = float(ds[0x5200, 0x9229][0][0x0028, 0x9110][0][0x0018, 0x0050].value)*1000 # micrometer 
#img = ds.pixel_array
#plt.imshow(img)
#
#pil_image = Image.fromarray(img)
#open_cv_image = np.array(pil_image) 
#open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_YCrCb2BGR)
#pil_image = Image.fromarray(open_cv_image)
#pil_image.save("test_fundus_1.png")





#
#img_1 = np.swapaxes(img, 0, 1)
##img_1 = img_1[::-1,::-1,::-1] # Z, y, x flip
#
##
##filename = patient_id+"_"+date+"_"+scan_time+"_"+eye+"_"+protocol.replace(" ", "_")+".mhd"
#fIO.outputImgITK(img_1, 1, 1, 1, "Test.mhd")
###
#im = Image.fromarray(img)
#
#im.show()



#im.save('test.png')
#







#
#plt.imshow(img, cmap=plt.cm.gray)
#plt.show()

for root, dirs, files in os.walk(folder):
    for f in files:
        file_name, file_ext = os.path.splitext(f)
        if file_ext == ".dcm":
            ds = pydicom.filereader.dcmread(os.path.join(root, f))
            print(f)
            
            try:       
                file_type = ds[0x2201, 0x1000].value
                print("  {}".format(file_type)) # 
            except:
                sop_class_uid = ds[0x0008, 0x0016].value
                if sop_class_uid == "1.2.840.10008.5.1.4.1.1.104.1":
                    file_type = "Encapsulated PDF Storage"
                elif sop_class_uid ==  "1.2.840.10008.5.1.4.1.1.7":
                    file_type = "Color Fundus Image"
            
            if ("PDF" in file_type) or ("Pdf" in file_type):
            
                document_title = ds[0x0042, 0x0010].value
                print("  {}".format(document_title)) # Document title
                
                if (document_title == "Visual Field") or (document_title == "SFA") or ("SITA" in document_title) or ("SUMMARY" in document_title):
                    laterality = ds[0x0020, 0x0060].value
                    if laterality == "R":
                        eye = "OD"
                    elif laterality == "L":
                        eye = "OS"
                    else:
                        print(f, "  Error!!")
                    
                    try:
                        os.rename(os.path.join(root, f), os.path.join(root, eye+ " " +document_title+".pdf"))
                    except:
                        new_name = "repeated_"+f.replace(".dcm", ".pdf")
                        os.rename(os.path.join(root, f), os.path.join(root, new_name)) 
                        
                elif (document_title == "_Color - T1"):
                    patient_id = ds[0x0010, 0x0020].value  
                    os.rename(os.path.join(root, f), os.path.join(root, patient_id+"_All_Color_Fundus_Photo.pdf"))
                
                elif (document_title == ""):
                    laterality = ds[0x0020, 0x0060].value
                    if laterality == "R":
                        eye = "OD"
                    elif laterality == "L":
                        eye = "OS"
                    else:
                        eye = "ERROR"
                    filename = eye + " GOLDMANN VISUAL FIELDM.pdf"
                    os.rename(os.path.join(root, f), os.path.join(root, filename))
                    
                else:
                    try:
                        os.rename(os.path.join(root, f), os.path.join(root, document_title+".pdf"))
                    except:            
                        os.rename(os.path.join(root, f), os.path.join(root, "repeated_"+f.replace(".dcm", ".pdf")))
                        print("Repeated file...")
                        
                        
            elif (file_type == "HfaOphthalmicVisualFieldStaticPerimetryMeasurements"):
                patient_id = ds[0x0010, 0x0020].value
                laterality = ds[0x0020, 0x0060].value
                if laterality == "R":
                    eye = "OD"
                elif laterality == "L":
                    eye = "OS"
                else:
                    eye = "ERROR"
                os.rename(os.path.join(root, f), os.path.join(root, patient_id+"_"+eye+"_"+file_type+".dcm"))
            
            elif (file_type == "Color Fundus Image"):
                patient_id = ds[0x0010, 0x0020].value
                date_time = ds[0x0008, 0x002a].value[:-4]
                
                # Speciral color interpretation: 
                # (0028, 0004) Photometric Interpretation          CS: 'YBR_FULL_422'
                img = ds.pixel_array
                pil_image = Image.fromarray(img)
                open_cv_image = np.array(pil_image) 
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_YCrCb2BGR)
                pil_image = Image.fromarray(open_cv_image)

                filename = patient_id + "_" + date_time + "_ColorFundus.png"
                pil_image.save(os.path.join(root, filename))                    
                os.remove(os.path.join(root, f))                
        
        
            else:              
                patient_id = ds[0x0010, 0x0020].value                
                laterality = ds[0x0020, 0x0060].value
                if laterality == "R":
                    eye = "OD"
                elif laterality == "L":
                    eye = "OS"
                else:
                    eye = "ERROR"
                     
                date_time = ds[0x0008, 0x002a].value[:-4]
                print("  date time: {}".format(date_time))                    
                
                
                modality = ds[0x0008, 0x0060].value
                if (modality == "OP"): # SLO image
                    #print("  {}".format("OP: 2D image"))
                    code_value = ds[0x0022, 0x0015][0][0x008, 0x0100].value
                    #print("  {}, eye: {}, image shape: {}".format(code_value, eye, ds.pixel_array.shape))

                    
                    if (file_type == "HfaPerimetryOphthalmicPhotography8BitImage"):
                        os.rename(os.path.join(root, f), os.path.join(root, patient_id + "_" + date_time + "_" + eye + "_" + file_type + ".dcm"))
                    elif (file_type == "OphthalmicPhotography8BitImage"):
                        lso_img = ds.pixel_array
                        im = Image.fromarray(lso_img)
                        
                        filename = patient_id + "_" + date_time + "_" + eye + "_LSO.png"
                        im.save(os.path.join(root, filename))                    
                        os.remove(os.path.join(root, f))
                        
                    
                    
                elif (modality == "OPV") and (file_type=="HfaVisualFieldRawData"):
#                    print("  id: {}, eye: {}".format(patient_id, laterality))
                    #print("  {}".format("visual filed raw data"))
                    os.rename(os.path.join(root, f), os.path.join(root, patient_id + "_" + date_time + "_" + eye + "_" + file_type + ".dcm"))
                   
                    
                elif (modality == "OPT"):
                    print("  {}".format(modality))
#                    series = ds[0x0008, 0x0031].value
#                    print("  {}".format(series)) # Series 

                    sd = series_Description = ds[0x0008, 0x103E].value
                    print("  {}".format(series_Description)) # Series
#                    protocol = ds[0x0040, 0x0254].value
#                    print("  {}".format(protocol)) # Series
                    
                    # slice_thickness = float(ds[0x5200, 0x9229][0][0x0028, 0x9110][0][0x0018, 0x0050].value)*1000 # micrometer                     
                    
                    
                    
                    # OCT volumetric scan
                    if ("OphthalmicTomographyImage" == file_type) and (sd == "Macular Cube 200x200"):
                        [spacing_z, spacing_x] = ds[0x5200, 0x9229][0][0x0028, 0x9110][0][0x0028, 0x0030].value
                        
                        img = ds.pixel_array
                        img_1 = np.swapaxes(img, 0, 1)
                        img_1 = img_1[::-1,::-1,::-1] # Z, y, x flip
                                                
                        filename = patient_id+"_"+date_time+"_"+eye+"_"+sd.replace(" ", "-")+".mhd"
                        fIO.outputImgITK(img_1, spacing_z, spacing_x, spacing_x, os.path.join(root,filename))                    
                        os.remove(os.path.join(root, f))                   

                    elif ("OphthalmicTomographyImage" == file_type) and [( (sd == "RASTER_21_LINES") or (sd == "HD 5 Line Raster") or (sd == "5 Line Raster")) or (sd == "RASTER_RADIAL")]:
                        [spacing_z, spacing_x] = ds[0x5200, 0x9229][0][0x0028, 0x9110][0][0x0028, 0x0030].value
                        
                        img = ds.pixel_array
                        img_1 = np.swapaxes(img, 0, 1)
                        img_1 = img_1[:,::-1,:]
                        
                        filename = patient_id+"_"+date_time+"_"+eye+"_"+sd.replace(" ", "-")+".mhd"
                        fIO.outputImgITK(img_1, spacing_x, 1, spacing_z, os.path.join(root,filename))                    
                        os.remove(os.path.join(root, f))    

                    elif ("CapeCodGuidedProgressionAnalysisRawData" == file_type): # can't access pixel_array                      
                        filename = patient_id + "_" + date_time + "_" + eye + "_" + file_type + "_" + sd.replace(" ", "-") + ".dcm"  
                        os.rename(os.path.join(root, f), os.path.join(root, filename))
                                           
                      
                    elif ("CapeCodMacularCubeRawData" == file_type): # can't access pixel_array                      
                        filename = patient_id + "_" + date_time + "_" + eye + "_" + file_type + "_" + sd.replace(" ", "-") + ".dcm"  
                        os.rename(os.path.join(root, f), os.path.join(root, filename))
                      
                    elif ("CapeCodMacularCubeAnalysisRawData" == file_type): # can't access pixel_array
                        diff_type = ds[0x0409, 0x1014].value
                        filename = patient_id + "_" + date_time + "_" + eye + "_" + file_type + "_" + sd.replace(" ", "-") + "_" + diff_type + ".dcm"
                        os.rename(os.path.join(root, f), os.path.join(root, filename))
                    
                    elif ( "Optic Disc Cube 200x200" == series_Description): # can't access pixel_array
                        filename = patient_id + "_" + date_time + "_" + eye + "_" + file_type + "_" + sd.replace(" ", "-") + ".dcm"  
                        os.rename(os.path.join(root, f), os.path.join(root, filename))                        

                    elif ( "CapeCodOpticDiscAnalysisRawData" == file_type): # can't access pixel_array
                        diff_type = ds[0x0409, 0x1014].value
                        filename = patient_id + "_" + date_time + "_" + eye + "_" + file_type + "_" + sd.replace(" ", "-") + "_" + diff_type + ".dcm"
                        os.rename(os.path.join(root, f), os.path.join(root, filename))
 
                    elif ( "CapeCodRasterRawData" == file_type):     # can't access pixel_array
                        filename = patient_id + "_" + date_time + "_" + eye + "_" + file_type + "_" + sd.replace(" ", "-") + ".dcm"  
                        os.rename(os.path.join(root, f), os.path.join(root, filename))     
        
                    elif ( "CapeCodEnhancedRasterRawData" == file_type):     # can't access pixel_array
                        filename = patient_id + "_" + date_time + "_" + eye + "_" + file_type + "_" + sd.replace(" ", "-") + ".dcm"  
                        os.rename(os.path.join(root, f), os.path.join(root, filename))     
    
     
    
    
    
    
                else:
                    continue
            
            
            
            

#            print("  {}".format(ds[0x0008, 0x0060].value)) # Modality
#            modality = ds[0x0008, 0x0060].value










































