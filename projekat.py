#za prikazivanje slika
from __future__ import print_function

#import potrebnih biblioteka
import numpy as np
#import imutils
import cv2
import imutils

import collections
import winsound

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

#Sklearn biblioteka sa  K-NN algoritmom
from sklearn.cross_validation import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure

from sklearn.cluster import KMeans
from scipy import ndimage
from scipy import stats
from vector import *


from sklearn.datasets import fetch_mldata



brojevi = []
id = -1
kernel = np.ones((2,2),np.uint8)

counter = 0




# FUNKCIJE REGRESIONI METOD


# na papiru nadjena formula
def odrediKoeficijentePrave(x1,y1,x2,y2):
   
    # k = (y - y0)/(x-x0)
    k = ((y2.astype(float))-(y1.astype(float))) / ((x2.astype(float))-(x1.astype(float)))
    
	# n = ((y - y0)*x0 - (x - x0)*y0)/(x-x0)
    n = - (((y2.astype(float))-(y1.astype(float)))*(x1.astype(float)) - ((x2.astype(float))-(x1.astype(float)))*(y1.astype(float)) )/((x2.astype(float))-(x1.astype(float)))
    
    
    #k = (y2-y1) / (x2-x1)
    #n = - ((y2-y1)*x1 - (x2-x1)*y1 )/(x2-x1)
   
    return k,n

# https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python/20679579
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    
    return x, y


# KRAJ FUNKCIJA REGRESIONI METOD

def idGenerator():
    global id
    id += 1
    return id

def kreirajFrejm(video):    
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    cv2.imwrite("frame-%d.jpg" % 0, image)     

def pomocnaFunkcijaNadjiNajblizi(broj,odgovarajuciBrojevi):
    najblizi = odgovarajuciBrojevi[0]
    minDistance = distance(broj['center'],odgovarajuciBrojevi[0]['center'])

    
    for br in odgovarajuciBrojevi:
        if (distance(br['center'],broj['center']) < minDistance):
            minDistance = distance(br['center'],broj['center'])
            najblizi = br
    
    najbliziNiz = []
    najbliziNiz.append(najblizi)
    
    return najbliziNiz


def nadjiKontureNaSlici(img_frame_bin):
    img, contours, hierarchy = cv2.findContours(img_frame_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_barcode = [] #ovde ce biti samo konture koje pripadaju bar-kodu
    for contour in contours: # za svaku konturu
        center, size, angle = cv2.minAreaRect(contour) # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        #if width > 5 and width < 70 and height > 25 and height < 100: # uslov da kontura pripada bar-kodu
        contours_barcode.append(contour) # ova kontura pripada bar-kodu
    
    for con in contours_barcode:
        x,y,w,h = cv2.boundingRect(con)
        dim = (28, 28)
        slika = img_frame[y:y+h, x:x+w]
        
        slika = cv2.resize(slika, dim, interpolation = cv2.INTER_LINEAR )
        
        cv2.imshow("slika", slika)
        cv2.waitKey()
    
    img = img_frame.copy()
    cv2.drawContours(img, contours_barcode, -1, (255, 0, 0), 1)
    print ('Ukupan broj regiona: %d' % len(contours_barcode))
    
    #plt.imshow(img, 'gray')
    return img
            

def nadjiNajblizi(broj,brojevi):
    odgovarajuciBrojevi = []
    for br in brojevi:
        if (distance(br['center'],broj['center'])<20):
            odgovarajuciBrojevi.append(br)
            
    if len(odgovarajuciBrojevi)>1 :
        return pomocnaFunkcijaNadjiNajblizi(broj,odgovarajuciBrojevi)
    else :
        return odgovarajuciBrojevi

def trenirajKNN():
    
    with np.load('knn_data.npz') as data:
        
        train = data['train']
        train_labels = data['train_labels']
    
        knn = cv2.ml.KNearest_create()
        knn.train(train.astype(np.float32), cv2.ml.ROW_SAMPLE, train_labels.astype(np.float32))
        
        
        img = cv2.imread('digits.png')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
        
        # Make it into a Numpy array. It size will be (50,100,20,20)
        x = np.array(cells)[0].astype(np.float32)
        
        
        
        test = x[0].astype(np.float32) # Size = (2500,400)
        
        
        ret,result,neighbours,dist = knn.findNearest(x,k=1)
        return result

def prepoznajLinije(img):
    kernel = np.ones((2,2),np.uint8)
    gray = cv2.dilate(img,kernel)
    
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    
    lines = cv2.HoughLinesP(edges,1,np.pi/180,40,600,8)
#    (edges, 1, np.pi/180,40, minLineLenght, maxLineGap

    
    xmin = lines[0][0][0]
    ymin = lines[0][0][1]
    xmax = lines[0][0][2]
    ymax = lines[0][0][3]
    
    for i in  range(len(lines)):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        if  x1 < xmin:
            ymin = y1
            xmin = x1
        if  x2 > xmax:
            xmax = x2
            ymax = y2
                
    
#    backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
#    cv2.circle(backtorgb, (xmin,ymin), 4, (25, 255, 25), 1)
#    cv2.circle(backtorgb, (xmax,ymax), 4, (25, 255, 25), 1)
#    cv2.imshow("test2",backtorgb)
#    cv2.waitKey(0)
    
    
    return xmin,ymin,xmax,ymax
        



    
#https://gurus.pyimagesearch.com/lesson-sample-k-nearest-neighbor-classification/
def makeModelKNNP():
    
    mnist = fetch_mldata('MNIST original')
    data=mnist.data;
    labels = mnist.target.astype('int')
    velicina = len(mnist.data)
    trainData = mnist.data
    trainLabels = labels
    #test dataset
#    valData = data[test_subset]
#    valLabels = labels[test_subset]
#   print(velicina)
# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k

#    kVals = range(1, 30, 2)
#    accuracies = []
## loop over various values of `k` for the k-Nearest Neighbor classifier
#    
#   for k in xrange(1, 30, 2):
#	# train the k-Nearest Neighbor classifier with the current value of `k`
#   
#   	model = KNeighborsClassifier(n_neighbors=k)
#   	model.fit(trainData, trainLabels)  
#
#	# evaluate the model and update the accuracies list
#   	score = model.score(valData, valLabels)
#   	print("k=%d, accuracy=%.2f%%" % (k, score * 100))
#   	accuracies.append(score)
#      
#
#
## find the value of k that has the largest accuracy
#   i = np.argmax(accuracies)
#   print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],accuracies[i] * 100))

    
# re-train our classifier using the best k value and predict the labels of the
# test data
    print("=-------------------------------------------------");
    mojaLista=[];
    print("velicina data seta: %d"%len(trainData));
    for it in range(0,len(trainData)):
        pom=trainData[it];
        dim = (28, 28)
        #prebacujem iz vektora u matricu
        pom = pom.reshape(dim).astype("uint8")
        rescale_bottom = 0
        rescale_up = 255
        pom = exposure.rescale_intensity(pom, out_range=(rescale_bottom, rescale_up))
        #print (pom)
        
        tresh_bottom = 127
        tresh_up = 255
        ret,pom=cv2.threshold(pom,tresh_bottom, tresh_up, cv2.THRESH_BINARY);
        #cv2.imshow('original',pom);
        #cv2.waitKey()

        pom_bin=pom.copy();
        iterationsOnce = 1
        kernel=np.ones((3,3));
        pom=cv2.dilate(pom,kernel,iterationsOnce);
        kernel=np.ones((1,1));
        iterationsTwice = 2
        pom=cv2.erode(pom,kernel,iterationsTwice);
        #pom=cv2.bitwise_not(pom);
        #cv2.imshow('region',pom);
        #cv2.waitKey();
        img2,contours,hierarchy =cv2.findContours(pom,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
        #print("broj kontura je:%d"%len(contours));
        izmedju = False
        #for cnt in contour:
            
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt); 
            if(h<15 or w<=7):
                #print("nije broj");
                izmedju = True
            else:
                   dim = (28, 28)
                   #print("jeste broj")
                   pom_cropped=pom_bin[y:y+h,x: x+w];
                   #print("sirina:%d,i visina:%d"%(w, h));
                   
                   pom_cropped = cv2.resize(pom_cropped, dim, interpolation = cv2.INTER_CUBIC )
                   pom_cropped_copy=pom_cropped.copy();
                   #print ("labela:%d"%trainLabels[it]);
                   #print ("rbr broj: %d"%it);
                   #if(trainLabels[it]==0):
                   #    cv2.imshow('kropovan',pom_cropped);
                   #    cv2.waitKey()
                   maks = 255
                   #plt.figure();
                   #plt.imshow(pom_cropped,'gray');
                   pom_cropped=pom_cropped.flatten();
                   pom_cropped = np.reshape(pom_cropped, (1,-1))
                   
                   
                   
                   minim = 0
                   
                   cv2.rectangle(pom_bin,(x,y),(x+w,y+h),(minim,maks,minim),2)
                   trainData[it]=pom_cropped;
                   mojaLista.append(pom_cropped);
                   
        #cv2.imshow('iscrtani reg',pom_cropped_copy);
        #cv2.waitKey()
    #print("------------------------------------");
    #print("broj mojih:%d"%len(mojaLista));
    i = 1
    model = KNeighborsClassifier(i)
    model.fit(trainData, trainLabels)
    

    return model;    


model = makeModelKNNP();
print("Obuka zavrsena.")

def vratiBrojNaOsnovuSlike(iseceniBroj):
    
    xdim = 28
    ydim = 28
    
    # resize sliku na 28x28 da se poklapa sa onom iz dataset-a
    iseceniBrojResizovan = cv2.resize(iseceniBroj, (xdim, ydim), interpolation = cv2.INTER_CUBIC)
    
    tip = type(iseceniBrojResizovan)
    # transformisi u oblik pogodan za uporedjivanje sa dataset-om
    iseceniBrojFlattenovan = iseceniBrojResizovan.flatten();
    
    tip2 = type(iseceniBrojFlattenovan)
    
    #print(tip)
    #print("A drugi je", tip2)
    
    # resava gresku 
    iseceniBrojReshapeovan = np.reshape(iseceniBrojFlattenovan, (1,-1))
    numberPrediction = model.predict(iseceniBrojReshapeovan);
    
    nadjenBroj = numberPrediction[0]
    
    #print("Predvidja se broj: ")
    #print (nadjenBroj)
             
   
    #cv2.imshow('Pronasao sledeci broj: ',iseceniBroj);
    #cv2.waitKey(0)
    return nadjenBroj;
 
    


def main():
    
    # FOR NE RADI PA CE SE MORATI OVAKO POKRETATI
    x=4
    if(x==4):
    #for x in range(0,10):
        print ("Video redni broj: ",x)
        video = "video-"
        video += str(x)
        video += ".avi"
        vid = cv2.VideoCapture(video)
        
        kreirajFrejm(video)
        img = cv2.imread('frame-0.jpg')
        
        suma = 0
        suma1 = 0
        frejm = 0
        test = 0
          
    
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
        #                               donja granica prva    gornja granica prva
        mask1 = cv2.inRange(img,np.array([0, 230, 0]),np.array([155, 255, 155]))
        #                               donja granica druga    gornja granica druga
        mask2 = cv2.inRange(img,np.array([230, 0, 0]),np.array([255, 155, 155]))
        #==========================Kraj maskiranja slika
        ivice1 = prepoznajLinije(mask1)
        ivice2 = prepoznajLinije(mask2)
       

        
        while  (1) :
            ret, currentFrame = vid.read()
            
            if not ret: 
                print ("Frejm nije ucitan, kraj izvrsavanja.")
                break
            else:
                test += 1
                #print ()
            frejm += 1
            
            for br in brojevi:
               test += 1
               if (frejm - br['frame']) <=5 :
                   cv2.putText(currentFrame,  str(br['id']), (br['center'][0],br['center'][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                   cv2.circle(currentFrame, (br['center'][0], br['center'][1]), 16, (25, 25, 255), 1)
                   
               
            
           
            donja = 220
            gornja = 255
            lower = np.array([donja, donja, donja],dtype = "uint8")
            
            
            upper = np.array([gornja, gornja, gornja],dtype = "uint8")
            
            trasholdImage = cv2.inRange(currentFrame, lower, upper)
           
            mnozilac = 1.0
            bw_img = trasholdImage * mnozilac
            
            
            bw_img = cv2.dilate(bw_img,kernel)
            bw_img = cv2.dilate(bw_img,kernel)
            #cv2.imshow('fr', bw_img)
            #cv2.waitKey(0)
            #FIND CONTURES 
            # PRONASAO SVE STO IZGLEDA KAO BROJ I STAVIO LABELU NA TO
            # PRONASAO TE OBJEKTE PO LABELAMA I STAVIO IH U NIZ PICELEM
            labeled, _ = ndimage.label(bw_img)
            pictureElements = ndimage.find_objects(labeled)
            
            for i in range(len(pictureElements)) :
                
                
    
                center = []
                dimension = []
                location = pictureElements[i]

                dimension.append(location[1].stop - location[1].start)
                dimension.append(location[0].stop - location[0].start)                
                
                center.append((location[1].stop + location[1].start) /2)
                center.append((location[0].stop + location[0].start) /2)
                
                

                
                if dimension[0]>=9 and dimension[1]>=9 :
                    crop_img = bw_img[location[0].start:location[0].stop,location[1].start:location[1].stop]
                    
                    broj = {'center': (center[0], center[1]), 'size': (dimension[0], dimension[1]), 'frame': frejm, 'img' : crop_img}
                    
                    istiBrojUProslomFrejmu = nadjiNajblizi( broj, brojevi)
                    
                    # https://github.com/ftn-ai-lab/sc-2016-e2/blob/master/teorija/notebooks/SC09-video-pracenje_objekata.ipynb
                    if len(istiBrojUProslomFrejmu) == 0:
                        
                        broj['prosaoPrvu'] = False
                        broj['prosaoDrugu'] = False
                        broj['id'] = idGenerator()
                        broj['tokKretanja'] = [{'center': (center[0], center[1]), 'size': (dimension[0], dimension[1]), 'frame': (dimension[0], dimension[1])}]
                        brojevi.append(broj)
                        
                    elif len(istiBrojUProslomFrejmu) == 1:
                        
                        istiBrojUProslomFrejmu[0]['frame'] = frejm
                        istiBrojUProslomFrejmu[0]['center'] = broj['center']
                        istiBrojUProslomFrejmu[0]['tokKretanja'].append({'center': (center[0], center[1]), 'size': (dimension[0], dimension[1]), 'frame': frejm})
                    

        # trenutni frejm - frejm ; br[frame] frejm u kome je broj trenutni
            for br in brojevi:
                x = []
                y = [] 
                if( (frejm-br['frame']) > 20 ):
                   
                   for p in br['tokKretanja']:
                       x.append(p['center'][0])
                       y.append(p['center'][1])
                
                   
                    
                   
                   n, c  = odrediKoeficijentePrave(ivice1[0], ivice1[1], ivice1[2], ivice1[3])
                   slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                   
                   nPlave, cPlave = odrediKoeficijentePrave(ivice2[0], ivice2[1], ivice2[2], ivice2[3])
                    

                   
                   tackaPresekaXPlave = (cPlave - intercept) / (slope-nPlave)                   
                   tackaPresekaYPlave = slope*tackaPresekaXPlave + intercept
                   
                   TP_PLAVE = (tackaPresekaXPlave,tackaPresekaYPlave)

                   
                   tackaPresekaX = (c - intercept) / (slope-n)                   
                   tackaPresekaY = slope*tackaPresekaX + intercept
                   
                   TP_ZELENE = (tackaPresekaX, tackaPresekaY)

                    
                   #presekProveraX = line_intersection( ( (ivice1[0], ivice1[1]), (ivice1[2], ivice1[3]) ), ( (x[0], y[0]), (x[-1], y[-1]) ) )
                    
                   #print ("moji ", tackaPresekaX, " ", tackaPresekaY)
                   #print ("line_inters ", presekProveraX, " ", presekProveraY)
                   #print("Tacka preseka plave uredjen par: ", TP_PLAVE)
                   #print("Tacka preseka zelene uredjen par: ", TP_ZELENE)
                   
                   if(tackaPresekaX > ivice1[0] and tackaPresekaX < ivice1[2]):
                       if(y[0] < tackaPresekaY):
                           suma -= vratiBrojNaOsnovuSlike(br['img'])
                           #print ("Prosao preko zelene, id broja: ", br['id'])
                           
                           
                   if(tackaPresekaXPlave > ivice2[0] and tackaPresekaXPlave < ivice2[2]):
                       if(y[0] < tackaPresekaYPlave):
                           suma += vratiBrojNaOsnovuSlike(br['img'])
                           #print ("Prosao preko plave, id broja: ", br['id'])
            
                   brojevi.remove(br)
                        #DRUGI NACIN BEGIN
                
                #========================= pronadje sliku broja na videu
            for br in brojevi:
    
                
                distancaBrojaOdLinije1 = pnt2line(br['center'], (ivice1[0], ivice1[1]),(ivice1[2], ivice1[3]))
                #print(distancaBrojaOdLinije1)
                
                if(distancaBrojaOdLinije1 < 10):
                    vecaOdLinije1 = False
                
                distancaBrojaOdLinije2 = pnt2line(br['center'], (ivice2[0], ivice2[1]),(ivice2[2], ivice2[3]))
                if(distancaBrojaOdLinije2 < 10):
                    vecaOdLinije2 = False
                    
                #print(distancaBrojaOdLinije2)
                    #print(vecaOdLinije1, vecaOdLinije2)
                    
                #    #cv2.line(img, pnt1, el['center'], (0, 255, 25), 1)
                #print("IF provera prvi param distancabroja treba manja od 10 : ", distancaBrojaOdLinije1)    
                #print("IF provera drugi param bool treba da je false : ", br['prosaoPrvu'])    
                    
                if (distancaBrojaOdLinije1 < 10 and not br['prosaoPrvu']):
                    #print ("doslo do kolizije sa prvom, a id broja je: ", br['id'])
                        #ovde transformacija
                    prediction = vratiBrojNaOsnovuSlike(br['img'])
                    suma1 = suma1 - prediction

                    if br['prosaoPrvu'] == False:
                        br['prosaoPrvu'] = True
                            
                        #dodati vrednost na sumu
                            
                if (distancaBrojaOdLinije2 < 10 and not br['prosaoDrugu']):
                    #print ("doslo do kolizije sa drugom, a id broja je: ", br['id'])
                    prediction = vratiBrojNaOsnovuSlike(br['img'])
                    suma1 = suma1 + prediction
                        #ispis = "SUMA: "
                        #ispis += str(suma)
                        
                        #cv2.putText(currentFrame ,ispis,(50,50), cv2.FONT_HERSHEY_SIMPLEX,  2,(255,255,255),1,cv2.LINE_AA)  
                        
                        #cv2.imshow("druga linija",currentFrame)
                        #cv2.waitKey(0)#cv2.imshow(str(br['id']),br['img'])
                    if br['prosaoDrugu'] == False:
                        br['prosaoDrugu'] = True
                        #oduzeti vrednost od sume
                            
                # DRUGI NACIN END  
	
    	 
	test = 0
       	                      
	for br in brojevi:           
	   x = []
	   y = [] 
       
	   for p in br['tokKretanja']:
		   x.append(p['center'][0])
		   y.append(p['center'][1])
	
	   
		
	   
	   
	   n, c  = odrediKoeficijentePrave(ivice1[0], ivice1[1], ivice1[2], ivice1[3])
	   slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
	   
	   nPlave, cPlave = odrediKoeficijentePrave(ivice2[0], ivice2[1], ivice2[2], ivice2[3])
		

	   


	   tackaPresekaXPlave = (cPlave - intercept) / (slope-nPlave)
	   tackaPresekaYPlave = slope*tackaPresekaXPlave + intercept

	   TP_PLAVE = (tackaPresekaXPlave,tackaPresekaYPlave)       

	   tackaPresekaX = (c - intercept) / (slope-n)	
	   tackaPresekaY = slope*tackaPresekaX + intercept
	
	   TP_ZELENE = (tackaPresekaX, tackaPresekaY)	          
	   #presekProveraX = line_intersection( ( (ivice1[0], ivice1[1]), (ivice1[2], ivice1[3]) ), ( (x[0], y[0]), (x[-1], y[-1]) ) )
		
	   #print ("moji ", tackaPresekaX, " ", tackaPresekaY)
	   #print ("line_inters ", presekProveraX, " ", presekProveraY)
	   #print(test)      
	   #print("Tacka preseka plave uredjen par: ", TP_PLAVE)
	   #print("Tacka preseka zelene uredjen par: ", TP_ZELENE)	   
	   razlika = tackaPresekaY - y[-1]
	   if(tackaPresekaX > ivice1[0] and tackaPresekaX < ivice1[2]):
		   if(y[0] < tackaPresekaY):
			   if(razlika < 30):
				   suma -= vratiBrojNaOsnovuSlike(br['img'])
				   #br['prosaoDrugu'] = True
				   #print ("Prosao preko zelene, id broja zaostali broj: ", br['id'])
			   
	   razlika = tackaPresekaYPlave - y[-1]	   
	   if(tackaPresekaXPlave > ivice2[0] and tackaPresekaXPlave < ivice2[2]):
		   if(y[0] < tackaPresekaYPlave):
			   if(razlika < 30):
				   suma += vratiBrojNaOsnovuSlike(br['img'])
				   #br['prosaoPrvu'] = True
				   #print ("Prosao preko plave, id broja zaostali broj: ", br['id'])                       
                   
      
    
    
    
        print ("Suma racunata na nacin sa regresionim pravama je: ", suma)
        print ("Suma racunata na nacin sa distancom blizu linije je: " , suma1)
        vid.release()
        cv2.destroyAllWindows()

    
main()  
