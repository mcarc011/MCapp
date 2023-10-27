import matplotlib.pyplot as plt
import numpy as np
import time
from shutil import copy as Hcopy
import streamlit
#%%


chckbox = ['LENT','SLAB','ROUND','PAL_HT','PAL_W']
default = ['REQ','DO','PIND','TIND','PTOK','RNGD','RNGH']
LocArr = ['PATH','BLUPC','BLADD','PRISM']

Frame = ['DBL','HBOX','VBOX','FED','PANTO','WRAP']
Perscr = ['SPH','CYL','AX','ADD','SEGHT','IPD','PRVM','PRVA']
#Prsmprsc = ['PRVM','PRVA']
Blank = ['LIND','FRNT','BACK','BLADD','BCTHK','DIA']#,'PRP']
Lens = ['LNAM','LENT','MINCTR','MINEDG','BCOCIN','BCOCUP','SLAB','ROUND','PAL_W']
Other = ['PRVM','PRVA','LNAM','LENT','ROUND','REQ','DO','PIND','TIND','RNGD','RNGH','PAL_HT','ROUND','MINEDG','FEDAX']
Checkbox = ['PTOK','PAL_W','SLAB','HRES']
Entries = [[Frame,'FRAME'],[Perscr,'RX'],[Blank,'BLANK'],[Lens,'DESIGN']]#,[Prsmprsc,'PRISM']
Rename = {'PAL_W':'REF','PTOK':'VTHIN','MINCTR':'MTHK'}

    
'============LED============='    
#Coordinates
x = np.arange(-42.5,43.0,0.5)
Y,X = np.meshgrid(x,x)
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan(Y/(-X+0.00000001))
ZT = R-R

'============================='
'=======[Formula Sheet]======='
'============================='

'---Optical Formulas---'
#This is just all the optical formulas

def Add_Round(Add):
    NewAdd = round(670.0/(740.0/Add),2)
    if NewAdd > 1.00:
        Deci = NewAdd - int(NewAdd)
    else:
        Deci = NewAdd
    for i in range(5): 
        if (i*0.25 - 0.125) <= Deci < (i*0.25 + .125):
            Deci = i*0.25
    NewAdd = int(NewAdd) + Deci
    return NewAdd

def Aspheric(r,c,k):
    Z = (r**2)/(c * (1 + np.sqrt(1-(1+k)*((r/c)**2))))
    return Z
    
def Blank_Back_Radius(ET,CT,Fc,Dia):
    Bc = ((Dia/2.0)**2 + (ET - CT + (Fc - np.sqrt(Fc**2 - (Dia/2.0)**2)))**2)/(2*(ET - CT + (Fc - np.sqrt(Fc**2 - (Dia/2.0)**2))))
    return Bc

def RoundBi(P,ins,N,h,D):
    Ri = np.sqrt((X-ins)**2 + (Y-h)**2)
    rc = 1000*(1-N)/(P+0.00000001)
    Zb = Sag(Ri,rc)
    for i in range(171):
        for j in range(171):
            if Ri[i,j] >=D:
                Zb[i,j] = Sag(D,rc)
    return Zb-Sag(D,rc)

'''def Blended_Bifocal(P,ins,N):
    E = 13.2
    Ri = np.sqrt((X-ins)**2 + (Y-E-5)**2)
    f = P*np.exp(-(Ri/E)**4)
    Zb = Sag(Ri,1000*(1-N)/(f+0.00000001))

    #delete from here on (Flat Top)
    #for i in range(171):
        #for j in range(171):
            #if Y[i,j] < 4:
                #Zb[i,j] = 0
    #to here

    return Zb'''
           
def Decenter_Surface(xdec,ydec):
    x = np.arange(-42.5,43.0,0.5)
    y = np.arange(-42.5,43.0,0.5)
    Y,X = np.meshgrid(x,y)
    X -= xdec
    Y -= ydec
    R = np.sqrt(X**2 + Y**2)
    return X,Y,R      

def Cross_Cylinder(Sphere, Cylinder, Axis,Thetau):
    Axis = np.radians(Axis)
    P = Sphere + Cylinder*np.sin(Thetau - Axis)*np.sin(Thetau-Axis)
    return P    

def Index_Power(Pf,Pb,Pt,Ct):
    n = -Ct*Pf*Pb/(Pt - Pf - Pb)
    return n

def Prism(Prism,Angle,Index,z,X,Y,Lent=0,Elabel='R'):
    #Anything x is actually y 
    A= np.radians(Angle)
    PrismX = Prism*np.sin(A) + 0.0000001
    PrismY = Prism*np.cos(A)*(-1) + 0.0000001
    Ax = (np.arctan(PrismX/100))/(Index-1)
    Ay = (np.arctan(PrismY/100))/(Index-1)
    t = (Y - np.sin(Ax)*z)/np.cos(Ax)
    s = (X + np.sin(Ay)*np.sin(Ax)*t - np.cos(Ax)*np.sin(Ay)*z)/np.cos(Ay)
    Z = -s*np.sin(Ay) + np.cos(Ay)*(-np.sin(Ax)*t + np.cos(Ax)*z)
    if Lent==1:
        R = np.sqrt(X**2 + Y**2)
        Z = Z*(0.5*((1/(1+np.exp(-(R+25))))-(1/(1+np.exp(-(R-25))))) + 0.5)
    return Z

def Prism_Induced(dZ,B,Index):
    P = (dZ/B)*(100*(Index-1))
    return P

def Radius_Power(n,p):
    c = abs(1000*(n-1)/p)
    return c

def Radius_Sag(r,z):
    c = (r**2 + z**2)/(2*z)
    return c

def SDF_Reader(s):
    f = open(s,'r')
    L = f.readlines()
    f.close()
    ZR = np.array(L[3:174])
    ZT = R-R
    for i in range(171):
        Zt = ZR[i]
        Zt = Zt[3:-2]
        m = 0
        n = '' 
        for j in Zt: 
            try:
                int(j)
                n +=j
            except ValueError:
                if j == '.':
                    n +=j
                else:
                    ZT[i,m] = float(n)
                    n = ''
                    m += 1
    return ZT
    
def Sag(r,c):
    Z = (r**2)/(c * (1 + np.sqrt(1-((r/c)**2))))
    return Z
    
def Slab(Z,PrismY,Index,h):
    #r = np.arange(-42.5,43.0,0.5)
    Ay = (np.arctan(PrismY/100))/(Index-1)
    for i in range(171):
        for j in range(171):
            yj = Y[i,j]
            if yj > h:
                Z[i,j] += (yj-h)*np.tan(Ay)
        #Coeff = np.polyfit(r,Z[i],5)
        #Z[i] = np.polyval(Coeff,r) #<-This blends in y direction (up & down)
    return Z
    
def Thick_Lens(Index,CT,Power,RF):
    Radius = ((Index/((Index*RF) - CT*(Index-1))) - (Power/(1000*(Index-1))))**(-1)
    return Radius

def Wrap_Tilt(Rx,Eye):
    S,C,A,W,N,T = Rx['SPH'],Rx['CYL'],Rx['AX'],Rx['PANTO'],Rx['LIND'],Rx['WRAP']
    '''T was mistakenly put
    as the wrap and W as the Tilt'''
    T,W,A = np.radians(T),np.radians(W),np.radians(A)
    phi = np.arctan(np.sqrt(np.sin(T)*np.sin(T) + np.tan(W)*np.tan(W))/np.cos(T))
    Tc = (2*N + np.sin(phi)*np.sin(phi))/(2*N*np.cos(phi)*np.cos(phi))
    Sc = 1 + ((np.sin(phi)*np.sin(phi))/(2*N))
    Hc = (Tc + Sc)/2
    App = np.arctan(np.tan(W)/np.sin(T))
    if Eye == 'L':
        App = (np.pi/2.0) - App
    A = A - App
    Px = S + C*np.sin(A)*np.sin(A)
    Py = S + C*np.cos(A)*np.cos(A)
    Pxy = -C*np.sin(A)*np.cos(A)
    P = np.matrix([[Px/Tc,Pxy/Hc],[Pxy/Hc,Py/Sc]])
    if Eye == 'L':
        P = np.matrix([[Px/Sc,Pxy/Hc],[Pxy/Hc,Py/Tc]])
    Tr = (P[0,0]+P[1,1])
    Cc = -np.sqrt((Tr**2)-4*np.linalg.det(P))
    Sc = 0.5*(Tr - Cc)
    Ac = np.arctan((Sc-P[0,0])/(P[0,1]+0.0000001))
    App = np.arctan(np.sin(W)/np.tan(T))
    if Eye == 'L':
        App = (np.pi/2.0)  - App
    At = np.degrees(Ac + App)
    if At<0:
        At += 180
    Rx['SPH'],Rx['CYL'],Rx['AX'] = Sc, Cc, At
    return Rx

def Zproj(m,n):
    s = 0
    for i in range(2):
        s += m[i]*n[i]
    s = s/0.5
    L = []
    for i in range(2):
        L.append(n[i]*s)
    r = np.sqrt(L[0]**2 + L[1]**2)
    return r
     
'========================'
'======[Job Object]======'
'========================'
#This is script that plays around with the files
class Job(str):
    
    '===File Methods==='
    def __init__(self,str):   
        self.JobN = str
        self.log = ''
        self.RightRx=[]
        self.LeftRx=[]
        #initally this is all empty but when an lds comes
        #in ill put something to fill this up
        #log is whats going into the LMS
        
    def __impLDS__(self,d):
        F = open(d + str(self.JobN) + '.LDS','r')
        S = F.readlines()
        for i in S:            
            try:
                m = i.index('=')
                n = i.index(';')
                o = len(i)
                self.RightRx.append((i[:m],float(i[m+1:n])))
                self.LeftRx.append((i[:m],float(i[n+1:o])))
            except ValueError:
                if i[:3] == 'JOB':
                    self.RightRx.append((i[:3],str(i[4:])))
                    self.LeftRx.append((i[:3],str(i[4:])))
                if i[:3] == 'DBL':
                    self.RightRx.append((i[:3],float(i[4:])))
                    self.LeftRx.append((i[:3],float(i[4:])))
                if i[:3] == 'PTO':
                    self.RightRx.append((i[:4],float(i[5:])))
                    self.LeftRx.append((i[:4],float(i[5:])))
                if i.split('=')[0] == 'LNSEN':
                    data = i.split('=')[-1][:-1]
                    self.RightRx.append(('LNSEN',data))
                    self.LeftRx.append(('LNSEN',data))
        self.RightRx = dict(self.RightRx)
        self.LeftRx = dict(self.LeftRx)
        F.close()
        #Here i import everything in the LDS as parameters in python
        #which allow me to play with it hence importlds
    
    def __expLMS__(self):
        pass
       
    def __expLOG__(self,d):
        f = open(d + str(self.JobN)+'.LMS','w')
        f.write(self.log)
        f.close()
        
    def __expXYZ__(self,d):
        f = open(d+'R'+ str(self.JobN)+'.XYZ', 'w+')
        for i in range(171):
            for j in range(171):
                f.write(str(X[i,j]) + '00000, ' + str(Y[i,j]) + '00000, ' + str(round(self.ZR[i,j],7)) + '\n')
        f.close()
        f = open(d + 'L' + str(self.JobN)+'.XYZ', 'w+')
        for i in range(171):
            for j in range(171):
                f.write(str(X[i,j]) + '00000, ' + str(Y[i,j]) + '00000, ' + str(round(self.ZL[i,j],7)) + '\n')     
        f.close()
    #more functions that export my lens into XYZ LMS.... ill have script later on thatll sort these files
    #right now it plays in the same directory but we dont want that 

def bethk(ctk,r,c,c2):
    r1,r2 = float(r[0])/2.0,float(r[1])/2.0
    fc1,fc2 = 530/float(c[0]),530/float(c[1])
    bc1,bc2 = 530/float(c2[0]),530/float(c2[1])
    dZ1 = float(ctk[0]) - (r1**2)/(fc1 * (1 + np.sqrt(1-((r1/fc1)**2)))) - (r1**2)/(bc1 * (1 + np.sqrt(1-((r1/bc1)**2))))
    dZ2 = float(ctk[1]) - (r2**2)/(fc2 * (1 + np.sqrt(1-((r2/fc2)**2)))) - (r2**2)/(bc2 * (1 + np.sqrt(1-((r2/bc2)**2))))
    return dZ1,dZ2   
        
def readlms(name):
    global JLOG
    with open(JLOG + name + '.lms','r') as f:
       data = f.read().split('\n') 
       dfound = False
       LMSDict = {}
       LMSDict['AXIS'] = []
       for i,d in enumerate(data):
           if dfound:
               dentry = d.split('\t')
               inf = []
               for de in dentry[1:]:
                   if de != '':
                       inf += [de]
               LMSDict[dentry[0]] = inf
           if 'Parameter' in d:
               dfound = True
           if '=' in d:
               dentry = d.split('=')
               LMSDict[dentry[0]] = dentry[1].split(';')
           if 'Wrap_Tilt' in d:
               LMSDict['AXIS'] += [d.split('x')[1]]

       BEthick = bethk(LMSDict['GTHK'],LMSDict['DIA'],LMSDict['FRNT'],LMSDict['BACK'])
       LMSDict['BETHK'] =  [str(round(BEthick[0],2)),str(round(BEthick[1],2))]
       
       lnsm = dict.fromkeys(['1.56','1.545','1.558'],6)
       lnsm.update(dict.fromkeys(['1.60','1.61'],11))
       lnsm.update(dict.fromkeys(['1.498','1.5'],1))
       lnsm.update(dict.fromkeys(['1.586','1.598'],10))
       lnsm['1.67'] = 8
       try:
           if LMSDict['LIND'][0] in lnsm or LMSDict['LIND'][1] in lnsm:
                LMSDict['LNSM'] = [lnsm[LMSDict['LIND'][0]],lnsm[LMSDict['LIND'][1]]] #lensmaterial
       except KeyError:
           print('Material not in table!')
           LMSDict['LNSM'] = [2,2]
       bhr = (9 - (530/float(LMSDict['FRNT'][0]) - (np.sqrt((530/float(LMSDict['FRNT'][0]))**2 - ((58/2)**2)))))
       bhl = (9 - (530/float(LMSDict['FRNT'][1]) - (np.sqrt((530/float(LMSDict['FRNT'][1]))**2 - ((58/2)**2)))))
       LMSDict['BHGHT'] = [round(bhr,13),round(bhl,13)]
    return LMSDict

Lblist = 'LNSM,LAPBASX,LAPCRSX,AXIS,GTHK,PRVM,PRVA,CRIB,FRNT,BCTHK,DIA,BETHK,BHGHT,DBL,HBOX,VBOX'.split(',')
def orfile(name): 
    global MAIN
    global ORF
    global OR2
    global Lblist
    
    smple = 'sample.or5'      
    if OR2:
        smple = 'sample.or2'  
    with open(MAIN+'/' + smple,'r') as f:
        form = f.read().split('\n')
        LMSDatum = readlms(name)
        if float(LMSDatum['CRIB'][0])<68:
            LMSDatum['CRIB'][0] = '68'
        if float(LMSDatum['CRIB'][1])<68:
            LMSDatum['CRIB'][1] = '68'
        orname = name + '.or5'
        if OR2:
            orname = name + '.or2'
            LMSDatum['PRVM'][0] = str(float(LMSDatum['PRVM'][0])+0.01)
            LMSDatum['PRVM'][1] = str(float(LMSDatum['PRVM'][1])+0.01)
        fillform = []
        for i,d in enumerate(form):
            if 1<=i<12:
                dat = LMSDatum[Lblist[i-1]]
                if OR2:
                    if i==1:
                        dat = LMSDatum['LNSM']
                    if i==2 or i==3:
                        dat = [str(-float(LMSDatum[Lblist[i-1]][0])),str(-float(LMSDatum[Lblist[i-1]][1]))]
                if not OR2:
                    if i==1:
                        dat = ['1','0']
                    if i==2:
                        dat = ['1','1']
                    if i==3:
                        dat = LMSDatum['LNSM']
                    if i==4 or i == 6 or i==7:
                        dat = ['0','0']
                for l in range(2):
                    dat[l] = float(dat[l])
                    if int(dat[l])==dat[l]:
                        dat[l] = str(int(dat[l]))
                    else:
                        dat[l] = str(round(dat[l],2))
                frmtdata = ''
                for k in range(2):
                    for j in range(13):
                        if j >= len(dat[k]):
                            frmtdata += ' '
                        else:
                            frmtdata += dat[k][j]
                fillform += [d.split('=')[0] + '=  ' + frmtdata[:-1]]
            else:
                temp = [d]
                if 'block_height(0)' in d:
                    temp = [d.split('=')[0] + '=  ' + str(LMSDatum['BHGHT'][0])]
                if 'block_height(1)' in d:
                    temp = [d.split('=')[0] + '=  ' + str(LMSDatum['BHGHT'][1])]
                if 'DBL(0)' in d:
                    temp = [d.split('=')[0] + '=  ' + LMSDatum['DBL'][0]]
                if 'HBOX(0)' in d:
                    temp = [d.split('=')[0] + '=  ' + LMSDatum['HBOX'][0]]
                if 'HBOX(1)' in d:
                    temp = [d.split('=')[0] + '=  ' + LMSDatum['HBOX'][1]]
                if 'VBOX(0)' in d:
                    temp = [d.split('=')[0] + '=  ' + LMSDatum['VBOX'][0]]
                if 'VBOX(1)' in d:
                    temp = [d.split('=')[0] + '=  ' + LMSDatum['VBOX'][1]]
                if 'pf_name(0)' in d:
                    temp = [d.split('=')[0] + '=  "R' + name+'.xyz' + '"']
                if 'pf_name(1)' in d:
                    temp = [d.split('=')[0] + '=  "L' + name+'.xyz' + '"']
                fillform += temp
        with open(ORF + orname,'w') as g:
            for txt in fillform:
                g.write(txt+'\n')
        f.close()
        
        
'================================='
'======[Heart of the Design]======'
'================================='    
def Make_Lens(Rx,Prog,Eye,Lent,Ozone,CT):
    '''Rx is dictionary import by LDS
       Prog is the Mold file location
       Eye = R for right and L for left
       Lent is the lenticular number
       Ozone is the size of the optical zone
       and finally CT is the final thickness'''
    global PRCorr,PLCorr
    x = np.arange(-42.5,43.0,0.5)
    Y,X = np.meshgrid(x,x)
    Xw = X
    Yw = Y
    R = np.sqrt(Xw**2 + Yw**2)
    Thetai = np.arctan(Yw/(-Xw+0.00000001))
    RFC = Radius_Power(Rx['TIND'],Rx['FRNT'])
    RBC = Radius_Power(Rx['TIND'],Rx['BACK'])
    P = Cross_Cylinder(Rx['SPH'],Rx['CYL'],Rx['AX'],Thetai)
    RC = Thick_Lens(Rx['LIND'],CT,P,RFC)

    #blank high power runoff solution
    def rfix(R,rc):
        if rc <65:
            rbint = (R<=(rc-5)).astype(int)
            return R*rbint + (rc-5)*(rbint*(-1) + (R*0+1))
        return R

    ZBb = Sag(rfix(R,RBC), RBC)
    Zf = Sag(rfix(R,RFC), RFC)
    if Rx['LENT'] == 1: # and abs(Rx['CYL']) < 2.00:
        RFCt = Radius_Power(Rx['TIND'],0.01)
        RCs = Thick_Lens(Rx['LIND'],CT,Rx['SPH'],RFCt)
        Za = Aspheric(R,RCs,-abs(R/Ozone)**3)
        if Rx['CYL'] < 0:
            Pc = Cross_Cylinder(0,Rx['CYL'],Rx['AX'],Thetai)
            RCc = Thick_Lens(Rx['LIND'],CT,Pc,RFCt)
            Zc = Aspheric(R,RCc,-abs(R/Ozone)**3)
        else:
            Zc = 0
        ZT = Zf + Za + Zc
        #Debugging the Aspheric
        '''
        RC = Thick_Lens(Rx['LIND'],CT,-Rx['SPH'],RFC)
        Pc = 1000*(Rx['TIND']-1)/RC - 1000*(Rx['TIND']-1)/Radius_Sag(10,ZT[85,125])
        RCc = Radius_Power(Rx['TIND'],Pc)
        Zr = Sag(R,RCc)
        ZT = Zf + Zr + Zc   
        Rf = Radius_Sag(5,Zf[85,75])/1000
        Rb = Radius_Sag(5,ZT[85,75])/1000
        print (Rx['LIND'] - 1)*(1/Rf - 1/Rb  + (Rx['LIND']-1)*(CT)/(Rx['LIND'])*Rf*Rb)'''
        
    '''if Rx['LENT'] == 1 and abs(Rx['CYL']) >= 2.00:
        ZT = Aspheric(R,RC,-abs(R/Ozone)**3)'''
    if Rx['LENT'] == 0 or Rx['LENT'] == -1:
        ZT = Sag(R,RC)
    
    #Dimple fix
    if Rx['BCOCIN'] ==0 and Rx['BCOCUP'] ==0:
        ZT[85,85] = 0.0
    
    #Decenter the SV
    if Eye == 'R':
        Pin = (Rx['BCOCIN']/20.0)*P[85,90]
        Pup = (Rx['BCOCUP']/20.0)*P[90,85]
        PRCorr = Pin, Pup
        ZT = Prism(Pin,0,Rx['LIND'],ZT,X,Y)
        ZT = Prism(Pup,90,Rx['LIND'],ZT,X,Y)
        
    if Eye == 'L':
        Pin = -(Rx['BCOCIN']/20.0)*P[85,90]
        Pup = (Rx['BCOCUP']/20.0)*P[90,85]
        PLCorr = Pin, Pup
        ZT = Prism(Pin,0,Rx['LIND'],ZT,X,Y)
        ZT = Prism(Pup,90,Rx['LIND'],ZT,X,Y)
        
    #Progressive    
    if abs(Rx['ADD']) > 0 and Rx['ROUND'] == 0:
        if Rx['LNSEN'] =='VRX':
            Prog = MOLD+'VariMold//' + str(float(Rx['ADD']+0.25)) + '.XYZ'
        if Rx['ADD'] > 0.75:
        	ZTg = Prism(100*np.tan(0.0227*(Rx['LIND']-1)),270,Rx['LIND'],ZT,X,Y)
        	ZT = ZTg
        try:          
            F = open(Prog,'r')
            S = F.readlines()
            ZO = []
            ZF = Sag(R,88.333)
            if Eye == 'R':
                for i in range(171):
                    Zt = []
                    for j in range(171):
                        s = 0
                        Z = ''
                        for k in S[171*j + i]:
                            if s == 2:
                                Z += k
                            if k == ',':
                                s +=1
                        Zt.append(float(Z))
                    if Prog[-5] == '1':  
                        ZO.append(Zt)
                    else:  
                        ZO.append(list(reversed(Zt)))
                if Prog[-5] == '1':
                    ZO = np.array(list(reversed(ZO)))
                else:
                    ZO = np.array(list(reversed(ZO))) - ZF
                
                #Decenter the mold
                ZO = list(ZO)
                ZA = []
                for i in range(171):
                    if Rx['BCOCIN'] >0:	
                        if i <= abs(Rx['BCOCIN']):
                            ZA.append(ZO[0])
                        else:
                            ZA.append(ZO[i-int(Rx['BCOCIN'])])
                    if Rx['BCOCIN'] <= 0:	
                        if i >= 170-abs(Rx['BCOCIN']):
                            ZA.append(ZO[170])
                        else:
                            ZA.append(ZO[i+abs(int(Rx['BCOCIN']))])
                ZO = np.array(ZA)
                
                if Rx['LNSEN'] =='VRX':
                    ZO = ((.502)/(Rx['LIND']-1))*ZO

                if Rx['ADD'] < 0:
                    ZT -= ZO
                if Rx['ADD'] > 0:
                    ZT += ZO

            if Eye == 'L':
                for i in range(171):
                    Zt = []
                    for j in range(171):
                        s = 0
                        Z = ''
                        for k in S[171*j + i]:
                            if s == 2:
                                Z += k
                            if k == ',':
                                s +=1
                        Zt.append(float(Z))
                    if Prog[-5] == '1':  
                        ZO.append(Zt)
                    else:  
                        ZO.append(list(reversed(Zt)))
                if Prog[-5] == '1':
                    ZO = np.array(ZO) 
                else:
                    ZO = np.array(list(reversed(ZO))) - ZF
                    
                #Decenter the mold
                ZO = list(reversed(list(ZO)))
                ZA = []
                for i in range(171):
                    if Rx['BCOCIN'] <0:	
                        if i <= abs(Rx['BCOCIN']):
                            ZA.append(ZO[0])
                        else:
                            ZA.append(ZO[i-abs(int(Rx['BCOCIN']))])
                    if Rx['BCOCIN'] >=0:	
                        if i >= 170-abs(Rx['BCOCIN']):
                            ZA.append(ZO[170])
                        else:
                            ZA.append(ZO[i+int(Rx['BCOCIN'])])
                ZO = np.array(ZA)

                if Rx['LNSEN'] =='VRX':
                    ZO = ((.502)/(Rx['LIND']-1))*ZO

                if Rx['ADD'] < 0:
                    ZT -= ZO
                if Rx['ADD'] > 0:
                    ZT += ZO           
                
        except IOError:
            pass

    #Round        
    if Rx['ROUND'] != 0: 
        if Eye == 'L':
            x0 = 1.5 + 0.5*Rx['BCOCIN']
        if Eye == 'R':
            x0 = -1.5 - 0.5*Rx['BCOCIN']
        if Rx['ROUND'] ==1:
            for i in range(5):
                Zbt = RoundBi((1/5.0)*Rx['ADD'],-x0,Rx['LIND'],17.5,12.0 + i*(1.0/4.0))
                ZT += Zbt 
        if Rx['ROUND'] ==2:
            for i in range(10):
                Zbt = RoundBi((1/10.0)*Rx['ADD'],-x0,Rx['LIND'],19.00,13.0 + i*(2.0/9.0))
                ZT += Zbt 
        if Rx['ROUND'] ==3:
            Zbt = RoundBi((0.5/2.0)*Rx['ADD'],-x0,Rx['LIND'],20.00,11.00)
            ZT += Zbt 
       	    Zbt = RoundBi((0.5/2.0)*Rx['ADD'],-x0,Rx['LIND'],20.00,11.50)
       	    ZT += Zbt 
       	    Zbt = RoundBi((0.5/3.0)*Rx['ADD'],-x0,Rx['LIND'],16.50,14.50)   
       	    ZT += Zbt 
       	    Zbt = RoundBi((0.5/3.0)*Rx['ADD'],-x0,Rx['LIND'],16.50,14.75)   
       	    ZT += Zbt 
            Zbt = RoundBi((0.5/3.0)*Rx['ADD'],-x0,Rx['LIND'],16.50,15.00)   
            ZT += Zbt 
        if Rx['ROUND'] > 15:
            Bz = Rx['PAL_W']
            Wi = Rx['ROUND']
            Inc = int(Bz/0.25) 
            for i in range(Inc):
                Zbt = RoundBi((1.0/Inc)*Rx['ADD'],-x0,Rx['LIND'],Wi + Bz/2.0 + 5.0,Wi + i*(Bz/(Inc-1)))
                ZT += Zbt
    
    if Rx['PAL_W'] == 1:
        if Eye == 'L':
            ZT[int(85 - Rx['BCOCIN'] - 34),int(85-Rx['BCOCUP'] + 3)] =0.999*ZT[int(85 - Rx['BCOCIN'] - 34),int(85-Rx['BCOCUP'] + 3)]
            ZT[int(85 - Rx['BCOCIN'] - 34),int(85-Rx['BCOCUP'] - 3)] =0.999*ZT[int(85 - Rx['BCOCIN'] - 34),int(85-Rx['BCOCUP'] + 3)]
            ZT[int(85 - Rx['BCOCIN'] + 34),int(85-Rx['BCOCUP'])] =0.999*ZT[int(85 - Rx['BCOCIN'] + 34),int(85-Rx['BCOCUP'])]
            ZT[int(85 - Rx['BCOCIN'] - 34),int(85-Rx['BCOCUP'])] =0.999*ZT[int(85 - Rx['BCOCIN'] - 34),int(85-Rx['BCOCUP'])]
            ZT[int(85 - Rx['BCOCIN'] - 50),int(85-Rx['BCOCUP'] + 3)] =0.999*ZT[int(85 - Rx['BCOCIN'] - 50),int(85-Rx['BCOCUP'] + 3)]
            ZT[int(85 - Rx['BCOCIN'] + 50),int(85-Rx['BCOCUP'])] =0.999*ZT[int(85 - Rx['BCOCIN'] + 50),int(85-Rx['BCOCUP'])]
            ZT[int(85 - Rx['BCOCIN'] - 50),int(85-Rx['BCOCUP'])] =0.999*ZT[int(85 - Rx['BCOCIN'] - 50),int(85-Rx['BCOCUP'])]
            ZT[int(85 - Rx['BCOCIN'] - 50),int(85-Rx['BCOCUP']+6)] =0.999*ZT[int(85 - Rx['BCOCIN'] - 50),int(85-Rx['BCOCUP']+6)]
            #ZT[int(85 - Rx['BCOCIN']),int(85+Rx['BCOCUP']) - 57] =0.999*
            if abs(Rx['SPH']) > 5:
                ZT[int(85 - Rx['BCOCIN'] - 34),int(85-Rx['BCOCUP'] + 6)] =0.999*ZT[int(85 - Rx['BCOCIN'] - 34),int(85+Rx['BCOCUP'] + 6)]
                    
        if Eye == 'R':
            ZT[int(85 + Rx['BCOCIN'] + 34),int(85-Rx['BCOCUP'] + 3)] =0.999*ZT[int(85 + Rx['BCOCIN'] + 34),int(85-Rx['BCOCUP'] + 3)]
            ZT[int(85 + Rx['BCOCIN'] + 34),int(85-Rx['BCOCUP'] - 3)] =0.999*ZT[int(85 + Rx['BCOCIN'] + 34),int(85-Rx['BCOCUP'] + 3)]
            ZT[int(85 + Rx['BCOCIN'] - 34),int(85-Rx['BCOCUP'])] =0.999*ZT[int(85 + Rx['BCOCIN'] - 34),int(85-Rx['BCOCUP'])]
            ZT[int(85 + Rx['BCOCIN'] + 34),int(85-Rx['BCOCUP'])] =0.999*ZT[int(85 + Rx['BCOCIN'] + 34),int(85-Rx['BCOCUP'])]
            ZT[int(85 + Rx['BCOCIN'] + 50),int(85-Rx['BCOCUP'] + 3)] =0.999*ZT[int(85 + Rx['BCOCIN'] + 50),int(85-Rx['BCOCUP'] + 3)]
            ZT[int(85 + Rx['BCOCIN'] - 50),int(85-Rx['BCOCUP'])] =0.999*ZT[int(85 + Rx['BCOCIN'] - 50),int(85-Rx['BCOCUP'])]
            ZT[int(85 + Rx['BCOCIN'] + 50),int(85-Rx['BCOCUP'])] =0.999*ZT[int(85 + Rx['BCOCIN'] + 50),int(85-Rx['BCOCUP'])]
            ZT[int(85 + Rx['BCOCIN'] + 50),int(85-Rx['BCOCUP']+6)] =0.999*ZT[int(85 + Rx['BCOCIN'] + 50),int(85-Rx['BCOCUP']+6)]
            #ZT[int(85 + Rx['BCOCIN']),int(85+Rx['BCOCUP']) - 56] =0.999*
   	    #ZT[int(85 + Rx['BCOCIN']),int(85+Rx['BCOCUP']) - 57] =0.999*
            if abs(Rx['SPH']) > 5:
                ZT[int(85 + Rx['BCOCIN'] + 34),int(85-Rx['BCOCUP'] + 6)] =0.999*ZT[int(85 + Rx['BCOCIN'] + 34),int(85+Rx['BCOCUP'] + 6)]

    ZTt = Prism(Rx['PRVM'],Rx['PRVA'],Rx['LIND'],ZT,X,Y, Lent=Rx['LENT'],Elabel=Eye)
    #Ay = (np.arctan(PrismY/100))/(Index-1)
    ZT = ZTt
    
    return ZT,ZBb,Zf
        
'============================='
'==========[Tests]============'
'============================='
#Tests wether the design fits
#^^^speaks for itself
#Using the surface and the blank, it will
#test if the design "fits"
def Thickness_Test(Rx,Zf,ZBb,Zb,CT,Eye):
    THCKb = Rx['BCTHK'] - Zf + ZBb
    THCKa = CT - Zf + Zb
    T1,T2,xy = [],[],[]
    if Eye =='R':
        Rt = np.sqrt((X-Rx['XDEC'])**2 + (Y+Rx['YDEC'])**2)
    if Eye =='L':
        Rt = np.sqrt((X+Rx['XDEC'])**2 + (Y+Rx['YDEC'])**2)
    for i in range(171):
        for j in range(171):
            Ri = Rt[i,j]
            if Eye == 'R':
                Xi = X[i,j] - Rx['XDEC']
            if Eye == 'L':
                Xi = X[i,j] + Rx['XDEC']
            Yi = Y[i,j] + Rx['YDEC']
            Rn = np.sqrt(3**2 + 3**2)
            if Ri <= Rn:
                T2.append(THCKa[i,j])
                T1.append(THCKb[i,j])
                xy.append([i,j])
                if THCKa[i,j] < Rx['MINCTR']:
                    return 'Too Thin'
            if Ri <= (Rx['FED']/2.0) and -Rx['HBOX']/2.0<=Xi<=Rx['HBOX']/2.0 and -Rx['VBOX']/2.0<=Yi<=Rx['VBOX']/2.0 and Ri>Rn:
                T2.append(THCKa[i,j])
                T1.append(THCKb[i,j])
                xy.append([i,j])
                if THCKa[i,j] < Rx['MINEDG']-0.8:
                    return 'Too Thin'
            if R[i,j]<29:
                T2.append(THCKa[i,j])
                T1.append(THCKb[i,j])
                xy.append([i,j])
                if THCKa[i,j] < 0.2:
                    return 'Too Thin'
    mxTcka = max(T2)
    mxTckb = max(T1)
    mnTcka = min(T2)
    if mxTckb-1.0 < mxTcka:
        return 'Not enough meat'
    else:
        return [xy[T2.index(mxTcka)],mxTcka,mnTcka]


def SliceTk(tuple):
    global grphscreen
    Eye,Rx,Z,ZBb,ZF,nCT = tuple
    A = Rx['HBOX']
    B = Rx['VBOX']
    ED = Rx['FED']
    dED = np.sqrt(Rx['XDEC']**2 + Rx['YDEC']**2)
    if Eye=='L':
        Eye='R'
    else:
        Eye='L'
    CT = Rx['BCTHK']
    x = np.arange(-42.5,43.0,0.5)
    XP = np.arange(-42.5,43.0,0.5)/2.0  

    if Eye=='L':
        grphscreen.create_text(100,30,text='R' + str(Rx['JOB']),tags='data')
    if Eye=='R':
        grphscreen.create_text(300,30,text='L' + str(Rx['JOB']),tags='data')

    'ygraphs'
    draw_layer(x,ZBb[85] + (CT-nCT) + 42.5,'BLUE',Eye)
    draw_layer(x,Z[85] + 42.5,'RED',Eye)
    draw_layer(x,ZF[85] - nCT +42.5,'BLUE',Eye)

    
    'XCRITICAL POINTSX'
    Thck = nCT - ZF + Z
    for i in range(171):
        for j in range(171):
            Thck[i,j] = round(Thck[i,j],3) 
                   
    TextL = [(-B/2.0 + Rx['YDEC'], 60, Thck[85,int(85 - B + 2*Rx['YDEC'])]),
    (B/2.0 + Rx['YDEC'], 60, Thck[85,int(85 + B + 2*Rx['YDEC'])]),
    (0,30,Thck[85,85])]

    if Eye == 'R':
        draw_layer(A/2.0 - Rx['XDEC'] + (x-x),XP-42.5, 'BLACK',Eye)
        draw_layer(-A/2.0 - Rx['XDEC'] + (x-x),XP-42.5, 'BLACK',Eye)
        TextL+=[(A/2.0 - Rx['XDEC'], -60, Thck[int(85 + A - 2*Rx['XDEC']),85])]
        TextL+=[(-A/2.0 - Rx['XDEC'],-60, Thck[int(85 - A - 2*Rx['XDEC']),85])]
    else:
        draw_layer( A/2.0 + Rx['XDEC'] + (x-x),XP-42.5, 'BLACK',Eye)
        draw_layer(-A/2.0 + Rx['XDEC'] + (x-x),XP-42.5, 'BLACK',Eye)
        TextL+=[(A/2.0 + Rx['XDEC'], -60, Thck[int(85 + A + 2*Rx['XDEC']),85])]
        TextL+=[(-A/2.0 + Rx['XDEC'],-60, Thck[int(85 - A + 2*Rx['XDEC']),85])]
        
    for t in TextL:
        if Eye=='L':
            grphscreen.create_text(pixelm(t[0],t[1])[0],pixelm(t[0],t[1])[1],text = str(t[2]), fill='RED',tags='data')
        if Eye=='R':
            grphscreen.create_text(pixelm(t[0],t[1])[0]+200,pixelm(t[0],t[1])[1],text = str(t[2]), fill='RED',tags='data')
    
    'LEGEND'
    Number = 0
    for i in Rx:
        if i=='PRSC' or i=='CRIB' or i=='OZONE' or i=='LIND' or i=='FRNT':
            if i=='FRNT':
                Rx[i] = round(Rx[i],2)
            if Rx[i] != 0.00 and i!= 'PRSC':
                t = (-5,10- Number*10,i + ': ' + str(Rx[i]))
                Number +=1
            if i == 'PRSC':
                t = (-5,10- Number*10,Rx[i])
                Number +=1
            if Eye=='L':
                grphscreen.create_text(pixelm(t[0],t[1])[0],pixelm(t[0],t[1])[1],text = str(t[2]),tags='data')
            if Eye=='R':
                grphscreen.create_text(pixelm(t[0],t[1])[0]+200,pixelm(t[0],t[1])[1],text = str(t[2]),tags='data')
                
    ZBb = np.transpose(ZBb)
    Z = np.transpose(Z)
    ZF = np.transpose(ZF)
    'xgraphs'
    draw_layer(-x,ZBb[85] + (CT-nCT)-42.5,'BLUE',Eye)
    draw_layer(x,Z[85] - 42.5,'RED',Eye)
    draw_layer(-x,ZF[85] - nCT -42.5, 'BLUE',Eye)
    'ylines'
    draw_layer( B/2.0 + Rx['YDEC'] + x-x,XP+42.5, 'BLACK',Eye)
    draw_layer(-B/2.0 + Rx['YDEC'] + x-x,XP+42.5, 'BLACK',Eye)
    return



#i figured out how to graph and keep the program running
#all you literally need to do is put so it keeps going...
#before it wouldnt show unless the program finished
#but now i can keep my plugin running and itll keep graphing
#ill make it show every graph the first time we try to implement it
#to better monitor the jobs and be able to fix my plug in easier
#eventually we can just turn them off


'---Smart Design---'
def Decenter_Box(J,Eye):
    if Eye =='R':
        J['XDEC'] = (J['HBOX']+J['DBL'])/2.0 - J['IPD']
        J['YDEC'] = 0
    if Eye =='L':
        J['XDEC'] = ((J['HBOX']+J['DBL'])/2.0 - J['IPD'])
        J['YDEC'] = 0
    if J['SEGHT'] != 0 and J['ADD'] != 0 and J['ROUND'] == 0:
        J['YDEC'] -= 3.0 + (J['VBOX'])/2.0 - J['SEGHT']
    if J['ROUND'] == 1 or J['ROUND'] ==2:
        J['YDEC'] -= (J['VBOX'])/2.0 - J['SEGHT'] - 5.0
        #J['YDEC'] = (J['VBOX'])/2.0 - J['SEGHT'] - 3.0
    if J['ROUND'] ==3:
        J['YDEC'] -= (J['VBOX'])/2.0 - J['SEGHT'] - 2.0
        
def Prism_Thinning(J):
    djr,djl = round(J.RightRx['VBOX']/2.0),round(J.LeftRx['VBOX']/2.0)
    r,l = int(J.RightRx['YDEC']*2), int(J.LeftRx['YDEC']*2)
    dZr,dZl = J.ZR[85+J.RightRx['BCOCIN'],85-djr+r] - J.ZR[85+J.RightRx['BCOCIN'],85+djr+r], J.ZL[85-J.LeftRx['BCOCIN'],85-djl+l] - J.ZL[85-J.LeftRx['BCOCIN'],85+djl+l]
    if abs(dZr) <= abs(dZl):
        dZ = dZl
        Theta = np.arctan(dZl/J.LeftRx['VBOX'])
        for i in range(170):
            for j in range(170):
                J.ZR[i,j] -= (85 - j + r)*0.5*np.tan(Theta)
                J.ZL[i,j] -= (85 - j + l)*0.5*np.tan(Theta)   
    if abs(dZl) < abs(dZr):
        dZ = dZr
        Theta = np.arctan(dZr/J.RightRx['VBOX'])
        for i in range(170):
            for j in range(170):
                J.ZR[i,j] -= (85 - j + r)*0.5*np.tan(Theta)
                J.ZL[i,j] -= (85 - j + l)*0.5*np.tan(Theta)  
    PRSM1 = Prism_Induced(dZ,J.RightRx['VBOX'],J.RightRx['LIND'])
    PRSM2 = Prism_Induced(dZ,J.LeftRx['VBOX'],J.LeftRx['LIND'])
 
    J.ZR -= J.ZR[85,85]
    J.ZL -= J.ZL[85,85]
    
    J.log += '\n\nPrism Thinning:' + str(round(PRSM1,2)) + ',' + str(round(PRSM2,2))
    return dZ
    
#Thickness optimization
def Think(Lens,Rx,Eye):
    '''
    Makes every possible lens and returns
    the thinnest design
    
    returns the thickest point, the CT, and Z
    and logs everything in object log
    '''
    
    #Plus cyl conversion
    if Rx['CYL'] > 0:
        Rx['SPH'] += Rx['CYL']
        Rx['CYL'] = (-1)*Rx['CYL']
        if Rx['AX'] > 90:
            Ctemp = -90 
        if Rx['AX'] <= 90:
            Ctemp = 90
        Rx['AX'] += Ctemp
    
    #LDS Corrections
    TempDec = (Rx['HBOX'] + Rx['DBL'])/2.0 - Rx['IPD']
    Rx['MBS'] = 2*abs(TempDec) + Rx['FED']
    Rx['CRIB'] = Rx['MBS'] + 4 - 2*abs(Rx['BCOCIN'])
    if Rx['CRIB'] > Rx['DIA']:
        Rx['CRIB'] = Rx['DIA']
        if TempDec == 0:
            Rx['BCOCIN'] = -((Rx['CRIB']-4-Rx['MBS'])/2.0)
        else:
            Rx['BCOCIN'] = -((Rx['CRIB']-4-Rx['MBS'])/2.0)*(TempDec/abs(TempDec))
        print('WARNING!!! BCOCIN is now ' + str(Rx['BCOCIN'])+' !!! WARNING')
    if Rx['FED'] < Rx['HBOX']:
        print('FED smaller than HBOX')
        return 'No Design Found'
    Rx['FRNT'] += 0.01

    Rx['BCOCIN'] = 2*Rx['BCOCIN']
    Rx['BCOCUP'] = 2*Rx['BCOCUP']
    
    #Aspheric Lens Parameters
    CTL = np.arange(Rx['MINCTR'],Rx['BCTHK'],0.25)
    Ozone = (Rx['MBS'] - 8 - abs(Rx['SPH']))/2.0
    '''if abs(Rx['SPH']) > 4.00 or abs(Rx['CYL']) > 4.00:
        if Rx['LENT'] != -1:
            Rx['LENT'] = 1'''
    
    if Rx['WRAP'] > 0 or Rx['PANTO']>0:
        if abs(Rx['SPH'])>4:
            Rx = Wrap_Tilt(Rx,Eye)
            
    ZLIST = []
    Thn = []
    THCK = []
    THCKl = []
    CTl = []
    if Rx['OZONE'] ==0:
    	OZl = [Ozone,22]
    else:
        OZl = [Rx['OZONE']/2.0, Ozone, 22]
    LInfo = []
    
    #Desides what mold to use
    #if Rx['SEGHT'] <= 24:
    CORR ='MC15'
    if Rx['LNSEN'] == 'MC17':
        CORR ='MC17'
    if Rx['LNSEN'] == 'MC18':
        CORR ='MC18'
    if abs(Rx['ADD']) > 0 and Rx['ROUND'] == 0:
        if Rx['LIND'] == 1.74 or Rx['LIND'] == 1.740:
            IND = str(1.670)
            ADD = str(Add_Round(abs(Rx['ADD'])))
        else:
            IND = str(Rx['LIND'])
            ADD = str(abs(Rx['ADD']))
        while len(IND)<5:
            IND +='0'
        while len(ADD)<4:
            if len(ADD)==1:
                ADD +='.'
            else:
                ADD +='0'
        PRGMLD = MOLD + CORR +'_'+ IND +'_' + ADD + '.XYZ'
        if os.path.exists(PRGMLD):
            pass
        else:
            print('Mold does not exist')
            return 'No Design Found'
    else: 
        PRGMLD = ''

    #Starts to think
    Found = 0
    print('Preliminary Testing\n')
    for t in OZl:
        for i in CTL:
            ZT,ZBb,Zf = Make_Lens(Rx,PRGMLD,Eye,t+2,t,i)
            T = Thickness_Test(Rx,Zf,ZBb,ZT,i,Eye) 
            if T=='Too Thin' or T=='Not enough meat':
                pass
            else:
                if Rx['LENT'] == 0:
                    t = -2
                ZLIST.append(ZT)
                Thn.append(T[2])
                THCK.append(T[1])
                THCKl.append(T[0])
                CTl.append(i)
                LInfo.append([t+2,t])
                Found+=1
                break
        if Found != 0:
            break
               
    #Checks back powers before thinking
    CT = 2.00
    RFC = Radius_Power(Rx['TIND'],Rx['FRNT'])
    RBC = Radius_Power(Rx['TIND'],Rx['BACK'])
    P = Cross_Cylinder(Rx['SPH'],Rx['CYL'],Rx['AX'],Theta)     
    RCi = Thick_Lens(Rx['LIND'],CT,P,RFC)
    RCi = 1000*(Rx['PIND']-1)/RCi
    
    #Polisher Colors
    P = RCi
    PList = list(P[0])
    for i in range(171):
        PList.append(P[i,170])
    Pback = max(PList)
    if Pback < 2.5:
        Rx['PCLR'] = 'GREY'
    if 2.5 <= Pback < 5.0:
        Rx['PCLR'] = 'RED'
    if Pback >= 5.0:
        Rx['PCLR'] = 'BLUE' 
        
    RC = []
    for i in RCi:
        for j in i:
            RC.append(j)
    
    if max(RC)>7.00 or min(RC)<2.00:
        Lens.log += '\nNot Recommended Blank'  
                                                                                           
    #Chosing the thinnest design
    if ZLIST == []: 
        Lens.log += '\nNo Design Found' 
        return 'No Design Found'          
    else:
        Rf = Radius_Sag(5,Zf[85,75])/1000
        Rb = Radius_Sag(5,ZT[85,75])/1000
        Rf = Radius_Sag(5,Zf[75,85])/1000
        Rb = Radius_Sag(5,ZT[75,85])/1000
        f  = THCK.index(min(THCK))          
        ZT = ZLIST[f]
        THCK = THCK[f]
        THCKl = THCKl[f]
        Thn = Thn[f]
        CT = CTl[f]
        LInfo = LInfo[f]
        LInfo[1] = round(LInfo[1],2)

        #Checks the Z-Array
        for i in ZT:
            for j in i:
                if j > 0 or j < 0 or j ==0:
                    pass        
                else:
                    print('NAN Found in Z-Array')
                    return 'No Design Found'
        
        #Assigns thickness to object so i can pass it to thinning
        if Rx['MBS']+4 < 65:
            Lens.log +='\n!!!CRIB is less than 65mm!!! (Changed to 65)'
            Rx['MBS'] = 61
            Rx['CRIB'] = 65
        if Eye == 'R':
            Lens.rCT = CT
            Lens.rZF = Sag(R,RFC)
            Lens.rZB = Sag(R,RBC)
            Lens.rP = RC
            Lens.rPRX = P
            Lens.rASX = min(RC)
            Lens.rSPH = max(RC)
            Lens.rCrib = Rx['CRIB']
        if Eye == 'L':
            Lens.lCT = CT
            Lens.lZF = Sag(R,RFC)
            Lens.lZB = Sag(R,RBC)
            Lens.lP = RC
            Lens.lPRX = P
            Lens.lASX = min(RC)
            Lens.lSPH = max(RC)     
            Lens.lCrib = Rx['CRIB']
        #Logs everything that happened in here
        if Rx['MBS']+4 > Rx['DIA']:
            Lens.log += '\n!!!CRIB is greater than Lens Diameter!!!'
        if Rx['WRAP'] > 0 or Rx['PANTO']>0:   
            Lens.log += '\nWrap_Tilt:' + str(round(Rx['SPH'],2)) +' '+ str(round(Rx['CYL'],2)) + 'x' + str(round(Rx['AX'],2))
            Rx['PRSC'] = str(round(Rx['SPH'],2)) +'  '+ str(round(Rx['CYL'],2)) + 'x' + str(round(Rx['AX'],2)) +'  '+ str(Rx['ADD'])
        Lens.log += '\nCross Curves:' + str(round(max(RC),2)) + '/' +  str(round(min(RC),2))
        if Rx['LENT'] !=0:
            Lens.log += '\nOzone: ' + str(LInfo[1]*2)
            Rx['OZONE'] = LInfo[1]*2
        if PRGMLD[len(MOLD):] != '':
            Lens.log += '\nMold:' + PRGMLD[len(MOLD):] 
        
        Rx['BCOCIN'] = Rx['BCOCIN']/2.0
        Rx['BCOCUP'] = Rx['BCOCUP']/2.0
        return ZT
    



'--------COMPUTATIONS--------'
def prl2vect(tup):
    Rx,Lx,Ry,Ly = tup
    Rx,Lx,Ry,Ly = float(Rx)+0.01,float(Lx)+0.01,float(Ry),float(Ly)
    Rm,Lm = np.sqrt(Ry**2 + Rx**2),np.sqrt(Ly**2 + Lx**2)
    Ra,La = np.degrees(np.arctan(abs(Ry/Rx))),np.degrees(np.arctan(abs(Ly/Lx)))
    if Rx > 0:
        Ra = 180-Ra
    if Lx < 0:
        La = 180-La
    if Ry < 0:
        Ra  = 360 - Ra
    if Ly < 0:
        La = 360 - La
    if Rx==0.01 and Ry==0:
        Rm,Ra = 0,0
    if Lx==0.01 and Ly==0:
        Lm,La = 0,0
    return [[str(round(Rm,2)),str(round(Lm,2))],[str(int(Ra)),str(int(La))]]
        
    
def prismvalue(addp=False):
    global JobData
    global prsmval
    global PRCorr,PLCorr
    Ry = float(prsmval[0][0].get())
    if prsmval[0][1].get() == 'DOWN':
        Ry = -1*Ry
        
    Rx = float(prsmval[1][0].get())+0.00000001
    if prsmval[1][1].get() == 'OUT':
        Rx = -1*Rx
    
    Ly = float(prsmval[2][0].get())
    if prsmval[2][1].get() == 'DOWN':
        Ly = -1*Ly
    
    Lx = float(prsmval[3][0].get())+0.0000001
    if prsmval[3][1].get() == 'IN':
        Lx = -1*Lx
        
    if addp:
        Rx += PRCorr[0]
        Ry += -PRCorr[1] 
        Lx += PLCorr[0]
        Ly += PLCorr[1]
    Rm,Lm = np.sqrt(Ry**2 + Rx**2),np.sqrt(Ly**2 + Lx**2)
    Ra,La = np.degrees(np.arctan(abs(Ry/Rx))),np.degrees(np.arctan(abs(Ly/Lx)))
    if Rx < 0:
        Ra = 180-Ra
    if Lx < 0:
        La = 180-La
    if Ry < 0:
        Ra  = 360 - Ra
    if Ly < 0:
        La = 360 - La
    if addp:
        Rm += 0.01
        Lm += 0.01
    JobData['PRVA'] = [str(int(Ra)),str(int(La))]
    JobData['PRVM'] = [str(round(Rm,2)),str(round(Lm,2))]  
    return
    
