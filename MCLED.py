import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import json
import requests
from Crypto.Cipher import AES
import base64
from numpy import array
#%%

#Coordinates
x = np.arange(-42.5,43.0,0.5)
Y,X = np.meshgrid(x,x)
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan(Y/(-X+0.00000001))
ZT = R-R

def downloadjobs(jobname):
    global ZT
    start,final = str(int(jobname)-1),str(int(jobname)+1)
    username,password,url,key = st.secrets['username'],st.secrets['password'],st.secrets['mainurl'],st.secrets['mckey'],
    aeskey = bytes.fromhex(key)
    auth = (username,password)
    response = requests.get(url, auth=auth)

    # Encode the username and password as base64
    auth_encoded = base64.b64encode((username + ':' + password).encode('utf-8')).decode('utf-8')

    # Make a request to the server with the authorization header
    headers = {'Authorization': 'Basic ' + auth_encoded}
    auth = (username,password)


    data = {'jobsrequest':[start,final]}

    response = requests.post(url, json=data, auth=auth)
    received = json.loads(response.content.decode('utf-8'))
    for fname in received:
        tempreceived = {}
        for i,k in enumerate(['data','tag','nonce','rslice','lslice']):
            tempreceived[k] = received[fname].split('&')[i] 
        cipher = AES.new(aeskey, AES.MODE_EAX, bytes.fromhex(tempreceived['nonce']))
        jobdata = cipher.decrypt_and_verify(bytes.fromhex(tempreceived['data']), bytes.fromhex(tempreceived['tag']))
        jobdata = jobdata.decode('utf-8')

        Zr,Zl = jobdata.split('@')[0],jobdata.split('@')[1]
        Zr,Zl = Zr.split('\n'), Zl.split('\n')
        ZTr,ZTl = ZT-ZT,ZT-ZT
        for i in range(len(ZTr)):
            for j in range(len(ZTl)):
                ZTr[i,j] = float(Zr[i*171+j].split(',')[-1])
                ZTl[i,j] = float(Zl[i*171+j].split(',')[-1])
        SliceTkR = tempreceived['rslice']
        SliceTkL = tempreceived['lslice']

        SliceTkR = eval(SliceTkR)
        SliceTkL = eval(SliceTkL)

        SliceTkR = SliceTkR[:2] + (ZTr,) + SliceTkR[3:]
        SliceTkL = SliceTkL[:2] + (ZTl,) + SliceTkL[3:]
        return SliceTkR, SliceTkL


def SliceTk(tuple):
    global grphscreen
    Eye,Rx,Z,ZBb,ZF,nCT = tuple
    A = Rx['HBOX']
    B = Rx['VBOX']
    ED = Rx['FED']
    dED = np.sqrt(Rx['XDEC']**2 + Rx['YDEC']**2)
    plt.figure(figsize=(10,10))
    if Eye=='L':
        Eye='R'
    else:
        Eye='L'
    CT = Rx['BCTHK']
    x = np.arange(-42.5,43.0,0.5)
    XP = np.arange(-42.5,43.0,0.5)/2.0  

    if Eye=='L':
        plt.title('Right')
    if Eye=='R':
        plt.title('Left')
    
    plt.plot(x,ZBb[85] + (CT-nCT) + 42.5,color='royalblue')
    plt.plot(x,Z[85] + 42.5,color='tomato')
    plt.plot(x,ZF[85] - nCT +42.5,color='royalblue')
    plt.axis('off')
    Thck = nCT - ZF + Z
    for i in range(171):
        for j in range(171):
            Thck[i,j] = round(Thck[i,j],3) 
                   
    TextL = [(-B/2.0 + Rx['YDEC'], 60, Thck[85,int(85 - B + 2*Rx['YDEC'])]),
    (B/2.0 + Rx['YDEC'], 60, Thck[85,int(85 + B + 2*Rx['YDEC'])]),
    (0,30,Thck[85,85])]

    if Eye == 'R':
        plt.plot(A/2.0 - Rx['XDEC'] + (x-x),XP-42.5, color='BLACK')
        plt.plot(-A/2.0 - Rx['XDEC'] + (x-x),XP-42.5, color='BLACK')
        TextL+=[(A/2.0 - Rx['XDEC'], -60, Thck[int(85 + A - 2*Rx['XDEC']),85])]
        TextL+=[(-A/2.0 - Rx['XDEC'],-60, Thck[int(85 - A - 2*Rx['XDEC']),85])]
    else:
        plt.plot( A/2.0 + Rx['XDEC'] + (x-x),XP-42.5, color='BLACK')
        plt.plot(-A/2.0 + Rx['XDEC'] + (x-x),XP-42.5, color='BLACK')
        TextL+=[(A/2.0 + Rx['XDEC'], -60, Thck[int(85 + A + 2*Rx['XDEC']),85])]
        TextL+=[(-A/2.0 + Rx['XDEC'],-60, Thck[int(85 - A + 2*Rx['XDEC']),85])]
        
    for t in TextL:
        if Eye=='L':
            plt.text(t[0],t[1], str(t[2]), color='RED')
        if Eye=='R':
            plt.text(t[0],t[1], str(t[2]), color='RED')
      
    ZBb = np.transpose(ZBb)
    Z = np.transpose(Z)
    ZF = np.transpose(ZF)
    plt.plot(-x,ZBb[85] + (CT-nCT)-42.5,color='royalblue')
    plt.plot(x,Z[85] - 42.5,color='tomato')
    plt.plot(-x,ZF[85] - nCT -42.5, color='royalblue')
    plt.plot( B/2.0 + Rx['YDEC'] + x-x,XP+42.5, color='BLACK')
    plt.plot(-B/2.0 + Rx['YDEC'] + x-x,XP+42.5, color='BLACK')
    plt.xlim(-85,85)
    st.set_option('deprecation.showPyplotGlobalUse', False)


    st.pyplot()
    return



#i figured out how to graph and keep the program running
#all you literally need to do is put so it keeps going...
#before it wouldnt show unless the program finished
#but now i can keep my plugin running and itll keep graphing
#ill make it show every graph the first time we try to implement it
#to better monitor the jobs and be able to fix my plug in easier
#eventually we can just turn them off


#st.write('# MCLED #')  
col1,col2 = st.columns(2)

Rx = {}
Rx['HBOX'],Rx['VBOX'],Rx['XDEC'],Rx['YDEC'],Rx['FED'],Rx['BCTHK'] = 50,40,0,0,58,8
ZT = 90 - np.sqrt(90**2 - R**2)
BrEye = 'R',Rx,ZT,ZT,ZT,2.5
BlEye = 'L',Rx,ZT,ZT,ZT,2.5

if 'show' not in st.session_state:
    with col1:
        SliceTk(BrEye)
    with col2:
        SliceTk(BlEye)

if 'show' in st.session_state:
    jobv = st.session_state['show']
    RightEye,LeftEye = downloadjobs(jobv)
    with col1:
        SliceTk(RightEye)
    with col2:
        SliceTk(LeftEye)


with col2:
    jobn = st.text_input('Job Number')
    plot = st.button('Plot')

if plot:
    st.session_state['show'] = jobn
