#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cProfile import label
import numpy as np
import array as arr
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

#1 a jusante
#surf00001_sonda00145.dat"  #x = 12.2; y=30; z=18 r/R = 0
#surf00002_sonda00145.dat" #x = 12.2; y=30; z=20,51 r/R = 0.5
#surf00002_sonda00397.dat" #x = 12.2; y=30; z=22,5261 r/R = 0.9
#ultimo jusante
#surf00001_sonda00625.dat"  #x = 12.2; y=30; z=18 r/R = 0
#surf00002_sonda00235.dat" #x = 12.2; y=80; z=20,51 r/R = 0.5
#surf00002_sonda00487.dat" #x = 12.2; y=80; z=22,5261 r/R = 0.9

def espectro(t,xc,u,v,w):
    ul = u - np.mean(u)
    vl = v - np.mean(v)
    wl = w - np.mean(w)

    Enel = (ul * ul + vl * vl + wl * wl)/2

    contUl = 0
    for i in u:
        if contUl<45:
            Enel[contUl] = Enel[contUl]*np.sin(contUl*2*np.pi/180)
        if (contUl>(len(xc)-46)):
            Enel[contUl] = Enel[contUl]*np.sin((2*(45 + contUl - (len(xc)-46)))*np.pi/180)
        contUl = contUl + 1
    yl = Enel
    nl = len(yl)
    kl = np.arange(nl)
    frql = kl
    frql = frql[range(nl//2)]
    Yl1 = np.fft.fft(yl)
    Yl = Yl1.real ** 2 + Yl1.imag ** 2
    Yl = Yl[range(nl//2)]

    #Ene = (u*u + v*v + w*w)/2
#
    #contU = 0
    #for i in u:
    #    if contU<45:
    #        Ene[contU] = Ene[contU]*np.sin(contU*2*np.pi/180)
    #    if (contU>(len(xc)-46)):
    #        Ene[contU] = Ene[contU]*np.sin((2*(45 + contU - (len(xc)-46)))*np.pi/180)
    #    contU = contU+1
    #y = Ene
    #n = len(y)
    #k = np.arange(n)
    #frq = k
    #frq = frq[range(n//2)]
    #Y1 = np.fft.fft(y)
    #Y = Y1.real ** 2 + Y1.imag ** 2
    #Y = Y[range(n//2)]
    #return Y,frq, Yl, frql
    return Yl, frql

font = FontProperties()
font.set_family('serif')

x1R = 1
x2R = 100000
#x2R = 1000000
xR = np.arange(x1R,x2R,10)
yR = np.exp((-5.0/3.0)*np.log(xR))*1000000000
yRl = np.exp((-25.0/3.0)*np.log(xR))* (10 ** 34)

dir = [""]
probe = ["surf00001_sonda00017.dat"]


filelist = []
for j in range(1):
    for i in range(1):
        Path = dir[j] + probe[i]
        filelist.append(Path)

print(filelist)

pularLinhas= 10000 #cerca de 10s
#max_linhas = 300000

t = []
xc = []
u = []
v = []
w = []

# filelist=[Path,Path2,Path3]
for file in filelist:
    [t0,x,u0,v0,w0] = np.loadtxt(file,unpack=True,skiprows=pularLinhas,usecols=(1,3,6,7,8))
    t.append(t0)
    xc.append(x)
    u.append(u0)
    v.append(v0)
    w.append(w0)
    #print(np.std(t))


    
ept_kinetic = []

for i in range(len(filelist)):
    ept = espectro(t[i],xc[i],u[i],v[i],w[i])
    ept_kinetic.append(ept)



raio = [0,0.5,0.9] *4
#jusante = ['Malha 50 mm','Malha 75 mm','Malha 150 mm','Malha 75 mm - 4 seg simulation']
cor =["darkgray","tab:blue","tab:red","black"]
plt.figure()
print(len(ept_kinetic))


for i in range(len(ept_kinetic)):

    plt.xlabel('$\log{f}$ [Hz]', fontsize=19, fontproperties=font)
    plt.ylabel('$\log{E(f)}$', fontsize=19, fontproperties=font)
    plt.loglog(ept_kinetic[i][1],abs(ept_kinetic[i][0]),color='red',linewidth=0.8)
    p_ret  = plt.loglog(xR,yR,'--',color='black',linewidth=1.5, label='$m = -5/3$')

for i in range(len(ept_kinetic)):
    
    #plt.xlabel('f [Hz]', fontsize=19, fontproperties=font)
    #plt.ylabel('E(f)', fontsize=19, fontproperties=font)
    #plt.loglog(ept_kinetic[i][3],abs(ept_kinetic[i][2]),color=cor[i],linewidth=0.8,label='Aristeu')
    p_ret  = plt.loglog(xR,yRl,'--',color='green',linewidth=1.5, label='$m = -25/3$')

#plt.text(100, 10000, '$-5/3$', fontsize=10)
#plt.text(195, 1000000, '$-25/3$', fontsize=10)
ax= plt.gca()
#ax.set_xlim([1,10000])
#ax.set_ylim([0.1,10000000])	
ax.set_xlim([1,100000])
ax.set_ylim([0.00001,10000000])	
ax.legend(title='Coeficiente Angular')
plt.title('Densidade Espectral de Energia Cinética Turbulenta')	
plt.grid()
plt.savefig('Espectro_final.png', format='png', dpi=350)
plt.show()
	
    # plt.legend()

#plt.show()
