import os 
import gmsh
import numpy as np
import gmsh_Vnotch

with open('RESULTATS.txt','w') as file :
          file.write('ident  Nb_elem  Nb_noeuds  K1_G  G_G  EP  Fmax\n')

#Geometry:
#a = 0.2
h = 0.0178
L = 0.0762
gamma = 60

Var1 = [0.00178, 0.00356, 0.00533, 0.00711]
Var2 = np.arange(0.025, 0.425, 0.025) 
#Var3 = range(5,95,5) 
for i in Var1: 
        a=i
        for j in Var2:
        #for k in range(len(Var3)):
                c0 = (h-a)*j
                gmsh_Vnotch.mesh_V(a, h, L, gamma, h/100, h/(500), key=1, c0=c0)

                with open('TP_notch_cast3M.dgibi', 'r') as filefem :
                        fileca = filefem.read()
                        filedata1 = fileca.replace('IDENT1 = 1','IDENT1 = ' + str(i))
                        filedata2 = filedata1.replace('IDENT2 = 2','IDENT2 = ' +str(j))
                with open('TP_notch_cast3M.dgibi', 'w') as filefem:
                        filefem.write(filedata2)

                os.system('castem21 TP_notch_cast3M.dgibi')

                with open('TP_notch_cast3M.dgibi', 'r') as filefem :
                        fileca = filefem.read()
                        filedata1 = fileca.replace('IDENT1 = ' + str(i), 'IDENT1 = 1')
                        filedata2 = filedata1.replace('IDENT2 = ' +str(j),'IDENT2 = 2')
                with open('TP_notch_cast3M.dgibi', 'w') as filefem:
                        filefem.write(filedata2)

                #del file geo & unv created 
                #os.remove('TDCB_' + filename2 + '.geo')
                #os.remove('TDCB_' + filename2 + '.unv')

                #put all results in one file.txt
                #del first line, calculation finished
                with open('RESULTAT_cast3m.txt', 'r') as fin:
                        expdata = fin.read().splitlines(True)
                        
                with open('RESULTATS.txt', 'a') as fout:
                        fout.writelines(expdata[1:])

                #Supprimer les fichier de résultat de chaque itération
                os.remove('RESULTAT_cast3m.txt')

