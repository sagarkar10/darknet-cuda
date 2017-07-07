import sys
import numpy as s
s.seterr(divide='ignore', invalid='ignore')
def rmsd(l1,l2):
    print("DARKNET stats: max: {:f} min: {:f} mean: {:f}, stdev: {:f}, variance: {:f}".format(max(l1), min(l1), s.mean(l1),s.std(l1),s.var(l1)))

    print("ARM stats: max: {:f} min: {:f} mean: {:f}, stdev: {:f}, variance: {:f}".format(max(l2), min(l2), s.mean(l2),s.std(l2),s.var(l2)))

    print("DARKNET/ARM ratio: max: {:f} min: {:f} mean: {:f}, stdev: {:f}, variance: {:f}".format(s.divide(max(l1),max(l2)),s.divide(min(l1),min(l2)),s.divide(s.mean(l1),s.mean(l2)),s.divide(s.std(l1),s.std(l2)),s.divide(s.var(l1),s.var(l2))))

    ret = sum([(x-y)**2 for (x,y) in zip(l1,l2)])
    print("rmsd:",ret**.5)
    return ret**.5

def str2float(fc):
    return [float(x) for x in fc]

w_d_st = "dump/weights"
b_d_st = "dump/biases"
b_a_st = "pi64/dump/biases"
w_a_st = "pi64/dump/weights"
k=0;
wt_dn=[w_d_st+str(x)+".txt" for x in range(9)]
bs_dn=[b_d_st+str(x)+".txt" for x in range(9)]
wt_arm=[w_a_st+str(x)+".txt" for x in range(9)]
bs_arm=[b_a_st+str(x)+".txt" for x in range(9)]

out_conv=["out_conv0.txt","out_conv1.txt","out_conv2.txt","out_conv3.txt","out_conv4.txt","out_conv5.txt","out_conv6.txt","out_conv7.txt","out_conv8.txt"]
out_act=["out_act0.txt","out_act1.txt","out_act2.txt","out_act3.txt","out_act4.txt","out_act5.txt","out_act6.txt","out_act7.txt","out_act8.txt"]
out_batchnorm=["out_batchnorm0.txt","out_batchnorm1.txt","out_batchnorm2.txt","out_batchnorm3.txt","out_batchnorm4.txt","out_batchnorm5.txt","out_batchnorm6.txt","out_batchnorm7.txt"]
out_pool=["out_pool0.txt","out_pool1.txt","out_pool2.txt","out_pool3.txt","out_pool4.txt","out_pool5.txt"]
#print(sys.argv)
if len(sys.argv)==3:
    print("darknet-arm filename")
    f1,f2 = sys.argv[1],sys.argv[2]
    print(f1,f2)
    f1_c = open(f1).read().split('\n')[:-1]
    f2_c = open(f2).read().split('\n')[:-1]
    print(len(f1_c),len(f2_c))
    assert (len(f1_c)==len(f2_c)),"Size mismatch!!\n"
    start = 111000
   
    f1_c = str2float(f1_c)
    f2_c = str2float(f2_c)
    print(f1_c[start:start+20])
    print(f2_c[start:start+20])
    print("rmsd/n: ",rmsd(f1_c,f2_c)/len(f1_c))

    exit()

for i in range(len(wt_dn)):
    w_arm=open(wt_arm[i]).read().split('\n')[:-1]
    b_arm=open(bs_arm[i]).read().split('\n')[:-1]
    w_dn=open(wt_dn[i]).read().split('\n')[:-1]
    b_dn=open(bs_dn[i]).read().split('\n')[:-1]
    print("matching: ", bs_arm[i], bs_dn[i])

for i in range(len(out_conv)):
    print("Out_Conv Comparison of Layer: ", i)
    f1_c = open("dump/"+out_conv[i]).read().split('\n')[:-1]
    f2_c = open("pi64/dump/"+out_conv[i]).read().split('\n')[:-1]
    print("darknet: ", len(f1_c),"arm: ", len(f2_c))
    assert (len(f1_c)==len(f2_c)),"Size mismatch!!\n"

    f1_c = str2float(f1_c)
    f2_c = str2float(f2_c)

    print("rmsd/n: ",rmsd(f1_c,f2_c)/len(f1_c))

    if i<7:
        print("Out_Batchnorm Comparison of Layer: ", i)
        f1_c = open("dump/"+out_batchnorm[i]).read().split('\n')[:-1]
        f2_c = open("pi64/dump/"+out_batchnorm[i]).read().split('\n')[:-1]
        print("darknet: ", len(f1_c),"arm: ", len(f2_c))
        assert (len(f1_c)==len(f2_c)),"Size mismatch!!\n"

        f1_c = str2float(f1_c)
        f2_c = str2float(f2_c)

        print("rmsd/n: ",rmsd(f1_c,f2_c)/len(f1_c))

        print("Out_Act Comparison of Layer: ", i)
        f1_c = open("dump/"+out_act[i]).read().split('\n')[:-1]
        f2_c = open("pi64/dump/"+out_act[i]).read().split('\n')[:-1]
        print("darknet: ", len(f1_c),"arm: ", len(f2_c))
        assert (len(f1_c)==len(f2_c)),"Size mismatch!!\n"

        f1_c = str2float(f1_c)
        f2_c = str2float(f2_c)

        print("rmsd/n: ",rmsd(f1_c,f2_c)/len(f1_c))

    if i<6:
        print("Out_Pool Comparison of Layer: ", i)
        f1_c = open("dump/"+out_pool[i]).read().split('\n')[:-1]
        f2_c = open("pi64/dump/"+out_pool[i]).read().split('\n')[:-1]
        print("darknet: ", len(f1_c),"arm: ", len(f2_c))
        assert (len(f1_c)==len(f2_c)),"Size mismatch!!\n"

        f1_c = str2float(f1_c)
        f2_c = str2float(f2_c)

        print("rmsd/n: ",rmsd(f1_c,f2_c)/len(f1_c))
