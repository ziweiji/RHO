import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default='opendialkg/')
parser.add_argument("--output_dir", type=str, default='opendialkg/')
parser.add_argument("--test_path", type=str, default="")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


train = open(args.input_dir+"/train2id.txt", "r")
valid = open(args.input_dir+"/val2id.txt", "r")
if args.test_path:
    test_path = args.test_path
else:
    test_path = args.input_dir+"/test2id.txt"
test = open(test_path, "r")


lef = {}
rig = {}
rellef = {}
relrig = {}
tot = (int)(train.readline())
for i in range(tot):
	content = train.readline()
	h,t,r = content.strip().split('\t')
	if not (h,r) in lef:
		lef[(h,r)] = []
	if not (r,t) in rig:
		rig[(r,t)] = []
	lef[(h,r)].append(t)
	rig[(r,t)].append(h)
	if not r in rellef:
		rellef[r] = {}
	if not r in relrig:
		relrig[r] = {}
	rellef[r][h] = 1
	relrig[r][t] = 1

tot = (int)(valid.readline())
for i in range(tot):
	content = valid.readline()
	h,t,r = content.strip().split('\t')
	if not (h,r) in lef:
		lef[(h,r)] = []
	if not (r,t) in rig:
		rig[(r,t)] = []
	lef[(h,r)].append(t)
	rig[(r,t)].append(h)
	if not r in rellef:
		rellef[r] = {}
	if not r in relrig:
		relrig[r] = {}
	rellef[r][h] = 1
	relrig[r][t] = 1

tot = (int)(test.readline())
for i in range(tot):
	content = test.readline()
	h,t,r = content.strip().split('\t')
	if not (h,r) in lef:
		lef[(h,r)] = []
	if not (r,t) in rig:
		rig[(r,t)] = []
	lef[(h,r)].append(t)
	rig[(r,t)].append(h)
	if not r in rellef:
		rellef[r] = {}
	if not r in relrig:
		relrig[r] = {}
	rellef[r][h] = 1
	relrig[r][t] = 1

test.close()
valid.close()
train.close()

f = open(args.output_dir+"/type_constrain.txt", "w")
f.write("%d\n"%(len(rellef)))
for i in rellef:
	f.write("%s\t%d"%(i,len(rellef[i])))
	for j in rellef[i]:
		f.write("\t%s"%(j))
	f.write("\n")
	f.write("%s\t%d"%(i,len(relrig[i])))
	for j in relrig[i]:
		f.write("\t%s"%(j))
	f.write("\n")
f.close()

rellef = {}
totlef = {}
relrig = {}
totrig = {}
# lef: (h, r)
# rig: (r, t)
for i in lef:
	if not i[1] in rellef:
		rellef[i[1]] = 0
		totlef[i[1]] = 0
	rellef[i[1]] += len(lef[i])
	totlef[i[1]] += 1.0

for i in rig:
	if not i[0] in relrig:
		relrig[i[0]] = 0
		totrig[i[0]] = 0
	relrig[i[0]] += len(rig[i])
	totrig[i[0]] += 1.0

s11=0
s1n=0
sn1=0
snn=0
f = open(test_path, "r")
tot = (int)(f.readline())
for i in range(tot):
	content = f.readline()
	h,t,r = content.strip().split('\t')
	rign = rellef[r] / totlef[r]
	lefn = relrig[r] / totrig[r]
	if (rign < 1.5 and lefn < 1.5):
		s11+=1
	if (rign >= 1.5 and lefn < 1.5):
		s1n+=1
	if (rign < 1.5 and lefn >= 1.5):
		sn1+=1
	if (rign >= 1.5 and lefn >= 1.5):
		snn+=1
f.close()




f = open(test_path, "r")
f11 = open(args.output_dir+"/1-1.txt", "w")
f1n = open(args.output_dir+"/1-n.txt", "w")
fn1 = open(args.output_dir+"/n-1.txt", "w")
fnn = open(args.output_dir+"/n-n.txt", "w")
fall = open(args.output_dir+"/test2id_all.txt", "w")

tot = (int)(f.readline())
fall.write("%d\n"%(tot))
f11.write("%d\n"%(s11))
f1n.write("%d\n"%(s1n))
fn1.write("%d\n"%(sn1))
fnn.write("%d\n"%(snn))


for i in range(tot):
	content = f.readline()
	h,t,r = content.strip().split('\t')
	rign = rellef[r] / totlef[r]
	lefn = relrig[r] / totrig[r]
	if (rign < 1.5 and lefn < 1.5):
		f11.write(content)
		fall.write("0"+"\t"+content)
	if (rign >= 1.5 and lefn < 1.5):
		f1n.write(content)
		fall.write("1"+"\t"+content)
	if (rign < 1.5 and lefn >= 1.5):
		fn1.write(content)
		fall.write("2"+"\t"+content)
	if (rign >= 1.5 and lefn >= 1.5):
		fnn.write(content)
		fall.write("3"+"\t"+content)
fall.close()
f.close()
f11.close()
f1n.close()
fn1.close()
fnn.close()
