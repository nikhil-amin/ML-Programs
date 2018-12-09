import csv
a=[]
with open('PRG_1.csv') as csfile:
	reader = csv.reader(csfile)
	for row in reader:
		a.append(row)
num_attributes=len(a[0])-1
hypothesis=a[1][:-1]
for i in range (len(a)):
	if a[i][num_attributes] == "yes":
		for j in range(num_attributes):
			if a[i][j]!=hypothesis[j]:
				hypothesis[j]='?';
print("\n The maximally specific hypothesis for training set is ", hypothesis[1:])