import os
from glob import glob

models = sorted([os.path.basename(f).replace(".txt", "") for f in glob("results/*.txt")])

metrics = []
for m in models:
    row = [m]
    for line in open(f"results/{m}.txt"):
        row.append(float(line.split(" ")[-1]))
    #end
    metrics.append(row)
#end

with open("report.csv", "w") as f:
    f.write("model,train,val,test\n")
    for row in metrics:
        f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")
    #end
#end
