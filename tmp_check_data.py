import csv
import json

with open('d:/20260407P2P-Insole-Realtime-2/P2P-Insole-Realtime/data/skeleton/skeleton.csv', 'r') as f:
    reader = csv.reader(f)
    for i in range(6):
        row = next(reader)
        if i == 2:
            print("Row 2 (Joints):", json.dumps([x for x in row if x]))
        if i == 3:
            print("Row 3 (XYZ):", json.dumps(row[:30]))

with open('d:/20260407P2P-Insole-Realtime-2/P2P-Insole-Realtime/data/insole/Insole_l.csv', 'r') as f:
    reader = csv.reader(f)
    for i in range(3):
        row = next(reader)
        if i == 1:
            print("Insole Cols:", json.dumps(row))
