source_file = './data/capture20110815_short.csv'

with open(source_file, 'r') as pcap:
    data=pcap.read().replace('\t\t\t',',').replace('\t\t',',').replace('\t',',').replace(' ',',')#.replace(' ',',',1).replace(':',',',2)#.replace('1970/','')
    #data=pcap.read().replace('')

filename = "./data/capture20110815_short_no_space.csv"
print filename
csv_file = open(filename, "w")
csv_file.write(data)
csv_file.close()

print 'Done ! '
