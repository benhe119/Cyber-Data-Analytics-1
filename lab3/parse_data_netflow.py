source_file = './data/2013-08-20_capture-win10.netflow'

with open(source_file, 'r') as pcap:
    data=pcap.read().replace('\t',',').replace('/',',') #.replace(' ',',',1).replace(':',',',2)#.replace('1970/','')
    #data=pcap.read().replace('')

filename = source_file + ".csv"
csv_file = open(filename, "w")
csv_file.write(data)
csv_file.close()
