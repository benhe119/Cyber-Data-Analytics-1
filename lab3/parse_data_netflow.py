source_file = './data/2013-08-20_capture-win10.netflow'

with open(source_file, 'r') as pcap:
    data=pcap.read().replace('\t',',')

filename = source_file + ".csv"
csv_file = open(filename, "w")
csv_file.write(data)
csv_file.close()