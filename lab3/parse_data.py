
source_file = './data/capture20110815-2.pcap.netflow.labeled'

with open(source_file, 'r') as pcap:
    data=pcap.read().replace('\t\t\t', ',').replace('\t\t',',').replace('\t',',')

filename = source_file + ".csv"
csv_file = open(filename, "w")
csv_file.write(data)
csv_file.close()