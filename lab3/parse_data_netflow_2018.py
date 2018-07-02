source_file = './data/capture20110818.pcap.netflow.labeled'

with open(source_file, 'r') as pcap:
	data=pcap.read()
	data=data.replace(' 0.000 ICMP ','	0.000	ICMP	')
 	data=data.replace(' -> ','	->	')
 	data=data.replace(' -> ','	->	')
 	data=data.replace('147.32.96.69 ___ 0 1 1066 1 Botnet', '147.32.96.69	___	0	1	1066	1	Botnet')
 	data=data.replace('\t\t\t',',').replace('\t\t',',').replace('\t',',')
    #data=pcap.read().replace('')

   #2011-08-18 12:18:37.527 0.000 ICMP 147.32.84.193 -> 147.32.96.69 ___ 0 1 1066 1 Botnet


filename = source_file + ".csv"
csv_file = open(filename, "w")
csv_file.write(data)
csv_file.close()
