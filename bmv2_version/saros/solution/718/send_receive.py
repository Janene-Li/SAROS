import nnpy
import struct
import threading
import time
import heapq
from p4utils.utils.topology import Topology
from p4utils.utils.sswitch_API import SimpleSwitchAPI
from scapy.all import Ether, sniff, Packet, BitField


class CpuHeader(Packet):
    name = 'CpuPacket'
    fields_desc = [BitField('hash',0,32), BitField('srcip', 0, 32), BitField('dstip', 0, 32),BitField('srcport', 0, 16),BitField('dstport', 0, 16),BitField('protocol', 0, 8)]


class myController(object):
    def __init__(self):
        self.topo = Topology(db="topology.db")
        self.controllers = {}
        self.connect_to_switches()
        self.samplelists = []
        self.mapping={} #map the sw_name and the order in samplelists

    def connect_to_switches(self):
        for p4switch in self.topo.get_p4switches():
            thrift_port = self.topo.get_thrift_port(p4switch)
            print "p4switch:", p4switch, "thrift_port:", thrift_port
            self.controllers[p4switch] = SimpleSwitchAPI(thrift_port) 
            #self.samplelists. append([])
            #self.mapping[p4switch]=len(self.samplelists)-1
            self.mapping[1]=1;
 

    def recv_msg_cpu(self, pkt):
        print "interface:", pkt.sniffed_on[0:3]
        sw_name=pkt.sniffed_on[0:3]
        sw_id=mapping[sw_name]
	packet = Ether(str(pkt))
	cpu_header = CpuHeader(bytes(packet.payload))
        print "srcip: %s  dstip: %s  srcport:%s dstport: %s hashvalue:%s" % (cpu_header.srcip,cpu_header.dstip,cpu_header.dstport,cpu_header.dstport,cpu_header.hash)
        selp.samplelists[sw_id].append([cpu_header.srcip,cpu_header.dstip,cpu_header.dstport,cpu_header.dstport,cpu_header.hash,cpu_header.protocol])

    def run_cpu_port_loop(self):
        cpu_interfaces = [str(self.topo.get_cpu_port_intf(sw_name).replace("eth0", "eth1")) for sw_name in self.controllers]
        sniff(iface=cpu_interfaces, prn=self.recv_msg_cpu)


    def test (self): #for timeing and change T
        second = sleeptime(0,0,5);
	while 1:
           time.sleep(second)
                   
           for p4switch in self.topo.get_p4switches():
               #T_now = self.controllers[p4switch].register_read(self, register_name, 0)
#############################################################
                                                            #update T
#############################################################
               self.controllers[p4switch].register_write(self, register_name, 0, T_new)
       

if __name__ == "__main__":
    controller = myController()
    controller.run_cpu_port_loop()
    #thread1 = threading.Thread(name='t1',target=controller.run_cpu_port_loop)
    #thread2 = threading.Thread(name='t2',target=controller.test)
   # thread1.start()  
   # thread2.start() 