from calendar import EPOCH
from xml.dom.expatbuilder import theDOMImplementation
import nnpy
import struct
import threading
import time
import logging
import heapq
from p4utils.utils.topology import Topology
from p4utils.utils.sswitch_API import SimpleSwitchAPI
from scapy.all import Ether, sniff, Packet, BitField,IP,sendp


class CpuHeader(Packet):
    name = 'CpuPacket'
    fields_desc = [BitField('hash',0,32), BitField('srcip', 0, 32), BitField('dstip', 0, 32),BitField('srcport', 0, 16),BitField('dstport', 0, 16),BitField('protocol', 0, 8),BitField('counter', 0, 32),BitField('epoch', 0, 16),BitField('outputport', 0, 8)]


class myController(object):
    def __init__(self):
        self.topo = Topology(db="topology.db")
        self.controllers = {}
        self.samplelists = []
        self.mergelist = []
        self.mapping = {} #map the sw_name and the order in samplelists
        self.theta=0.01;
        self.alpha=0.8;
        self.p_thereshold={};
        self.switchcounter={};
        self.epoch=1;
        self.connect_to_switches()

    def connect_to_switches(self):
        for p4switch in self.topo.get_p4switches():
            thrift_port = self.topo.get_thrift_port(p4switch)
            print "p4switch:", p4switch, "thrift_port:", thrift_port
            self.controllers[p4switch] = SimpleSwitchAPI(thrift_port) 
            if (len(p4switch)==4):
                continue;
            self.samplelists.append([])
            self.mapping[p4switch]=len(self.samplelists)-1
            self.p_thereshold[p4switch]=0.3
            self.switchcounter[p4switch]=1
 

    def recv_msg_cpu(self, pkt):
        #print "interface:", pkt.sniffed_on[0:3]
        sw_name=pkt.sniffed_on[0:3]
        sw_id=self.mapping[sw_name]
        packet = Ether(str(pkt))
        cpu_header = CpuHeader(bytes(packet.payload))
        #print "srcip: %s  dstip: %s  srcport:%s dstport: %s hashvalue:%s" % (cpu_header.srcip,cpu_header.dstip,cpu_header.dstport,cpu_header.dstport,cpu_header.hash)
        self.samplelists[sw_id].append([cpu_header.srcip,cpu_header.dstip,cpu_header.dstport,cpu_header.dstport,cpu_header.hash,cpu_header.protocol])
        if self.switchcounter[sw_name] < cpu_header.counter and cpu_header.epoch == self.epoch:
            self.switchcounter[sw_name] = cpu_header.counter
        print "pkghash:%s switch:%s epoch:%d counter:%d" %(cpu_header.hash,sw_name,cpu_header.epoch,self.switchcounter[sw_name])

    def run_cpu_port_loop(self):
        cpu_interfaces = [str(self.topo.get_cpu_port_intf(sw_name).replace("eth0", "eth1")) for sw_name in self.controllers]
        sniff(iface=cpu_interfaces, prn=self.recv_msg_cpu)

    def test (self): #for timeing and change T
        dIP ='1.1.1.1'
        sIP = '0.0.0.0'
        packets=[]
        p = Ether() / IP(dst=dIP, src=sIP);
        packets.append(p);

        while 1:
            time.sleep(200)

            t1=time.time();

            #merge
            print "epoch:%d" %(self.epoch)
            logging.info('epoch:%d', self.epoch)
            self.epoch=self.epoch+1;
            mergelist=[]
            x=0

            #print (self.samplelists)
            for list in self.samplelists:
                mergelist=mergelist+list; #get the list of all sample packet
                if x==0 or x > len(list):
                    x=len(list);
                    if(len(list)==0):
                        x=1;
            networksample = heapq.nsmallest(x,mergelist,lambda k:k[4]) #get the x sample result
            
            if(x==1):
            	x=1;
		khash=2**32;
	    else:
            	khash=networksample[-1][4]+1;
            V = (x-1)/(float(khash)/(2**32)) #get the volum of whole network
            print "estimated_volume: %d" %(V)
            T=V*self.theta;
            #rou=x/V;
            #HH detection
            flowcounter={};
            for flow in networksample:
                flowi=str(flow[0])+str(flow[1])+str(flow[2])+str(flow[3])+str(flow[5])
                if flowi in flowcounter:
                    flowcounter[flowi]=flowcounter[flowi]+1
                else:
                    flowcounter[flowi]=1
            for flowi in flowcounter.keys():
                if flowcounter[flowi] > x*self.theta:
                    print "heavy hitter: %s" %(flowi)
                    logging.info('heavy hitter: %s', flowi)
                
            #update T
            for p4switch in self.topo.get_p4switches():
                if (len(p4switch)==4):
                    continue;
                p_real=float(x-1)/self.switchcounter[p4switch];
                self.p_thereshold[p4switch] = self.p_thereshold[p4switch] * (1-self.alpha) + p_real * self.alpha;
                #T_now = self.controllers[p4switch].register_read(self, register_name, 0)
                T_write = int(self.p_thereshold[p4switch] * (2**32))
                self.controllers[p4switch].register_write("threshold", 0, T_write)
                self.controllers[p4switch].register_write("pkgcounter", 0, 0)
                self.controllers[p4switch].register_write("epoch", 0, self.epoch)
                print "x:%d counter:%d preal:%s new p_threshole in %s: %s" %(x,self.switchcounter[p4switch],p_real, p4switch , self.p_thereshold[p4switch])
                logging.info('x:%d counter:%d preal:%s new p_threshole in %s: %s', x,self.switchcounter[p4switch],p_real, p4switch , self.p_thereshold[p4switch])     
                self.switchcounter[p4switch] = 1
                self.samplelists[self.mapping[p4switch]]=[];
            #print (self.samplelists)
            sendp(packets, iface="sw-cpu")
            t2=time.time();
            print "running time:%s" %(t2-t1)
       

if __name__ == "__main__":
    controller = myController()
    #controller.run_cpu_port_loop()

    logging.basicConfig(filename='example.log',level=logging.DEBUG) 
    logging.info('start')

    thread1 = threading.Thread(name='t1',target=controller.run_cpu_port_loop)
    thread2 = threading.Thread(name='t2',target=controller.test)
    thread1.start()  
    thread2.start() 