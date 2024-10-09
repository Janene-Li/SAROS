/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

#define HASH_THRESHOLD 1288490189  //threshold=0.3 for kmv hash，Replace floating with integer in the range 0 to 4294967295 （32 bits）
#define BLOOM_SIZE 65536
#define PKT_INSTANCE_TYPE_NORMAL 0
#define PKT_INSTANCE_TYPE_INGRESS_CLONE 1
#define PKT_INSTANCE_TYPE_EGRESS_CLONE 2
#define PKT_INSTANCE_TYPE_COALESCED 3
#define PKT_INSTANCE_TYPE_INGRESS_RECIRC 4
#define PKT_INSTANCE_TYPE_REPLICATION 5
#define PKT_INSTANCE_TYPE_RESUBMIT 6

const bit<16> TYPE_IPV4 = 0x0800;
const bit<16> VirtualLAN= 0x8100;
/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/
typedef bit<9>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;

typedef bit<16> ether_type_t;
const ether_type_t ETHERTYPE_IPV4 = 16w0x0800;
const ether_type_t ETHERTYPE_ARP = 16w0x0806;
const ether_type_t ETHERTYPE_IPV6 = 16w0x86dd;
const ether_type_t ETHERTYPE_VLAN = 16w0x8100;

typedef bit<8> ip_protocol_t;
const ip_protocol_t IP_PROTOCOLS_ICMP = 1;
const ip_protocol_t IP_PROTOCOLS_TCP = 6;
const ip_protocol_t IP_PROTOCOLS_UDP = 17;


header cpu_t {
    bit<32> hash;
    bit<32> srcip;
    bit<32> dstip;
    bit<16> srcport;
    bit<16> dstport;
    bit<8> protocol;
    bit<32> counter;
    bit<16> epoch;
    bit <8> outputport;
}

header ethernet_t {
    macAddr_t dstAddr;
    macAddr_t srcAddr;
    bit<16>   etherType;
}

header ipv4_t {
    bit<4>    version;
    bit<4>    ihl;
    bit<8>    tos;
    bit<16>   totalLen;
    bit<16>   identification;
    bit<3>    flags;
    bit<13>   fragOffset;
    bit<8>    ttl;
    bit<8>    protocol;
    bit<16>   hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
}

header tcp_t {
    bit<16> src_port;
    bit<16> dst_port;
    bit<32> seq_no;
    bit<32> ack_no;
    bit<4> data_offset;
    bit<4> res;
    bit<8> flags;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgent_ptr;
}

header udp_t {
    bit<16> src_port;
    bit<16> dst_port;
    bit<16> hdr_length;
    bit<16> checksum;
}

header icmp_t {
    bit<8> type;
    bit<8> code;
    bit<16> checksum;
}

header vlan_tag_h {
    bit<3> pcp;
    bit<1> cfi;
    bit<12> vid;
    bit<16> type;
}

struct metadata {   
    bit<32>  dstip;
    bit<32>  srcip;
    bit<16>  srcport;
    bit<16>  dstport;
    bit<8>     protocol;
    bit<8>     kmvflag;
    bit<32>  khash;
    bit<32>  counter;
    bit<16>  epoch;
}


struct headers {
    ethernet_t   ethernet;
    ipv4_t        ipv4;
    cpu_t         cpu;
    udp_t        udp;
    tcp_t          tcp;
    icmp_t      icmp;
    vlan_tag_h vlan;
}

 

/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/
 

parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {
 
    state start {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType){
            TYPE_IPV4: ipv4;
            VirtualLAN:vlan;
            default: accept;
        }
    }
 
    state vlan{
        packet.extract(hdr.vlan);
        transition select(hdr.vlan.type){
            TYPE_IPV4: ipv4;
            default: accept;
        }
    }

    state ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol){
            IP_PROTOCOLS_TCP : parse_tcp;
            IP_PROTOCOLS_UDP : parse_udp;
            IP_PROTOCOLS_ICMP : parse_icmp;
            default : accept;
        }
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        transition accept;
    }

    state parse_udp {
        packet.extract(hdr.udp);
        transition accept;
    }
    state parse_icmp {
        packet.extract(hdr.icmp);
        transition accept;
    }
}

 

 

/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply {  }
}

 
/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/


control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {

    action drop() {
        mark_to_drop(standard_metadata);
    }

    action ipv4_forward(macAddr_t dstAddr, egressSpec_t port) {
        //set the src mac address as the previous dst, this is not correct right?
        hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;
       //set the destination mac address that we got from the match in the table
        hdr.ethernet.dstAddr = dstAddr;
        //set the output port that we also get from the table
        standard_metadata.egress_spec = port;
        //decrease ttl by 1
        hdr.ipv4.ttl = hdr.ipv4.ttl -1;
    }
    table ipv4_lpm {
        key = {
            hdr.ipv4.dstAddr: lpm;
        }
        actions = {
            ipv4_forward;
            drop;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    register<bit<16>>(BLOOM_SIZE) bloomfilter;
    register<bit<32>>(1) pkgcounter;
    register<bit<16>>(1) epoch;

    register<bit<32>>(1) threshold;
    action bloom_hash(in bit<32> ipv4_src, in bit<32> ipv4_dst, in bit<16> srcport, in bit<16> dstport, in bit<8> protocol, in bit<16> checksum, in bit<32> seq_no, in bit<16> hdlen, out bit<32> h1, out bit<32> h2, out bit<32> h3) {
        hash(h1, HashAlgorithm.crc32, 32w0, {ipv4_src, ipv4_dst,srcport,dstport,checksum,seq_no,hdlen}, 32w0x0000ffff);
        hash(h2, HashAlgorithm.crc16, 32w0, {ipv4_src, ipv4_dst,srcport,dstport,checksum,seq_no,hdlen}, 32w0x0000ffff);
        hash(h3, HashAlgorithm.identity, 32w0, {ipv4_src, ipv4_dst,srcport,dstport,checksum,seq_no,hdlen}, 32w0x0000ffff);
    }
    action kmv_hash(in bit<32> ipv4_src, in bit<32> ipv4_dst, in bit<16> srcport, in bit<16> dstport, in bit<8> protocol, in bit<16> checksum, in bit<32> seq_no, in bit<16> hdlen, out bit<64> khash) {
        hash(khash, HashAlgorithm.crc32_custom, 64w0, {ipv4_src, ipv4_dst,srcport,dstport,protocol,checksum,seq_no,hdlen}, 64w0xffffffffffffffff);
    }

    action routehash(in bit<32> ipv4_src, in bit<32> ipv4_dst, in bit<16> srcport, in bit<16> dstport, in bit<8> protocol, out bit<32> rhash) {
        hash(rhash, HashAlgorithm.crc32, 32w00000000, {ipv4_src, ipv4_dst,srcport,dstport,protocol}, 32w0x00000004);
    }
    
    action forward(in bit<32> rhash){
       bit<9> finalport = 1;
       if((standard_metadata.ingress_port == 3 ||  standard_metadata.ingress_port  == 4) && (rhash[8:0]==2 || rhash[8:0] ==3)) finalport= rhash[8:0] -2;
       if((standard_metadata.ingress_port == 3 ||  standard_metadata.ingress_port  == 4) && (rhash[8:0]==0 || rhash[8:0] ==1)) finalport= rhash[8:0];
       if((standard_metadata.ingress_port == 1 ||  standard_metadata.ingress_port  == 2)){
           if(rhash[8:0] == standard_metadata.ingress_port - 1) finalport=rhash[8:0] +1;
           else finalport=rhash[8:0];
           if(finalport > 3) finalport = 0;
       }
       //if(rhash[8:0]==3) 
       //finalport = 4;
       standard_metadata.egress_spec = finalport+1;
    }

    apply {
        //standard_metadata.egress_spec = 4;
        bit<32> rhash;
        //only if IPV4 the rule is applied. Therefore other packets will not be forwarded.
        if (hdr.ipv4.isValid()){
            //ipv4_lpm.apply();
            meta.srcip=hdr.ipv4.srcAddr;
            meta.dstip=hdr.ipv4.dstAddr;
            meta.protocol=hdr.ipv4.protocol;
       }

        bit<32> T;
        threshold.read(T,0);
        if(T==0){
              T=HASH_THRESHOLD;
        }

        bit<32> h1=0;
        bit<32> h2=0;
        bit<32> h3=0;
        bit<32> khash=0;
        bit<64> hash64=0;
        if (hdr.tcp.isValid()) {
            bloom_hash(hdr.ipv4.srcAddr, hdr.ipv4.dstAddr, hdr.tcp.src_port, hdr.tcp.dst_port,hdr.ipv4.protocol, hdr.tcp.checksum,hdr.tcp.seq_no,0, h1, h2, h3);
            kmv_hash(hdr.ipv4.srcAddr, hdr.ipv4.dstAddr, hdr.tcp.src_port, hdr.tcp.dst_port,hdr.ipv4.protocol, hdr.tcp.checksum,hdr.tcp.seq_no,0, hash64);
            routehash(hdr.ipv4.srcAddr, hdr.ipv4.dstAddr, hdr.tcp.src_port, hdr.tcp.dst_port,hdr.ipv4.protocol, rhash);
            khash[31:24]=hash64[15:8]; khash[23:16]=hash64[31:24]; khash[15:8]=hash64[7:0]; khash[7:0]=hash64[23:16];
            khash=~khash;
            meta.khash=khash;
            meta.srcport=hdr.tcp.src_port;
            meta.dstport=hdr.tcp.dst_port;
            forward(rhash);
        }
        else if (hdr.udp.isValid()) {
            bloom_hash(hdr.ipv4.srcAddr, hdr.ipv4.dstAddr, hdr.udp.src_port, hdr.udp.dst_port,hdr.ipv4.protocol, hdr.udp.checksum, 0,hdr.udp.hdr_length,h1, h2, h3);
            kmv_hash(hdr.ipv4.srcAddr, hdr.ipv4.dstAddr, hdr.udp.src_port, hdr.udp.dst_port, hdr.ipv4.protocol, hdr.udp.checksum, 0,hdr.udp.hdr_length,hash64);
            routehash(hdr.ipv4.srcAddr, hdr.ipv4.dstAddr, hdr.udp.src_port, hdr.udp.dst_port,hdr.ipv4.protocol, rhash);
            khash[31:24]=hash64[15:8]; khash[23:16]=hash64[31:24]; khash[15:8]=hash64[7:0]; khash[7:0]=hash64[23:16];
            khash=~khash;
            meta.khash=khash;
            meta.srcport=hdr.udp.src_port;
            meta.dstport=hdr.udp.dst_port;
            forward(rhash);
        }else{
            standard_metadata.egress_spec =0;
        }



        bit<16> bloom1;
        bit<16> bloom2;
        bit<16> bloom3;
        bit<16> epochnow=0;
        bit<8> flag;

        bloomfilter.read(bloom1,h1);
        bloomfilter.read(bloom2,h2);
        bloomfilter.read(bloom3,h3);
        epoch.read(epochnow,0);

        if(epochnow==0) epochnow=1;
        meta.epoch=epochnow;

        if(bloom1 == epochnow){
            flag=0; //its a old packet
            pkgcounter.read(meta.counter,0);
        }else{
            flag=1;
            pkgcounter.read(meta.counter,0);
            meta.counter=meta.counter+1;
            pkgcounter.write(0, meta.counter);
        }

        bloomfilter.write(h1,epochnow);
        //bloomfilter.write(h2,epochnow);
        //bloomfilter.write(h3,epochnow);

        bit<8> kmvflag;
        if(khash < HASH_THRESHOLD && flag == 1){
            meta.kmvflag = 1;
        }else{
            meta.kmvflag = 0;
        }
    }
}

 

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/
control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    apply {
     if (standard_metadata.instance_type == 0 && meta.kmvflag==1 ){
         clone3(CloneType.E2E,100, meta);
     }
     //handle the cloned packet   meta.kmvflag   standard_metadata.instance_type != 0
     if (standard_metadata.instance_type != 0){
        hdr.ethernet.etherType =0x1234;       // EtherType 换成 0x1234，controller 用这个来过滤
        hdr.cpu.setValid();                                         // 把克隆的 packet 的 cpu header 设置成 valid
        hdr.vlan.setInvalid();                                         // 把克隆的 packet 的 cpu header 设置成 valid
        hdr.cpu.hash = meta.khash;
        hdr.cpu.srcip = meta.srcip;  
        hdr.cpu.dstip = meta.dstip;
        hdr.cpu.srcport = meta.srcport;
        hdr.cpu.dstport = meta.dstport;
        hdr.cpu.protocol = meta.protocol;
        hdr.cpu.counter = meta.counter;
        hdr.cpu.epoch= meta.epoch;
        hdr.cpu.outputport = standard_metadata.egress_spec[7:0];
        //truncate((bit<32>)22);                              // 截断 ether(48+48+16 bits)+cpu(48+16 bits) header = 22 bytes 之后的部分，因为我们只需要 cpu header 里面的信息作为 payload
     }
    }

}
 

/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers hdr, inout metadata meta) {
     apply {
        update_checksum(
            hdr.ipv4.isValid(),
            { hdr.ipv4.version,
              hdr.ipv4.ihl,
              hdr.ipv4.tos,
              hdr.ipv4.totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16);
    }
}

 
/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/

control MyDeparser(packet_out packet, in headers hdr) {
    apply {
        //parsed headers have to be added again into the packet.
        packet.emit(hdr.ethernet);    
        packet.emit(hdr.vlan);    
        packet.emit(hdr.cpu);   
        packet.emit(hdr.ipv4);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
        packet.emit(hdr.icmp);
    }
}


/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

//switch architecture

V1Switch(
MyParser(),
MyVerifyChecksum(),
MyIngress(),
MyEgress(),
MyComputeChecksum(),
MyDeparser()
) main;
