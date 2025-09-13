from scapy.all import sniff, IP
import time
import socket
import struct
import re

previous_time = None  # To track inter-arrival time

def is_valid_ipv4(ip):
    pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    return pattern.match(ip) is not None

def ip_to_int(ip):
    return struct.unpack("!I", socket.inet_aton(ip))[0]

def extract_packet_features(pkt):
    global previous_time

    if IP in pkt:
        current_time = time.time()
        inter_arrival = current_time - previous_time if previous_time else 0
        previous_time = current_time

        tot_bytes = len(pkt)
        duration = 0.1  # Placeholder
        tot_pkts = 1
        pkts_per_sec = tot_pkts / duration

        # Convert IP addresses to integers
        src_addr = ip_to_int(pkt[IP].src) if is_valid_ipv4(pkt[IP].src) else None
        dst_addr = ip_to_int(pkt[IP].dst) if is_valid_ipv4(pkt[IP].dst) else None

        return {     
            "SrcAddr": src_addr,           # Converted to integer
            "DstAddr": dst_addr,           # Converted to integer
            "Dur": duration,
            "TotPkts": tot_pkts,
            "TotBytes": tot_bytes,
            "SrcJitter": inter_arrival,
            "DstJitter": inter_arrival / 2,
            "Proto": pkt[IP].proto            # Numeric protocol value
        }

    return None

def get_event():
    result = {"data": None}

    def callback(pkt):
        features = extract_packet_features(pkt)
        if features:
            result["data"] = features
            return True  # Stop sniffing

    sniff(prn=callback, filter="ip", store=False, stop_filter=lambda x: result["data"] is not None, timeout=5)
    return result["data"]