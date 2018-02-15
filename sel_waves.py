import time
import struct
import datetime
import numpy

def mg_readv(mg, alist):
    result = mg.readwrite(alist=alist, write=0)
    # print "mgr", len(alist), len(result)
    # pr = mg.parse_readvalue(result)
    pr = list(struct.unpack('!'+'i'*len(alist)*2, result[8:8+8*len(alist)]))
    # print pr
    return pr[1::2]
    # return [struct.unpack('!I', r[2])[0] for r in result]

def reg_read_value(mg, alist):
    result = []
    while len(alist) > 128:
        result.extend(mg_readv(mg, alist[0:128]))
        # print "rrv1",len(result)
        alist = alist[128:]
    if alist:
        result.extend(mg_readv(mg, alist))
        # print "rrv2",len(result)
    return result

def interpret_slow_data(slow_data):
    # circle_count, circle_stat, adc1_min, adc1_max, adc2_min, adc2_max, adc3_min, adc3_max, tag_now, tag_old, timestamp
    circle_count = slow_data[0]*256+slow_data[1]
    circle_stat = slow_data[2]*256+slow_data[3]
    tag_now = slow_data[16]
    tag_old = slow_data[17]
    # Three channels of ADC min/max
    mm = [slow_data[ix]*256+slow_data[ix+1] for ix in range(4, 16, 2)]
    mm = [m-65536 if m > 32767 else m for m in mm]
    # (almost) 64-bit timestamp
    time_stamp = 0
    for ix in range(8):
        time_stamp = time_stamp*256 + slow_data[25-ix]
    time_stamp = time_stamp/32
    return circle_count, circle_stat, mm, tag_now, tag_old, time_stamp

def iq_buf_collect(prc, chcount, npt, cav_mask=1, verbose=False):
    mg = prc.mg
    # XXX hard-coded circle_data array base address!
    b_base = 0x1a0000 if cav_mask == 1 else 0x1b0000
    s_base = 0x180000 if cav_mask == 1 else 0x190000
    # Poll until buffer is ready
    retry = 0
    while True:
        r = prc.query_resp_list(["llrf_circle_ready"])
        b_status = r[0]
        if b_status & cav_mask == cav_mask:
            break
        retry += 1
        time.sleep(0.003)
    if verbose:
        print "%d retries, status %x" % (retry, r[0])
    # Note the local system time that the buffer became ready
    datetimestr = datetime.datetime.utcnow().isoformat()+"Z"
    # Read array data from hardware
    r = reg_read_value(mg, range(b_base, b_base+npt))
    # Read slow data from hardware
    slow_data = reg_read_value(mg, range(s_base+17, s_base+43))
    # Acknowledge
    prc.query_resp_list(["llrf_circle_ready", ("circle_buf_flip", cav_mask), "llrf_circle_ready"])
    # Convert raw data into a sensible form
    try:
        from read_regmap import get_reg_info
        ri = get_reg_info(prc.regmap, [], "shell_0_circle_data")
        cmoc_circle_dw = ri['data_width']
    except:
        # backup is not robust if data width is changed
        cmoc_circle_dw = 16
    # print("cmoc_circle_dw %d" % cmoc_circle_dw)
    cmoc_circle_dd = 2**cmoc_circle_dw
    r = [x & (cmoc_circle_dd-1) for x in r]
    r = [x-cmoc_circle_dd if x >= cmoc_circle_dd/2 else x for x in r]
    block = sum([[r[ix::chcount]] for ix in range(chcount)], [])
    nblock = numpy.array(block).transpose()
    # use interpret_slow_data() to unpack the 24 octets of aw slow_data
    return (nblock, datetimestr, slow_data)

def usage():
    print 'python iq_test.py -a 192.168.21.12 -c 1'

if __name__ == "__main__":
    import getopt
    import sys
    from prc import c_prc

    opts, args = getopt.getopt(sys.argv[1:], 'ha:p:C:c:n:i:z:', ['help', 'addr=', 'port=',
                                                                 'chcount=', 'count=', 'mask=', 'npt=', 'id=', 'zone='])
    ip_addr = '192.168.21.44'
    port = 50006
    npt_wish = 0
    chcount = 8
    count = 1
    header_id = None
    cav = 0  # a.k.a. zone
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()
        elif opt in ('-a', '--address'):
            ip_addr = arg
        elif opt in ('-p', '--port'):
            port = int(arg)
        elif opt in ('-C', '--chcount'):
            chcount = int(arg)
        elif opt in ('-c', '--count'):
            count = int(arg)
        elif opt in ('-i', '--id'):
            header_id = arg
        elif opt in ('-z', '--zone'):
            cav = int(arg)
    # min3 = port == 3000
    cav_mask = 1 << cav
    prc = c_prc(ip_addr, port)
    print "cav mask", cav_mask
    b_status = prc.query_resp_list(["llrf_circle_ready"])[0]
    npt = 8192  # 2 ** cmoc_circle_aw
    print "starting status", b_status
    if b_status & cav_mask == cav_mask:
        s = prc.query_resp_list(["llrf_circle_ready", ("circle_buf_flip", cav_mask), "llrf_circle_ready"])
        # print s
    for run_n in range(count):
        (nblock, datetimestr, slow_data) = iq_buf_collect(prc, chcount, npt, cav_mask=cav_mask, verbose=True)
        isd = interpret_slow_data(slow_data)
        header = datetimestr + " %d %d" % (isd[0], isd[5])  # circle_count and time_stamp
        if header_id:
            header += "\n" + header_id
        numpy.savetxt('ctl%d_iq_%3.3d' % (cav, run_n), nblock, fmt="%d", header=header)
