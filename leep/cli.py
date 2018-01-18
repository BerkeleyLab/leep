
from __future__ import print_function

import logging
_log = logging.getLogger(__name__)

import json, sys, tempfile, shutil

from collections import defaultdict

from . import open

def readwrite(args, dev):
    for pair in args.reg:
        name, _eq, val = pair.partition('=')
        if len(val):
            dev.reg_write([(name, int(val, 0))], instance=args.inst)
        else:
            value, = dev.reg_read((name,), instance=args.inst)
            print("%s \t%08x"%(name, value))


def listreg(args, dev):
    regs = list(dev.regmap)
    regs.sort()
    for reg in regs:
        print(reg)

def dumpaddrs(args, dev):
    regs = []
    for reg, info in dev.regmap.items():
        if 'r' in info.get('access',''):
            regs.append(reg)

    values = dev.reg_read(regs, instance=None)
    addrs = []
    for name, value in zip(regs, values):
        info = dev.get_reg_info(name, instance=None)
        base = info['base_addr']
        if info.get('addr_width', 0)==0:
            # scalar
            addrs.append((base, value & 0xffffffff))
        else:
            # vector
            for pair in enumerate(value & 0xffffffff, base):
                addrs.append(pair)

    # sort by address increasing
    addrs.sort(key=lambda pair:pair[0])

    for addr, value in addrs:
        if value==0 and args.ignore_zeros:
            continue
        print("%08x %08x"%(addr, value))        

def dumpjson(args, dev):
    json.dump(dev.regmap, sys.stdout, indent=2)
    sys.stdout.write('\n')

def gentemplate(args, dev):

    files = defaultdict(list)
    for name, info in dev.regmap.items():
        if len(name)==0:
            _log.warn("Zero length register name")
            continue
        components = {
            'access': info.get('access', ''),
            'type': 'scalar' if info.get('addr_width',0)==0 else 'array',
        }
        values = {
            'name':name,
            'pv':'reg:'+name, # TODO: apply naming convention here
            'size':1<<info.get('addr_width',0),
        }
        values.update(info)

        name = 'feed_reg_%(access)s_%(type)s.template'%components

        files[name].append(values)

    # sort to get stable output order
    files = list(files.items())
    files.sort(key=lambda i:i[0])

    out = tempfile.NamedTemporaryFile('r+')
    out.write('# Generated from\n# FW: %s\n# JSON: %s\n# Code: %s\n\n'%(dev.descript, dev.jsonhash, dev.codehash))

    out.write('file "feed_base.template"\n{\n{PREF="$(P)ctrl:"}\n}\n\n')

    for fname, infos in files:
        out.write('file "%s"\n{\n'%fname)

        infos.sort(key=lambda i:i['pv'])

        for info in infos:
            out.write('{PREF="$(P)%(pv)s",\tREG="%(name)s",\tSIZE="%(size)s"}\n'%info)

        out.write('}\n\n')

    out.flush()
    shutil.copyfile(out.name, args.output)

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('-d','--debug',action='store_const', const=logging.DEBUG, default=logging.INFO)
    P.add_argument('-q','--quiet',action='store_const', const=logging.WARN, dest='debug')
    P.add_argument('-l','--list', action='store_true', help='List register names')
    P.add_argument('-t','--timeout', type=float, default=1.0)
    P.add_argument('dest', metavar="URI", help="Server address.  ca://Prefix or leep://host[:port]")

    SP = P.add_subparsers()

    S = SP.add_parser('reg', help='read/write registers')
    S.set_defaults(func=readwrite)
    S.add_argument('-i','--inst', action='append', default=[])
    S.add_argument('reg', nargs='+', help="register[=newvalue]")

    S = SP.add_parser('list', help='list registers')
    S.set_defaults(func=listreg)

    S = SP.add_parser('dump', help='dump registers')
    S.add_argument('-Z','--ignore-zeros', action='store_true', help="Only print registers with non-zero values")
    S.set_defaults(func=dumpaddrs)

    S = SP.add_parser('json', help='print json')
    S.set_defaults(func=dumpjson)

    S = SP.add_parser('template', help='Generate MSI substitutions file')
    S.set_defaults(func=gentemplate)
    S.add_argument('output', help='Output file')

    return P.parse_args()


def main():
    args = getargs()
    logging.basicConfig(level=args.debug)
    dev = open(args.dest)
    args.func(args, dev)

if __name__=='__main__':
    main()
