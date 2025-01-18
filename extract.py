from lib.iso import ISO9660
from lib.si import SI

with ISO9660("./LEGO_ISLANDI.ISO") as iso:
    si = SI(iso.open("LEGO/SCRIPTS/ACT2/ACT2MAIN.SI"))
    with open("extract/590.wav", "wb") as f:
        f.write(si.object_list()[590].open().read())
