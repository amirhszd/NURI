"""
    @author: Eon Rehman
"""

def get_header_file_reflectance(Header_File,Output_Header_File,description_header):

    f = open(Header_File, 'r')
    linelist = f.readlines()
    f.close

    # Re-open file here
    f2 = open(Output_Header_File, 'w')
    for line in linelist:
        line = line.replace([s for s in linelist if "description" in s][0],
                            'description = {[HEADWALL Hyperspec III],' + description_header + '}\r\n')
        line = line.replace('bsq','bip')
        line = line.replace('{49,86,191}','{149,93,38}')
        f2.write(line)
    f2.close()

    return

def get_header_file_radiance(Header_File,Output_Header_File):

    f = open(Header_File, 'r')
    linelist = f.readlines()
    f.close

    # Re-open file here
    f2 = open(Output_Header_File, 'w')
    for line in linelist:
        line = line.replace('{[HEADWALL Hyperspec III],Radiance - Own Calibration}',
                            '{[HEADWALL Hyperspec III],Radiance - Correct Orientation}')
        line = line.replace('bsq','bip')
        line = line.replace('{49,86,191}', '{149,93,38}')
        f2.write(line)
    f2.close()

    return